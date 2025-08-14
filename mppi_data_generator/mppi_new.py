import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as mplCircle

from mppi_data_generator.cost import Cost
from robot.robot2D import Robot2D
from robot.geometry import Circle
from SDF.sdf import SDF
from torch.distributions import MultivariateNormal


SEED = 40
np.random.seed(SEED)
torch.manual_seed(SEED)
import random
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def plot_2link_with_obstacle(states,
                              link_lengths=(2, 2),
                              obstacles=None,  # [(center_x, center_y, radius), ...]
                              show_traj=True):
    """
    绘制 2-link 机械臂轨迹与障碍物
    states:      [N, 2] array-like of joint angles (rad)
    link_lengths: tuple (l1, l2)
    obstacles:   list of (center_x, center_y, radius)
    show_traj:   whether to plot the red trajectory line
    """
    l1, l2 = link_lengths
    traj = np.array(states)  # [N,2]

    plt.figure(figsize=(8, 8))
    ax = plt.gca()


    if obstacles:
        for cx, cy, r in obstacles:
            obs_circle = mplCircle((cx, cy), r, color='gray', alpha=0.4)
            ax.add_patch(obs_circle)


    for q in traj:
        theta1, theta2 = q
        j1 = np.array([l1 * math.cos(theta1), l1 * math.sin(theta1)])
        j2 = j1 + np.array([l2 * math.cos(theta1 + theta2),
                            l2 * math.sin(theta1 + theta2)])
        plt.plot([0, j1[0], j2[0]], [0, j1[1], j2[1]], 'o-', alpha=0.3)


    if show_traj:
        plt.plot(traj[:, 0], traj[:, 1], 'r.-')

    plt.axis('equal')
    plt.grid(True)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2-Link Robot Trajectory with Obstacles")
    plt.show()


class SamplingMPPIPlannerTorch:
    def __init__(self, q_ref, obstacle, robot: Robot2D,
                 dt=0.01, H=50, n_samples=200,
                 beta=0.5, gamma=1.0,
                 eta_mean=0.5, eta_cov=0.3,        # 新增：均值/协方差更新步长
                 sigma_init=2.0, sigma_min=1e-3,   # 新增：协方差初始化与下限
                 device='cpu'):
        self.q_ref = torch.tensor(q_ref, dtype=torch.float32, device=device)
        self.dt, self.H, self.n_samples = dt, H, n_samples
        self.beta, self.gamma = beta, gamma
        self.device, self.obs = device, obstacle

        # 策略参数：均值轨迹 U[h,2] 与 协方差轨迹 Sigma[h,2,2]
        self.U = torch.zeros((H, 2), device=device)
        self.Sigma = (sigma_init ** 2) * torch.eye(2, device=device).repeat(H, 1, 1)

        # 更新/数值参数
        self.eta_mean = eta_mean
        self.eta_cov = eta_cov
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self._epsI = 1e-6 * torch.eye(2, device=device)  # 数值稳定项

        self.x_traj = []
        self.sampled_us = []

        self.robot = robot
        self.robot_sdf = SDF(self.robot, device=device)
        q_min = torch.tensor([-torch.pi, -torch.pi], device=device)
        q_max = torch.tensor([ torch.pi,  torch.pi], device=device)
        self.cost_fn = Cost(q_ref, q_min, q_max)

    def dynamics(self, x, u):
        return x + u * self.dt

    def cost(self, x0, u_seq):
        device = self.device
        B = u_seq.shape[0]
        all_q = torch.zeros((B, self.H + 1, 2), device=device)
        all_q[:, 0] = x0
        for h in range(self.H):
            all_q[:, h + 1] = self.dynamics(all_q[:, h], u_seq[:, h])

        q_flat = all_q.reshape(-1, 2)
        sdf_values = self.robot_sdf.inference_sdf(q_flat, self.obs, return_grad=False)
        sdf_all = sdf_values.reshape(B, self.H + 1)

        total_cost = self.cost_fn.evaluate_costs(all_q, sdf_all)
        return total_cost

    def _sample_noises(self):
        # z ~ N(0, I), noise = L * z ; 对每个 h 用各自的 Sigma[h]
        z = torch.randn(self.n_samples, self.H, 2, device=self.device)
        L = torch.linalg.cholesky(self.Sigma)  # [H,2,2]
        noise = torch.einsum('nhi,hij->nhj', z, L)
        return noise


    def sample_controls(self):
        """
        与 SamplingCBF.mppi 一致：
        - 将 U[h,2] 拉平为 [2H]
        - 将 Sigma[h,2,2] 用 block_diag 拼成 [2H,2H]
        - MultivariateNormal 采样 N 条长度 H 的控制序列
        - 统一做 clamp
        返回: u_all [N,H,2]
        """
        H = self.H
        flat_mean = self.U.reshape(-1)  # [2H]
        # block_diag 需要一个列表
        block_cov = torch.block_diag(*[self.Sigma[h] for h in range(H)])  # [2H,2H]
        dist = MultivariateNormal(flat_mean, block_cov)
        samples = dist.sample((self.n_samples,))  # [N,2H]
        u_all = samples.view(self.n_samples, H, 2).clamp_(-3.0, 3.0)
        return u_all

    def update_policy(self, x, u_all):
        """
        与第二段一致：对截断后的 u_all 计算权重、加权均值 mu_hat、
        再对 mu_hat 做带权协方差，最后用 eta_* 指数平滑更新 U 和 Sigma。
        """
        # 代价与权重
        costs = self.cost(x, u_all)                  # [N]
        costs = costs - costs.min()
        w = torch.exp(-self.beta * costs)            # [N]
        w_sum = torch.clamp(w.sum(), min=1e-12)

        # 加权均值
        mu_hat = (w.view(-1,1,1) * u_all).sum(0) / w_sum   # [H,2]
        self.U = (1.0 - self.eta_mean) * self.U + self.eta_mean * mu_hat

        # 带权协方差（以 mu_hat 为中心）
        diff = u_all - mu_hat.unsqueeze(0)                 # [N,H,2]
        cov_w = torch.einsum('n,nhc,nhd->hcd', w, diff, diff) / w_sum  # [H,2,2]

        # 数值稳定 + 指数平滑
        reg = 1e-2
        Sigma_target = cov_w + reg * torch.eye(2, device=self.device)
        self.Sigma = (1.0 - self.eta_cov) * self.Sigma + self.eta_cov * Sigma_target

        # 对角线下限
        diag = torch.diagonal(self.Sigma, dim1=-2, dim2=-1)  # [H,2]
        diag = torch.clamp(diag, min=self.sigma_min**2)
        self.Sigma = self.Sigma.clone()
        self.Sigma[..., 0, 0] = diag[..., 0]
        self.Sigma[..., 1, 1] = diag[..., 1]

        return self.U[0].clone()


    def _shift_policy(self):
        # U、Sigma 一起滚动；末端重置为“静止+默认方差”
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1].zero_()

        self.Sigma = torch.roll(self.Sigma, shifts=-1, dims=0)
        self.Sigma[-1] = (self.sigma_init ** 2) * torch.eye(2, device=self.device)

    def reset(self, x_start):
        self.U.zero_()
        self.Sigma = (self.sigma_init ** 2) * torch.eye(2, device=self.device).repeat(self.H, 1, 1)
        self.x_traj = [torch.tensor(x_start, dtype=torch.float32, device=self.device)]
        self.sampled_us = []

    def step(self):
        x = self.x_traj[-1]
        u_all = self.sample_controls()          # 不再接收 noise
        self.sampled_us.append(u_all)
        u0 = self.update_policy(x, u_all)       # 不传 noise
        x_next = self.dynamics(x, u0)
        self.x_traj.append(x_next)
        self._shift_policy()                    # H>1 时做 receding-horizon
        return x_next

    def get_trajectory(self):
        return torch.stack(self.x_traj)
    

def joint_path_length_from_list(x_traj_list):
    """
    x_traj_list: [q0, q1, ...]，元素是 1x2 的 torch.Tensor
    返回：关节空间折线长度之和（float）
    """
    if len(x_traj_list) < 2:
        return 0.0
    arr = torch.stack(x_traj_list).detach().cpu().numpy()  # [T,2]
    diffs = np.diff(arr, axis=0)                            # [T-1,2]
    return float(np.linalg.norm(diffs, axis=1).sum())

def run_experiment_from_files(start_arr: np.ndarray,
                              goal_arr:  np.ndarray,
                              circles,
                              robot: Robot2D,
                              device: str = "cpu",
                              max_steps: int = 500,
                              tol: float = 0.1,
                              verbose_every: int = 10,
                              joint_min: float = -math.pi,
                              joint_max: float =  math.pi):
    """
    用给定的 (start_i, goal_i) 对逐个运行 MPPI 实验。
    新增失败判定：
      - 关节限制：任一关节越界 -> joint_limit
      - 碰撞：SDF < 0 -> sdf_collision
    统计每条轨迹的关节空间长度，并区分全部/成功/失败三类。
    """
    assert len(start_arr) == len(goal_arr), "start 与 goal 数量不一致"
    num_trials = len(start_arr)

    success_steps = []            # 仅成功 trial 的步数
    path_lengths_all = []         # 所有 trial 的长度
    path_lengths_succ = []        # 成功 trial 的长度
    path_lengths_fail = []        # 失败 trial 的长度

    for i in range(num_trials):
        q_start = torch.tensor(start_arr[i], dtype=torch.float32, device=device)
        q_goal  = torch.tensor(goal_arr[i],  dtype=torch.float32, device=device)

        planner = SamplingMPPIPlannerTorch(
            q_ref=q_goal, obstacle=circles, robot=robot, device=device
        )
        planner.reset(q_start)

        reached = False
        failed  = False
        fail_reason = None

        def _check_limits_and_sdf(q_tensor: torch.Tensor):
            if (q_tensor < joint_min).any() or (q_tensor > joint_max).any():
                return False, "joint_limit"
            sdf_val = planner.robot_sdf.inference_sdf(
                q_tensor.view(1, -1), planner.obs, return_grad=False
            ).item()
            if sdf_val < 0.0:
                return False, "sdf_collision"
            return True, None

        ok, reason = _check_limits_and_sdf(planner.x_traj[-1])
        if not ok:
            failed = True
            fail_reason = reason

        t = 0
        while (not reached) and (not failed) and (t < max_steps):
            x_next = planner.step()
            if torch.norm(x_next - q_goal) < tol:
                reached = True
                break
            ok, reason = _check_limits_and_sdf(x_next)
            if not ok:
                failed = True
                fail_reason = reason
                break
            t += 1

        steps_used = len(planner.x_traj)  # 含初始
        # 记录该 trial 的关节路径长度（无论成功/失败都记录）
        L = joint_path_length_from_list(planner.x_traj)
        path_lengths_all.append(L)

        msg = (f"Start: {q_start.detach().cpu().numpy().flatten()}, "
               f"Goal: {q_goal.detach().cpu().numpy().flatten()}, ")
        if reached:
            print(msg + f"Reached: True, Steps: {steps_used}, PathLen: {L:.4f}")
            success_steps.append(steps_used - 1)
            path_lengths_succ.append(L)
        else:
            if failed and fail_reason is not None:
                print(msg + f"Reached: False, Steps: {steps_used}, FailReason: {fail_reason}, PathLen: {L:.4f}")
            else:
                print(msg + f"Reached: False, Steps: {steps_used}, FailReason: timeout, PathLen: {L:.4f}")
            path_lengths_fail.append(L)

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"[{i + 1}/{num_trials}]  成功 {len(success_steps)} 次")

    # 汇总统计
    success_cnt  = len(success_steps)
    avg_steps    = float(np.mean(success_steps)) if success_steps else float('nan')
    avg_len_all  = float(np.mean(path_lengths_all))  if path_lengths_all  else float('nan')
    avg_len_succ = float(np.mean(path_lengths_succ)) if path_lengths_succ else float('nan')
    avg_len_fail = float(np.mean(path_lengths_fail)) if path_lengths_fail else float('nan')

    print("--------------------------------------------------")
    print(f"总实验次数           : {num_trials}")
    print(f"成功到达次数         : {success_cnt}")
    print(f"成功率               : {success_cnt / num_trials:.2%}")
    print(f"平均步数(仅成功)     : {avg_steps:.2f}")
    print(f"平均路径长度(全部)   : {avg_len_all:.4f}")
    print(f"平均路径长度(成功)   : {avg_len_succ:.4f}")
    print(f"平均路径长度(失败)   : {avg_len_fail:.4f}")
    print("--------------------------------------------------")

    # 返回与上面一致的结构（三类长度均返回）
    return success_cnt, avg_steps, path_lengths_all, path_lengths_succ, path_lengths_fail


# def main():
#     device = "cpu"
#     x = torch.tensor([0.0, 0.0], device=device)
#     robot = Robot2D(num_joints=2, init_states=x.unsqueeze(0),
#                     link_length=torch.tensor([[2.0, 2.0]], device=device))

#     circles = [
#         Circle(center=torch.tensor([0.0, 2.45], device=device), radius=0.3),
#         Circle(center=torch.tensor([2.3, -2.3], device=device), radius=0.3),
#     ]

#     q_start = torch.tensor([ 2.1651123,  1.2108364 ], device=device)
#     q_goal  = torch.tensor([-2.1001303, -0.88734066], device=device)

#     planner = SamplingMPPIPlannerTorch(q_goal, circles, robot, device=device)
#     planner.reset(q_start)

#     max_steps = 5000
#     for idx in range(max_steps):
#         print(f"Step {idx + 1}/{max_steps}...")
#         x_next = planner.step()
#         print(f"Current state: {x_next.cpu().numpy()}")
#         if torch.norm(x_next - q_goal) < 0.1:
#             print("Reached goal!")
#             break

#     traj = planner.get_trajectory().cpu().numpy()
#     print("Trajectory shape:", traj.shape)


#     obs_list = [(float(c.center[0]), float(c.center[1]), float(c.radius)) for c in circles]
#     np.save("/Users/yulinli/Desktop/Exp/sampling_CBF/2Dexamples/mppi_original_1.npy", traj)
#     plot_2link_with_obstacle(traj, link_lengths=(2, 2), obstacles=obs_list, show_traj=True)
    
#     start = np.load("/Users/yulinli/Desktop/Exp/sampling_CBF/2Dexamples/start.npy", allow_pickle=True)
#     goal = np.load("/Users/yulinli/Desktop/Exp/sampling_CBF/2Dexamples/goal.npy", allow_pickle=True)


def main():
    device = "cpu"
    x = torch.tensor([0.0, 0.0], device=device)
    robot = Robot2D(num_joints=2, init_states=x.unsqueeze(0),
                    link_length=torch.tensor([[2.0, 2.0]], device=device))
    circles = [
        Circle(center=torch.tensor([0.0, 2.45], device=device), radius=0.3),
        Circle(center=torch.tensor([2.3, -2.3], device=device), radius=0.3),
    ]

    # 加载给定的 start / goal
    start = np.load("/Users/yulinli/Desktop/Exp/sampling_CBF/2Dexamples/start.npy", allow_pickle=True)
    goal  = np.load("/Users/yulinli/Desktop/Exp/sampling_CBF/2Dexamples/goal.npy",  allow_pickle=True)
    start = np.asarray(start)  # 确保是 [N,2]
    goal  = np.asarray(goal)

    run_experiment_from_files(start, goal, circles, robot, device=device,
                              max_steps=5000, tol=0.1, verbose_every=10)


if __name__ == "__main__":
    main()
