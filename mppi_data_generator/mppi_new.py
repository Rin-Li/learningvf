import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as mplCircle

from mppi_data_generator.cost import Cost
from robot.robot2D import Robot2D
from robot.geometry import Circle
from SDF.sdf import SDF


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
                 device='cpu'):
        self.q_ref = torch.tensor(q_ref, dtype=torch.float32, device=device)
        self.dt = dt
        self.H = H
        self.n_samples = n_samples
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.obs = obstacle

        self.U = torch.zeros((H, 2), device=device)
        sigma_init = 2.0
        self.Sigma = (sigma_init ** 2) * torch.eye(2, device=device).repeat(H, 1, 1)

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
        z = torch.randn(self.n_samples, self.H, 2, device=self.device)
        L = torch.linalg.cholesky(self.Sigma)
        noise = torch.einsum('nhd,hde->nhe', z, L)
        return noise

    def sample_controls(self):
        noise = self._sample_noises()
        u_all = self.U.unsqueeze(0) + noise
        return u_all, noise

    def update_policy(self, x, u_all, noise):
        costs = self.cost(x, u_all)
        costs = costs - costs.min()
        w = torch.exp(-self.beta * costs)
        w_sum = torch.clamp(w.sum(), min=1e-12)
        w_norm = (w / w_sum).view(-1, 1, 1)
        delta_U = torch.sum(w_norm * noise, dim=0)
        self.U = self.U + delta_U
        return self.U[0].clone()

    def _shift_U(self):
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1].zero_()

    def reset(self, x_start):
        self.U.zero_()
        sigma_init = 2.0
        self.Sigma = (sigma_init ** 2) * torch.eye(2, device=self.device).repeat(self.H, 1, 1)
        self.x_traj = [torch.tensor(x_start, dtype=torch.float32, device=self.device)]
        self.sampled_us = []

    def step(self):
        x = self.x_traj[-1]
        u_all, noise = self.sample_controls()
        self.sampled_us.append(u_all)
        u0 = self.update_policy(x, u_all, noise)
        x_next = self.dynamics(x, u0)
        self.x_traj.append(x_next)
        self._shift_U()
        return x_next

    def get_trajectory(self):
        return torch.stack(self.x_traj)


def main():
    device = "cpu"
    x = torch.tensor([0.0, 0.0], device=device)
    robot = Robot2D(num_joints=2, init_states=x.unsqueeze(0),
                    link_length=torch.tensor([[2.0, 2.0]], device=device))

    circles = [
        Circle(center=torch.tensor([0.0, 2.45], device=device), radius=0.3),
        Circle(center=torch.tensor([2.3, -2.3], device=device), radius=0.3),
    ]

    q_start = torch.tensor([ 2.1651123,  1.2108364 ], device=device)
    q_goal  = torch.tensor([-2.1001303, -0.88734066], device=device)

    planner = SamplingMPPIPlannerTorch(q_goal, circles, robot, device=device)
    planner.reset(q_start)

    max_steps = 300
    for idx in range(max_steps):
        print(f"Step {idx + 1}/{max_steps}...")
        x_next = planner.step()
        print(f"Current state: {x_next.cpu().numpy()}")
        if torch.norm(x_next - q_goal) < 0.1:
            print("Reached goal!")
            break

    traj = planner.get_trajectory().cpu().numpy()
    print("Trajectory shape:", traj.shape)


    obs_list = [(float(c.center[0]), float(c.center[1]), float(c.radius)) for c in circles]
    plot_2link_with_obstacle(traj, link_lengths=(2, 2), obstacles=obs_list, show_traj=True)


if __name__ == "__main__":
    main()
