#!/usr/bin/env python
#
# MIT License

import torch
from mppi_data_generator.cost import Cost
from robot.robot2D import Robot2D
from robot.geometry import Circle
from SDF.sdf import SDF

class SamplingMPPIPlannerTorch:
    def __init__(self, q_ref, obstacle: Circle, robot: Robot2D, dt=0.01, H=50, n_samples=200,
                 beta=2.0, gamma=1.0, alpha_mean=0.3, alpha_cov=0.8, device='cpu'):
        self.q_ref = torch.tensor(q_ref, dtype=torch.float32, device=device)
        self.dt, self.H, self.n_samples = dt, H, n_samples
        self.beta, self.gamma = beta, gamma
        self.alpha_mean, self.alpha_cov = alpha_mean, alpha_cov
        self.device, self.obs, self.robot = device, obstacle, robot
        self.mean = torch.zeros((H, 2), device=device)
        self.cov = torch.eye(2, device=device).repeat(H, 1, 1)
        self.x_traj, self.sampled_us = [], []
        self.robot_sdf = SDF(self.robot, device=device)
        q_min = torch.tensor([-torch.pi, -torch.pi], device=device)
        q_max = torch.tensor([torch.pi, torch.pi], device=device)
        self.cost_fn = Cost(q_ref, q_min, q_max)
        self.last_best_u = None

    def dynamics(self, x, u):
        return x + u * self.dt

    def cost(self, x0, u_seq):
        if x0.ndim == 1:
            x0 = x0.expand(self.n_samples, -1)
        cumu = torch.cumsum(u_seq, dim=1) * self.dt
        all_q = torch.cat([x0.unsqueeze(1), x0.unsqueeze(1) + cumu], dim=1)
        q_flat = all_q.reshape(-1, 2)
        sdf_all = self.robot_sdf.inference_sdf(q_flat, self.obs, return_grad=False).view(self.n_samples, self.H + 1)
        return self.cost_fn.evaluate_costs(all_q, sdf_all)

    def sample_controls(self):
        chol = torch.linalg.cholesky(self.cov)
        eps = torch.randn(self.n_samples, self.H, 2, device=self.device)
        return self.mean.unsqueeze(0) + torch.einsum('bhi,hij->bhj', eps, chol)

    def update_policy(self, x, u_all):
        costs = self.cost(x, u_all)
        w = torch.softmax(-costs / self.beta, dim=0)
        mu_new = (w[:, None, None] * u_all).sum(0)
        diff = u_all - mu_new.unsqueeze(0)
        Sigma_new = torch.einsum('b,bhi,bhj->hij', w, diff, diff)
        self.mean = (1 - self.alpha_mean) * self.mean + self.alpha_mean * mu_new
        # self.cov = (1 - self.alpha_cov) * self.cov + self.alpha_cov * Sigma_new
        # Exploration term
        self.cov += 1e-2 * torch.eye(2, device=self.device).repeat(self.H, 1, 1)
        return self.mean[0], costs

    def reset(self, x_start):
        self.mean.zero_()
        sigma_init = 2.0
        self.cov = (sigma_init ** 2) * torch.eye(2, device=self.device).repeat(self.H, 1, 1)
        self.x_traj = [torch.tensor(x_start, dtype=torch.float32, device=self.device)]
        self.sampled_us = []
        self.last_best_u = None

    def step(self, step_idx=None):
        x = self.x_traj[-1]
        u_all = self.sample_controls()
        self.sampled_us.append(u_all)
        best_u, costs = self.update_policy(x, u_all)
        x_next = self.dynamics(x, best_u)
        self.x_traj.append(x_next)

        # Debug prints
        goal_dir = (self.q_ref - x).cpu().numpy()
        cov_diag_mean = torch.mean(torch.diagonal(self.cov, dim1=-2, dim2=-1)).item()
        print(f"[Step {step_idx if step_idx is not None else len(self.x_traj)-1}]")
        print(f"  Best control: {best_u.cpu().numpy()}")
        print(f"  Goal direction: {goal_dir}")
        print(f"  Cov diag mean: {cov_diag_mean:.6f}")
        print(f"  Avg cost: {costs.mean().item():.6f}")

        if self.last_best_u is not None and torch.allclose(best_u, self.last_best_u, atol=1e-3):
            print("  ⚠ Best_u unchanged → Possible covariance collapse")

        self.last_best_u = best_u.clone()
        return x_next

    def get_trajectory(self):
        return torch.stack(self.x_traj)


def main():
    from robot.plt_robot import plt_robot
    device = 'cpu'
    robot = Robot2D(num_joints=2, init_states=torch.tensor([[0.0, 0.0]], device=device), link_length=torch.tensor([[2.0, 2.0]], device=device))
    circles = [Circle(center=torch.tensor([0.0, 2.45], device=device), radius=0.3), Circle(center=torch.tensor([2.3, -2.3], device=device), radius=0.3)]
    q_start = torch.tensor([2.1651123, 1.2108364], device=device)
    q_goal = torch.tensor([-2.1001303, -0.88734066], device=device)
    planner = SamplingMPPIPlannerTorch(q_goal, circles, robot, device=device)
    planner.reset(q_start)
    for _ in range(300):
        x_next = planner.step()
        if torch.norm(x_next - q_goal) < 0.1:
            break
    traj = planner.get_trajectory().cpu()
    plt_robot(robot, traj, circles)

if __name__ == '__main__':
    main()
