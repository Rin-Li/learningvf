import torch
from mppi_data_generator.cost import Cost
from robot.robot2D import Robot2D
from robot.geometry import Circle
from SDF.sdf import SDF

class SamplingMPPIPlannerTorch:
    def __init__(self, q_ref, obstacle : Circle, robot : Robot2D, dt=0.1, H=20, n_samples=200,
                 beta=1.0, gamma=0.98, alpha_mean=0.1, alpha_cov=0.1, device='cpu'):
        self.q_ref = torch.tensor(q_ref, dtype=torch.float32, device=device)
        self.dt = dt
        self.H = H
        self.n_samples = n_samples
        self.beta = beta
        self.gamma = gamma
        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov
        self.device = device
        self.obs = obstacle

        self.mean = torch.zeros((H, 2), device=device)
        self.cov = torch.eye(2, device=device).repeat(H, 1, 1)

        self.x_traj = []
        self.sampled_us = []
        
        self.robot = robot
        self.robot_sdf = SDF(self.robot, device=device)
        q_min = torch.tensor([-torch.pi, -torch.pi], device=device)
        q_max = torch.tensor([torch.pi, torch.pi], device=device)
        self.cost_fn = Cost(q_ref, q_min, q_max)

    def dynamics(self, x, u):
        return x + u * self.dt

    def cost(self, x0, u_seq):
        '''
        Args: 
            x0: Initial state (B, 2)
            u_seq: Control sequence (B, H, 2)
            obs: List of obstacles (e.g., list of Circle objects)
        Returns:
            total_cost: (B,) tensor of total cost for each trajectory
        '''
        device = self.device

        # Initialize all_q with shape (B, H+1, 2)
        all_q = torch.zeros((self.n_samples, self.H + 1, 2), device=device)
        all_q[:, 0] = x0

        # Rollout trajectories
        for h in range(self.H):
            all_q[:, h + 1] = self.dynamics(all_q[:, h], u_seq[:, h])

        # Flatten all_q to shape (B*(H+1), 2) for SDF inference
        q_flat = all_q.reshape(-1, 2)
        sdf_values = self.robot_sdf.inference_sdf(q_flat, self.obs, return_grad=False)
        
        # Reshape back to (B, H+1)
        sdf_all = sdf_values.reshape(self.n_samples, self.H + 1)
        
        weights = self.gamma ** torch.arange(self.H+1, device=device)  # (H+1,)
        sdf_all = sdf_all * weights  # broadcasting

        # Evaluate cost
        total_cost = self.cost_fn.evaluate_costs(all_q, sdf_all)
        return total_cost

    def sample_controls(self):
        us = torch.zeros((self.n_samples, self.H, 2), device=self.device)
        for i in range(self.n_samples):
            for h in range(self.H):
                dist = torch.distributions.MultivariateNormal(self.mean[h], self.cov[h])
                us[i, h] = dist.sample()
        return us

    def update_policy(self, x, u_all):
        '''
        Update the policy using the sampled controls and the cost function.
        Args: 
            x: Current state (B, 2)
            obs: Obstacles (N, 2)
        Returns:
            mu_new: Updated mean control sequence (H, 2)
        '''
        costs = self.cost(x, u_all)
        
        w = torch.exp(-self.beta * (costs - costs.min()))
        w_sum = w.sum()

        mu_new = torch.zeros_like(self.mean)
        Sigma_new = torch.zeros_like(self.cov)

        for h in range(self.H):
            weighted_u = (w[:, None] * u_all[:, h]).sum(dim=0) / w_sum
            mu_new[h] = (1 - self.alpha_mean) * self.mean[h] + self.alpha_mean * weighted_u

            diff = u_all[:, h] - mu_new[h]
            cov_h = torch.einsum('i,ij,ik->jk', w, diff, diff) / w_sum
            Sigma_new[h] = (1 - self.alpha_cov) * self.cov[h] + self.alpha_cov * cov_h
            Sigma_new[h] += 1e-6 * torch.eye(2, device=self.device)

        self.mean = mu_new
        self.cov = Sigma_new
        return mu_new[0]

    def reset(self, x_start):
        self.mean = torch.zeros((self.H, 2), device=self.device)
        self.cov = torch.eye(2, device=self.device).repeat(self.H, 1, 1)
        self.x_traj = [torch.tensor(x_start, dtype=torch.float32, device=self.device)]
        self.sampled_us = []

    def step(self):
        x = self.x_traj[-1]
        u_all = self.sample_controls()
        self.sampled_us.append(u_all)

        best_u = self.update_policy(x, u_all)
        x_next = self.dynamics(x, best_u)
        self.x_traj.append(x_next)
        return x_next

    def get_trajectory(self):
        return torch.stack(self.x_traj)

def main():
    from robot.plt_robot import plt_robot
    device = "cpu"
    x = torch.tensor([0.0, 0.0], device=device)


    robot = Robot2D(num_joints=2, init_states=x.unsqueeze(0),
                    link_length=torch.tensor([[2.0, 2.0]], device=device))


    circles = [
        Circle(center=torch.tensor([4.0, 0.0], device=device), radius=0.5),
        Circle(center=torch.tensor([0.0, 4.0], device=device), radius=0.5),
    ]


    q_start = torch.tensor([-2.5, 0.0], device=device)
    q_goal = torch.tensor([2.5, -0.6], device=device)


    planner = SamplingMPPIPlannerTorch(q_goal, circles, robot, device=device)


    planner.reset(q_start)


    max_steps = 300
    for idx in range(max_steps):
        print(f"Step {idx + 1}/{max_steps}...")


        x_next = planner.step()
        print(f"Current state: {x_next.cpu().numpy()}")

        if torch.norm(x_next - q_goal) < 0.05:
            print("Reached goal!")
            break


    traj = planner.get_trajectory().cpu()
    print("Trajectory shape:", traj.shape)
    plt_robot(robot, traj, circles)

if __name__ == "__main__":
    main()