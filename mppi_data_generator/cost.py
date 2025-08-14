# import torch
# class Cost:
#     def __init__(self, q_f, q_min, q_max):
#         '''
#         q_f: final goal position (B, num_joints)
#         q_min: minimum joint limits (num_joints,)
#         q_max: maximum joint limits (num_joints,)
#         Initializes the cost function with the final goal position and joint limits.
#         '''
#         self.qf = q_f
#         self.q_min = q_min
#         self.q_max = q_max
#         self.rest = self.q_min + (self.q_max - self.q_min) * 0.5
#         self.gamma = 0.99  # discount factor for future costs
#     # def evaluate_costs(self, all_traj, sdf_all):
#     #     B, T, _ = all_traj.shape
#     #     weights = self.gamma ** torch.arange(T, device=all_traj.device)  # (T,)
        
#     #     # goal
#     #     goal_err = (all_traj - self.qf).norm(p=2, dim=2)  # (B,T)
#     #     goal_cost = (weights * goal_err).sum(dim=1)
        
#     #     # collision (smooth)
#     #     coll_penalty = torch.relu(-sdf_all)               # (B,T)
#     #     collision_cost = (weights * coll_penalty).sum(dim=1)
        
#     #     # joint limits
#     #     under = torch.relu(self.q_min - all_traj)
#     #     over  = torch.relu(all_traj - self.q_max)
#     #     jl_penalty = (under + over).pow(2).sum(-1)        # (B,T)
#     #     joint_limits_cost = (weights * jl_penalty).sum(dim=1)
        
#     #     total_cost = (10 * goal_cost +
#     #                 1000 * collision_cost +
#     #                 50 * joint_limits_cost)
#     #     return total_cost
#     def evaluate_costs(self, all_traj, closest_dist_all):
#         '''
#         Args:
#         all_traj: (B, T, num_joints) tensor of trajectories
#         closest_dist_all: (B, T) tensor of closest distances to obstacles
        
#         Returns:
#         total_cost: (B,) tensor of total costs for each trajectory
#         '''
#         goal_cost = 10*self.goal_cost(all_traj[:, -1, :], self.qf)
#         collision_cost = 50*self.collision_cost(closest_dist_all)
#         joint_limits_cost = 50*self.joint_limits_cost(all_traj)
#         # stagnation_cost = 10*goal_cost * self.stagnation_cost(all_traj)
        
#         total_cost = goal_cost + collision_cost + joint_limits_cost
#         return total_cost

#     def goal_cost(self, traj_end, qf):
#         return (traj_end - qf).norm(p=2, dim=1)

#     def collision_cost(self, closest_dist_all):
#         return (closest_dist_all < 0).sum(dim=1)

#     def joint_limits_cost(self, all_traj):
#         out_of_bounds = (all_traj < self.q_min) | (all_traj > self.q_max)
#         time_step_violation = out_of_bounds.any(dim=2) 

#         violation_count = time_step_violation.sum(dim=1)
#         return violation_count

#     def stagnation_cost(self, all_traj):
#         dist = (all_traj[:, 0, :] - all_traj[:, -1, :]).norm(2, dim=1)
#         return (1/dist).nan_to_num(0)


import torch

class Cost:
    def __init__(self, q_f, q_min, q_max):
        """
        Initialize the cost function for MPPI.

        Args:
            q_f: Final desired joint configuration (goal state), shape (dof,)
            q_min: Lower joint limits, shape (dof,)
            q_max: Upper joint limits, shape (dof,)
        """
        self.qf = q_f
        self.q_min = q_min
        self.q_max = q_max
        # "Rest" (neutral) configuration: midpoint between min and max limits
        self.rest = self.q_min + (self.q_max - self.q_min) * 0.5

    def evaluate_costs(self, all_traj, closest_dist_all):
        """
        Evaluate the total cost for each trajectory.

        Args:
            all_traj: All sampled trajectories, shape (B, T, dof)
            closest_dist_all: Closest obstacle distances for each state in the trajectory, shape (B, T)

        Returns:
            total_cost: Vector of total costs for each trajectory, shape (B,)
        """
        # Goal-reaching cost (distance to goal at final timestep)
        goal_cost = 10.0 * self.goal_cost(all_traj[:, -1, :], self.qf)

        # Collision cost (penalize penetration into obstacles)
        collision_cost = 100.0 * self.collision_cost(closest_dist_all)

        # Stagnation cost (penalize trajectories that barely move)
        stagnation_cost = 10.0 * goal_cost * self.stagnation_cost(all_traj)

        # Joint limit cost (penalize exceeding joint limits)
        joint_limits_cost = 100.0 * self.joint_limits_cost(all_traj)

        # Sum all cost components
        total_cost = goal_cost + collision_cost + joint_limits_cost + stagnation_cost
        return total_cost

    def goal_cost(self, traj_end, qf):
        """
        Compute Euclidean distance in joint space between the final state
        of the trajectory and the goal configuration.

        Args:
            traj_end: Final state of each trajectory, shape (B, dof)
            qf: Goal state, shape (dof,)

        Returns:
            Goal distance cost, shape (B,)
        """
        return (traj_end - qf).norm(p=2, dim=1)

    def collision_cost(self, closest_dist_all):
        """
        Penalize obstacle collisions using signed distance field (SDF) values.
        Only negative distances (penetration) contribute to the cost.

        Args:
            closest_dist_all: Closest distances to obstacles for each state, shape (B, T)

        Returns:
            Collision cost, shape (B,)
        """
        return torch.relu(-closest_dist_all).sum(dim=1)

    def joint_limits_cost(self, all_traj):
        """
        Penalize exceeding joint limits with a smooth squared penalty.

        Args:
            all_traj: Trajectories, shape (B, T, dof)

        Returns:
            Joint limit cost, shape (B,)
        """
        under = torch.relu(self.q_min - all_traj)  # Below lower limit
        over  = torch.relu(all_traj - self.q_max)  # Above upper limit
        return (under + over).pow(2).sum(dim=(1, 2))

    def stagnation_cost(self, all_traj):
        """
        Penalize trajectories that move very little from start to end.

        Args:
            all_traj: Trajectories, shape (B, T, dof)

        Returns:
            Stagnation cost, shape (B,)
        """
        dist = (all_traj[:, 0, :] - all_traj[:, -1, :]).norm(2, dim=1)
        # Avoid division by zero and limit the penalty magnitude
        return torch.clamp(1.0 / (dist + 1e-3), max=10.0)
