import torch
class Cost:
    def __init__(self, q_f, q_min, q_max):
        '''
        q_f: final goal position (B, num_joints)
        q_min: minimum joint limits (num_joints,)
        q_max: maximum joint limits (num_joints,)
        Initializes the cost function with the final goal position and joint limits.
        '''
        self.qf = q_f
        self.q_min = q_min
        self.q_max = q_max
        self.rest = self.q_min + (self.q_max - self.q_min) * 0.5
        self.gamma = 0.99  # discount factor for future costs
    # def evaluate_costs(self, all_traj, sdf_all):
    #     B, T, _ = all_traj.shape
    #     weights = self.gamma ** torch.arange(T, device=all_traj.device)  # (T,)
        
    #     # goal
    #     goal_err = (all_traj - self.qf).norm(p=2, dim=2)  # (B,T)
    #     goal_cost = (weights * goal_err).sum(dim=1)
        
    #     # collision (smooth)
    #     coll_penalty = torch.relu(-sdf_all)               # (B,T)
    #     collision_cost = (weights * coll_penalty).sum(dim=1)
        
    #     # joint limits
    #     under = torch.relu(self.q_min - all_traj)
    #     over  = torch.relu(all_traj - self.q_max)
    #     jl_penalty = (under + over).pow(2).sum(-1)        # (B,T)
    #     joint_limits_cost = (weights * jl_penalty).sum(dim=1)
        
    #     total_cost = (10 * goal_cost +
    #                 1000 * collision_cost +
    #                 50 * joint_limits_cost)
    #     return total_cost
    def evaluate_costs(self, all_traj, closest_dist_all):
        '''
        Args:
        all_traj: (B, T, num_joints) tensor of trajectories
        closest_dist_all: (B, T) tensor of closest distances to obstacles
        
        Returns:
        total_cost: (B,) tensor of total costs for each trajectory
        '''
        goal_cost = 10*self.goal_cost(all_traj[:, -1, :], self.qf)
        collision_cost = 1000*self.collision_cost(closest_dist_all)
        joint_limits_cost = 50*self.joint_limits_cost(all_traj)
        # stagnation_cost = 10*goal_cost * self.stagnation_cost(all_traj)
        print(f"Goal cost: {goal_cost}, Collision cost: {collision_cost}, Joint limits cost: {joint_limits_cost}")
        
        total_cost = goal_cost + collision_cost + joint_limits_cost
        return total_cost

    def goal_cost(self, traj_end, qf):
        return (traj_end - qf).norm(p=2, dim=1)

    def collision_cost(self, closest_dist_all):
        return (closest_dist_all < 0).sum(dim=1)

    def joint_limits_cost(self, all_traj):
        out_of_bounds = (all_traj < self.q_min) | (all_traj > self.q_max)
        time_step_violation = out_of_bounds.any(dim=2) 

        violation_count = time_step_violation.sum(dim=1)
        return violation_count

    def stagnation_cost(self, all_traj):
        dist = (all_traj[:, 0, :] - all_traj[:, -1, :]).norm(2, dim=1)
        return (1/dist).nan_to_num(0)
