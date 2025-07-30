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
    def evaluate_costs(self, all_traj, closest_dist_all):
        '''
        Args:
        all_traj: (B, T, num_joints) tensor of trajectories
        closest_dist_all: (B, T) tensor of closest distances to obstacles
        
        Returns:
        total_cost: (B,) tensor of total costs for each trajectory
        '''
        goal_cost = 10*self.goal_cost(all_traj[:, -1, :], self.qf)
        collision_cost = 100*self.collision_cost(closest_dist_all)
        joint_limits_cost = 100*self.joint_limits_cost(all_traj)
        stagnation_cost = 10*goal_cost * self.stagnation_cost(all_traj)

        total_cost = goal_cost + collision_cost + joint_limits_cost + stagnation_cost 
        return total_cost

    def goal_cost(self, traj_end, qf):
        return (traj_end - qf).norm(p=2, dim=1)

    def collision_cost(self, closest_dist_all):
        return (closest_dist_all < 0).sum(dim=1)

    def joint_limits_cost(self, all_traj):
        mask = (all_traj < self.q_min).sum(dim=1) + (all_traj > self.q_max).sum(dim=1)
        mask = mask.sum(dim=1)
        return (mask > 0) + 0

    def stagnation_cost(self, all_traj):
        dist = (all_traj[:, 0, :] - all_traj[:, -1, :]).norm(2, dim=1)
        return (1/dist).nan_to_num(0)

    def rest_cost(self, all_traj):
        return (all_traj - self.rest).norm(2, dim=2).sum(dim=1)