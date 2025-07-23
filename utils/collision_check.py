import torch
from SDF.sdf import SDF

class CollisionCheck:
    def __init__(self, robot, obj_lists, device='cpu'):
        self.robot = robot
        self.obj_lists = obj_lists
        self.device = device
        self.sdf = SDF(robot, device=device)
    
    def is_collision(self, q):
        """
        Check if the robot configuration q is collision-free.
        Args:
            q: Robot configuration (B, num_joints)
        Returns:
            bool: False if collision, True otherwise.
        """
        sdf_value = self.sdf.inference_sdf(q, self.obj_lists)
        return not torch.all(sdf_value > 0.05, dim=-1).item()

    def is_path_collision_free(self, q):
        """
        Check if the entire trajectory is collision-free.
        Args:
            q: Robot configuration (B, num_joints)
        Returns:
            bool: True if collision-free, False otherwise.
        """
        for i in range(q.size(0)):
            if self.is_collision(q[i]):
                return True 
        return False

    def is_segment_collision_free(self, start, end):
        """
        Check if the segment between start and end is in collision.
        Args:
            start: Start configuration (B, num_joints)
            end: End configuration (B, num_joints)
        Returns:
            bool: True if the segment is in collision, False otherwise.
        """
        num_points = 50
        t = torch.linspace(0, 1, num_points, device=self.device).unsqueeze(1)
        segment_points = (1 - t) * start + t * end

        for point in segment_points:
            if self.is_collision(point):
                return True 
        return False
