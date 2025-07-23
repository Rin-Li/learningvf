import torch
from math import log
from typing import Sequence, Optional, List
from utils.collision_check import CollisionCheck


class RRTStarCSpace:
    class _Node:
        __slots__ = ("q", "parent", "cost")

        def __init__(self, q: torch.Tensor, parent=None, cost: float = 0.0):
            self.q = q
            self.parent = parent
            self.cost = cost

    def __init__(
        self,
        joint_limits: Sequence[Sequence[float]],
        robot,
        obj_lists,
        *,
        max_iter: int = 2000,
        step_size: float = 0.3,
        goal_tol: float = 0.1,
        goal_bias: float = 0.1,
        gamma_star: float = 1.5,
        rng: Optional[torch.Generator] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.lims = torch.tensor(joint_limits, dtype=torch.float32, device=device)
        self.dim = self.lims.shape[0]
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_tol = goal_tol
        self.goal_bias = goal_bias
        self.gamma_star = gamma_star
        self.rng = rng or torch.Generator(device=device)
        self.robot = robot
        self.colliion_utils = CollisionCheck(robot, obj_lists, device=device)
        self.in_collision = self.colliion_utils.is_collision
        self.segment_in_collision = self.colliion_utils.is_segment_collision_free
        self.validate_path_collision_free = self.colliion_utils.is_path_collision_free

    def plan(
        self,
        q_start: Sequence[float],
        q_goal: Sequence[float],
        *,
        prune: bool = False,
        optimize: bool = False,
    ) -> Optional[torch.Tensor]:
        raw_path = self._plan_raw(
            torch.as_tensor(q_start, dtype=torch.float32, device=self.device),
            torch.as_tensor(q_goal, dtype=torch.float32, device=self.device),
        )
        if raw_path is None:
            return None
        path = torch.stack(raw_path)
        if prune:
            path = self._prune_path(path)
        if optimize:
            path = self._optimize_path(path)
        if self.validate_path_collision_free(path):
            print("Warning: final path in collision")
            return None
        return path

    def _plan_raw(self, q_start: torch.Tensor, q_goal: torch.Tensor) -> Optional[List[torch.Tensor]]:
        if self.in_collision(q_start) or self.in_collision(q_goal):
            raise ValueError("Start or goal is in collision.")
        nodes: List[RRTStarCSpace._Node] = [self._Node(q_start)]
        best_goal: Optional[RRTStarCSpace._Node] = None
        for it in range(1, self.max_iter + 1):
            q_rand = q_goal.clone() if torch.rand(1, generator=self.rng) < self.goal_bias else self._sample_free()
            node_near = min(nodes, key=lambda n: torch.norm(n.q - q_rand))
            q_new = self._steer(node_near.q, q_rand)
            if self.segment_in_collision(node_near.q, q_new):
                continue
            r_n = min(
                self.gamma_star * (log(it) / it) ** (1.0 / self.dim),
                self.step_size * 2.0,
            )
            neighbor_ids = [
                idx
                for idx, n in enumerate(nodes)
                if torch.norm(n.q - q_new) <= r_n
                and not self.segment_in_collision(n.q, q_new)
            ]
            parent_idx = min(
                neighbor_ids or [nodes.index(node_near)],
                key=lambda idx: nodes[idx].cost + torch.norm(nodes[idx].q - q_new),
            )
            parent_node = nodes[parent_idx]
            new_cost = parent_node.cost + torch.norm(parent_node.q - q_new)
            new_node = self._Node(q_new, parent=parent_node, cost=new_cost)
            nodes.append(new_node)
            for idx in neighbor_ids:
                nbr = nodes[idx]
                cost_through_new = new_node.cost + torch.norm(nbr.q - q_new)
                if cost_through_new < nbr.cost and not self.segment_in_collision(nbr.q, q_new):
                    nbr.parent = new_node
                    nbr.cost = cost_through_new
            if torch.norm(q_new - q_goal) <= self.goal_tol and not self.segment_in_collision(q_new, q_goal):
                g_cost = new_node.cost + torch.norm(q_new - q_goal)
                if best_goal is None or g_cost < best_goal.cost:
                    best_goal = self._Node(q_goal, parent=new_node, cost=g_cost)
                    break
        if best_goal is None:
            return None
        path = []
        node = best_goal
        while node is not None:
            path.append(node.q.clone())
            node = node.parent
        return path[::-1]

    def _sample_free(self) -> torch.Tensor:
        for _ in range(1000):
            u = torch.rand(self.dim, generator=self.rng, device=self.device)
            q = self.lims[:, 0] + (self.lims[:, 1] - self.lims[:, 0]) * u
            if not self.in_collision(q):
                return q
        raise RuntimeError("Sampling failed: configuration space might be fully occupied.")

    def _steer(self, q_from: torch.Tensor, q_to: torch.Tensor) -> torch.Tensor:
        vec = q_to - q_from
        dist = torch.norm(vec)
        if dist <= self.step_size:
            return q_to.clone()
        return q_from + (vec / dist) * self.step_size

    def _prune_path(self, path: torch.Tensor) -> torch.Tensor:
        if path.shape[0] < 3:
            return path
        pruned = [path[0]]
        anchor = path[0]
        for i in range(2, path.shape[0]):
            if self.segment_in_collision(anchor, path[i]):
                pruned.append(path[i - 1])
                anchor = path[i - 1]
        pruned.append(path[-1])
        return torch.stack(pruned)

    def _optimize_path(self, path: torch.Tensor) -> torch.Tensor:
        return path


if __name__ == "__main__":
    from robot.robot2D import Robot2D
    from robot.geometry import Circle
    from SDF.sdf import SDF
    from robot.plt_robot import plt_robot
    from utils.upsample_path import upsample_path

    device = "cpu"
    x = torch.tensor([0.0, 0.0], device=device)
    robot = Robot2D(num_joints=2, init_states=x.unsqueeze(0), link_length=torch.tensor([[2.0, 2.0]], device=device))
    circles = [
        Circle(center=torch.tensor([4.0, 0.0], device=device), radius=0.5),
        Circle(center=torch.tensor([0.0, 4.0], device=device), radius=0.5),
    ]
    sdf = SDF(robot, device=device)
    joint_limits = [(-torch.pi, torch.pi), (-torch.pi, torch.pi)]

    planner = RRTStarCSpace(
        joint_limits,
        max_iter=1500,
        step_size=0.08,
        goal_tol=0.01,
        goal_bias=0.1,
        robot=robot,
        obj_lists=circles,
        device=device,
    )

    q_start = torch.tensor([-2.5, 0.0], device=device)
    q_goal = torch.tensor([2.5, -0.6], device=device)

    path = planner.plan(q_start, q_goal, prune=True, optimize=False)
    if path is None:
        print("No path found.")
    else:
        print(f"Path ({len(path)} way-points):")
        for p in path:
            print(p)
            


    path = upsample_path(path, target_len=50)
        
    plt_robot(robot, path, circles)
