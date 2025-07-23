from data_generator.RRT_star import RRTStarCSpace
from utils.collision_check import CollisionCheck
from utils.upsample_path import upsample_path
from robot.robot2D import Robot2D
from robot.geometry import Circle


import torch

need_path = 100
each_path_len = 32

all_path = torch.zeros((need_path, each_path_len, 2), dtype=torch.float32)
device = "cpu"
x = torch.tensor([0.0, 0.0], device=device)
robot = Robot2D(num_joints=2, init_states=x.unsqueeze(0), link_length=torch.tensor([[2.0, 2.0]], device=device))
circles = [
    Circle(center=torch.tensor([4.0, 0.0], device=device), radius=0.5),
    Circle(center=torch.tensor([0.0, 4.0], device=device), radius=0.5),
]

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

in_collision = CollisionCheck(robot, circles, device=device).is_collision

for idx in range(need_path):
    print(f"Generating path {idx + 1}/{need_path}...")
    while True:
        q_start = torch.rand(2, device=device) * (2 * torch.pi) - torch.pi
        q_goal = torch.rand(2, device=device) * (2 * torch.pi) - torch.pi
        if not (in_collision(q_start) or in_collision(q_goal)):
            break
        print(f"Collision detected at start {q_start} or goal {q_goal}. Retrying...")
    
    while True:
        path = planner.plan(q_start, q_goal, prune=True, optimize=False)
        if path is not None and len(path) < 32:
            break
    
    path = upsample_path(path, target_len=each_path_len)
    all_path[idx] = path

torch.save(all_path, "data_generator/rrt_star_paths.pt")
print(f"Generated {need_path} paths, each with {each_path_len} waypoints.")