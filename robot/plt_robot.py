import matplotlib.pyplot as plt

def plt_robot(robot, traj, obj_lists, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for c in obj_lists:
        ax.add_patch(c.create_patch(color="red"))
    for q in traj:
        joints = robot.forward_kinematics_all_joints(q.unsqueeze(0))[0]
        ax.plot(joints[0], joints[1], "bo-", alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("Robot Trajectory")
    if ax is None:
        plt.show()

def plt_multiple_robots(robot, trajs, obj_lists, cols=10):

    num_trajs = len(trajs)
    rows = (num_trajs + cols - 1) // cols  

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3)) 
    axes = axes.flatten()  

    for i, traj in enumerate(trajs):
        plt_robot(robot, traj, obj_lists, ax=axes[i])  

    for j in range(len(trajs), len(axes)):
        axes[j].axis("off")
    plt.savefig("robot_trajectories.pdf", bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()

def main():
    from robot.robot2D import Robot2D
    from robot.geometry import Circle
    import torch
    
    device = "cpu"
    x = torch.tensor([0.0, 0.0], device=device)
    robot = Robot2D(num_joints=2, init_states=x.unsqueeze(0), link_length=torch.tensor([[2.0, 2.0]], device=device))
    circles = [
        Circle(center=torch.tensor([4.0, 0.0], device=device), radius=0.5),
        Circle(center=torch.tensor([0.0, 4.0], device=device), radius=0.5),
    ]
    traj = torch.load("data_generator/rrt_star_paths.pt")
    trajs = [traj[i] for i in range(traj.shape[0])]
    plt_multiple_robots(robot, trajs, circles, cols=5)
if __name__ == "__main__":
    main()
    