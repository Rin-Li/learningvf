import matplotlib.pyplot as plt

def plt_robot(robot, traj, obj_lists):
    
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
    plt.show()