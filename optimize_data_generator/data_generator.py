# -*- coding: utf-8 -*-
"""
Single-step MPC demo for a 2-DoF planar robot using CasADi + SDF.
Run with:  python3 -m optimize_data_generator.data_generator
"""
import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt

from robot.robot2D import Robot2D
from robot.geometry import Circle
from SDF.sdf import SDF


def solve_optimization_problem_for_robot2D(
    x0: torch.Tensor,
    x_goal: torch.Tensor,
    sdf_value: torch.Tensor,
    sdf_grad: torch.Tensor,
    dt: float = 0.1,
    cons_q: float = np.pi,
    cons_u: float = 1.0,
    safety_buffer: float = 0.1,
    solver: str = "ipopt",
):
    """
    Args
    ----
      x0, x_goal : shape (n_q,)  -- Torch tensors on CPU
      sdf_value  : (),  sdf_grad : (n_q,)   (Torch)
    Returns
    -------
      q_opt : (n_q, 2)   first col = q0, second col = q1
      u_opt : (n_q, 1)
    """

    n_q = x0.numel()
    x0_np     = x0.detach().cpu().numpy().astype(float).flatten()
    x_goal_np = x_goal.detach().cpu().numpy().astype(float).flatten()
    grad_np   = sdf_grad.detach().cpu().numpy().reshape(1, -1).astype(float)
    safe_val  = max(float(sdf_value), 1e-6)  

    Q = ca.MX.sym("Q", n_q, 2)   # q0, q1
    U = ca.MX.sym("U", n_q, 1)   # u0


    w_q = 10.0 * ca.DM_eye(n_q)
    w_u = 0.1 * ca.DM_eye(n_q)
    delta_q = Q[:, 1] - ca.DM(x_goal_np)
    cost = ca.mtimes([delta_q.T, w_q, delta_q]) + ca.mtimes([U[:, 0].T, w_u, U[:, 0]])


    g, lbg, ubg = [], [], []


    g  += [Q[:, 0] - ca.DM(x0_np)]
    lbg += [0]*n_q
    ubg += [0]*n_q

    # q1 = q0 + u0*dt
    g  += [Q[:, 1] - (Q[:, 0] + U[:, 0]*dt)]
    lbg += [0]*n_q
    ubg += [0]*n_q

    # 3)-grad·u*dt <= log(d + buffer)
    avoid = -ca.mtimes(ca.DM(grad_np), U) * dt   # 标量 (1×1)
    g  += [avoid]
    lbg += [-ca.inf]
    ubg += [np.log(safe_val + safety_buffer)]


    vars_ = ca.vertcat(Q.reshape((-1, 1)), U) 
    nlp   = {"x": vars_, "f": cost, "g": ca.vertcat(*g)}
    opts  = {"print_time": False, "ipopt.print_level": 0}
    solver_fn = ca.nlpsol("nlp", solver, nlp, opts)

    lbx = [-cons_q]*(2*n_q) + [-cons_u]*n_q
    ubx = [ cons_q]*(2*n_q) + [ cons_u]*n_q
    sol  = solver_fn(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)["x"].full().ravel()

    q_opt = sol[:2*n_q].reshape(n_q, 2)
    u_opt = sol[2*n_q:].reshape(n_q, 1)
    return q_opt, u_opt



def main():

    x = torch.tensor([-1.6, -0.75])  
    x_goal = torch.tensor([1.5, 1.2])


    robot = Robot2D(num_joints=2,
                    init_states=x.unsqueeze(0),
                    link_length=torch.tensor([[2.0, 2.0]])
                    )
    circles = [
        Circle(center=torch.tensor([4.0, 0.0]), radius=0.5),
        Circle(center=torch.tensor([0.0, 4.0]), radius=0.5),
    ]
    sdf = SDF(robot, device='cpu')

    dt, max_steps = 0.05, 100
    traj = [x.clone()]
    for step in range(max_steps):
        d_val, d_grad = sdf.inference_sdf(x.unsqueeze(0), circles, return_grad=True)
        print(f"Step {step}, SDF Value: {d_val.item()}, Gradient: {d_grad.squeeze().numpy()}")
        q_opt, u_opt = solve_optimization_problem_for_robot2D(
            x0 = x,
            x_goal = x_goal,
            sdf_value = d_val,
            sdf_grad = d_grad.squeeze(),
            dt=dt,
        )


        x = torch.tensor(q_opt[:, 1], dtype=torch.float32)
        traj.append(x.clone())
        if torch.norm(x - x_goal) < 0.05:
            print(f"Reached goal at step {step}")
            break


    fig, ax = plt.subplots()
    for c in circles:
        ax.add_patch(c.create_patch(color="red"))
    for q in traj:
        joints = robot.forward_kinematics_all_joints(q.unsqueeze(0))[0]
        ax.plot(joints[0], joints[1], "bo-", alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("Robot Trajectory")
    plt.show()


if __name__ == "__main__":
    main()
