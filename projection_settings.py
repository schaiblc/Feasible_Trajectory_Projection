#
# projection_settings.py
#
# Constructs the acados OCP for feasible unicycle trajectory projection
# with obstacle avoidance.
#
# Key design choices:
#   - Stage cost tracks (px, py) with high weight -> minimal distortion
#   - Terminal cost weight is LOW (no endpoint pinning)
#   - Obstacle constraints are hard nonlinear inequalities, softened
#     with large L1 slack penalty for QP feasibility
#

import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from unicycle_model import unicycle_model


def projection_settings(N, Tf, x_ref, y_ref, x0, obstacles=None):
    """
    Build and return an AcadosOcpSolver for trajectory projection.

    Parameters
    ----------
    N        : int    number of shooting intervals
    Tf       : float  prediction horizon [s]
    x_ref    : (N+1,) array of reference x positions
    y_ref    : (N+1,) array of reference y positions
    x0       : (5,)   initial state [px, py, theta, v, omega]
    obstacles: list of (ox, oy, r_safe) or None

    Returns
    -------
    acados_solver, model, constraint
    """
    ocp = AcadosOcp()

    model, constraint = unicycle_model(obstacles=obstacles)

    # ------------------------------------------------------------------ #
    #  AcadosModel
    # ------------------------------------------------------------------ #
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x    = model.x
    model_ac.xdot = model.xdot
    model_ac.u    = model.u
    model_ac.z    = model.z
    model_ac.p    = model.p
    model_ac.name = model.name

    if constraint.n_obstacles > 0:
        model_ac.con_h_expr   = constraint.expr
        model_ac.con_h_expr_e = constraint.expr   # enforce at terminal node too

    ocp.model = model_ac

    # ------------------------------------------------------------------ #
    #  Dimensions
    # ------------------------------------------------------------------ #
    nx   = model.x.rows()   # 5
    nu   = model.u.rows()   # 2
    ny   = 2 + nu            # [px, py, a, adot]  at each stage
    ny_e = 2                 # [px, py]            at terminal node

    ocp.solver_options.N_horizon = N

    # ------------------------------------------------------------------ #
    #  Cost
    #
    #  Stage  : penalise position deviation + small input regularisation
    #  Terminal: SMALL weight — we do NOT want to pin the endpoint.
    #            The terminal term merely provides mild guidance; the
    #            obstacle constraints at the terminal node do the work.
    # ------------------------------------------------------------------ #
    w_pos   = 1e2    # position tracking (stage)
    w_a     = 1e-2   # linear acceleration regularisation
    w_alpha = 1e-2   # angular acceleration regularisation
    w_pos_e = 1e0    # terminal position weight — intentionally small

    W   = np.diag([w_pos, w_pos, w_a, w_alpha])
    W_e = np.diag([w_pos_e, w_pos_e])

    ocp.cost.cost_type   = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Vx selects [px, py] from x = [px, py, theta, v, omega]
    Vx = np.zeros((ny, nx))
    Vx[0, 0] = 1.0
    Vx[1, 1] = 1.0

    # Vu selects [a, adot] from u = [a, adot]
    Vu = np.zeros((ny, nu))
    Vu[2, 0] = 1.0
    Vu[3, 1] = 1.0

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0, 0] = 1.0
    Vx_e[1, 1] = 1.0

    ocp.cost.W    = W
    ocp.cost.W_e  = W_e
    ocp.cost.Vx   = Vx
    ocp.cost.Vu   = Vu
    ocp.cost.Vx_e = Vx_e

    ocp.cost.yref   = np.array([x_ref[0], y_ref[0], 0.0, 0.0])
    ocp.cost.yref_e = np.array([x_ref[N], y_ref[N]])

    # ------------------------------------------------------------------ #
    #  Slack variables for obstacle constraints
    #  Large linear penalty (zl/zu) makes violation very costly;
    #  small quadratic penalty (Zl/Zu) adds curvature for the QP solver.
    # ------------------------------------------------------------------ #
    n_obs = constraint.n_obstacles
    if n_obs > 0:
        # Slacks at stage nodes
        nsh = n_obs
        ocp.cost.zl = 1e4 * np.ones(nsh)
        ocp.cost.zu = 1e4 * np.ones(nsh)
        ocp.cost.Zl = 1e1 * np.ones(nsh)
        ocp.cost.Zu = 1e1 * np.ones(nsh)
        # Slacks at terminal node (same size — one per obstacle)
        ocp.cost.zl_e = 1e4 * np.ones(nsh)
        ocp.cost.zu_e = 1e4 * np.ones(nsh)
        ocp.cost.Zl_e = 1e1 * np.ones(nsh)
        ocp.cost.Zu_e = 1e1 * np.ones(nsh)

    # ------------------------------------------------------------------ #
    #  State bounds: v and omega
    # ------------------------------------------------------------------ #
    ocp.constraints.lbx   = np.array([model.v_min,     model.omega_min])
    ocp.constraints.ubx   = np.array([model.v_max,     model.omega_max])
    ocp.constraints.idxbx = np.array([3, 4])

    # ------------------------------------------------------------------ #
    #  Input bounds
    # ------------------------------------------------------------------ #
    ocp.constraints.lbu   = np.array([model.a_min,     model.alpha_min])
    ocp.constraints.ubu   = np.array([model.a_max,     model.alpha_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # ------------------------------------------------------------------ #
    #  Nonlinear obstacle constraints (stage + terminal)
    # ------------------------------------------------------------------ #
    if n_obs > 0:
        lh = constraint.lh_obs
        uh = constraint.uh_obs

        # Stage
        ocp.constraints.lh    = lh
        ocp.constraints.uh    = uh
        ocp.constraints.lsh   = np.zeros(nsh)
        ocp.constraints.ush   = np.zeros(nsh)
        ocp.constraints.idxsh = np.arange(nsh)

        # Terminal
        ocp.constraints.lh_e    = lh
        ocp.constraints.uh_e    = uh
        ocp.constraints.lsh_e   = np.zeros(nsh)
        ocp.constraints.ush_e   = np.zeros(nsh)
        ocp.constraints.idxsh_e = np.arange(nsh)

    # ------------------------------------------------------------------ #
    #  Initial condition
    # ------------------------------------------------------------------ #
    ocp.constraints.x0 = x0

    # ------------------------------------------------------------------ #
    #  Solver options
    # ------------------------------------------------------------------ #
    ocp.solver_options.tf                     = Tf
    ocp.solver_options.qp_solver             = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type       = "SQP_RTI"
    ocp.solver_options.hessian_approx        = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type       = "ERK"
    ocp.solver_options.sim_method_num_stages  = 4
    ocp.solver_options.sim_method_num_steps   = 3

    acados_solver = AcadosOcpSolver(ocp, json_file="acados_proj_ocp.json")

    return acados_solver, model, constraint