#
# unicycle_model.py
#
# Defines the unicycle (differential drive) kinematic model and constraints
# for feasible trajectory projection with obstacle avoidance.
#
# State:    x = [px, py, theta, v, omega]^T
#   px, py  : Cartesian position [m]
#   theta   : heading angle [rad]
#   v       : linear speed [m/s]
#   omega   : angular rate [rad/s]
#
# Controls: u = [a, alpha]^T
#   a       : linear acceleration [m/s^2]
#   alpha   : angular acceleration [rad/s^2]
#
# The model is expressed in continuous time and discretized by ERK4 inside acados.

import numpy as np
import types
from casadi import *


def unicycle_model(obstacles=None):
    """
    Build the CasADi symbolic unicycle model.

    Parameters
    ----------
    obstacles : list of (ox, oy, r_safe)
        Each obstacle is a tuple of (x-position, y-position, safety radius).
        If None, no obstacle constraints are added.

    Returns
    -------
    model      : SimpleNamespace  acados-compatible model struct
    constraint : SimpleNamespace  constraint expressions and bounds
    """
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "unicycle_projection"

    # ------------------------------------------------------------------ #
    #  States
    # ------------------------------------------------------------------ #
    px    = MX.sym("px")       # x position [m]
    py    = MX.sym("py")       # y position [m]
    theta = MX.sym("theta")    # heading [rad]
    v     = MX.sym("v")        # linear speed [m/s]
    omega = MX.sym("omega")    # angular rate [rad/s]
    x = vertcat(px, py, theta, v, omega)

    # ------------------------------------------------------------------ #
    #  Controls
    # ------------------------------------------------------------------ #
    a     = MX.sym("a")        # linear acceleration [m/s^2]
    alpha = MX.sym("alpha")    # angular acceleration [rad/s^2]
    u = vertcat(a, alpha)

    # ------------------------------------------------------------------ #
    #  State derivatives (for implicit form)
    # ------------------------------------------------------------------ #
    pxdot    = MX.sym("pxdot")
    pydot    = MX.sym("pydot")
    thetadot = MX.sym("thetadot")
    vdot     = MX.sym("vdot")
    omegadot = MX.sym("omegadot")
    xdot = vertcat(pxdot, pydot, thetadot, vdot, omegadot)

    # ------------------------------------------------------------------ #
    #  Kinematics
    # ------------------------------------------------------------------ #
    f_expl = vertcat(
        v * cos(theta),     # px_dot
        v * sin(theta),     # py_dot
        omega,              # theta_dot
        a,                  # v_dot
        alpha,              # omega_dot
    )

    # ------------------------------------------------------------------ #
    #  Physical bounds
    # ------------------------------------------------------------------ #
    # Speed limits
    model.v_min   = -0.5   # allow small reversing
    model.v_max   =  3.0   # [m/s]

    # Angular rate limits
    model.omega_min = -2.0  # [rad/s]
    model.omega_max =  2.0  # [rad/s]

    # Acceleration limits (control bounds)
    model.a_min    = -3.0   # [m/s^2]
    model.a_max    =  3.0   # [m/s^2]
    model.alpha_min = -4.0  # [rad/s^2]
    model.alpha_max =  4.0  # [rad/s^2]

    # ------------------------------------------------------------------ #
    #  Nonlinear obstacle-avoidance constraints
    #  h_i(x) = r_safe^2 - (px - ox_i)^2 - (py - oy_i)^2 <= 0
    #  i.e.  we require  (px-ox)^2 + (py-oy)^2 >= r_safe^2
    #  acados convention: lh <= h(x,u) <= uh
    #  So we write  h_i = (px-ox)^2 + (py-oy)^2  and impose  h_i >= r_safe^2
    # ------------------------------------------------------------------ #
    if obstacles is None:
        obstacles = []

    constraint.obstacles    = obstacles
    constraint.n_obstacles  = len(obstacles)

    if len(obstacles) > 0:
        h_obs_list = []
        for (ox, oy, r_safe) in obstacles:
            dist_sq = (px - ox)**2 + (py - oy)**2
            h_obs_list.append(dist_sq)
        h_obs = vertcat(*h_obs_list)
        constraint.expr         = h_obs
        constraint.lh_obs       = np.array([r**2 for (_, _, r) in obstacles])
        constraint.uh_obs       = np.array([1e6  for _           in obstacles])
    else:
        constraint.expr         = vertcat([])
        constraint.lh_obs       = np.array([])
        constraint.uh_obs       = np.array([])

    # ------------------------------------------------------------------ #
    #  Initial condition (overridden at runtime)
    # ------------------------------------------------------------------ #
    model.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ #
    #  Pack model struct for acados
    # ------------------------------------------------------------------ #
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.x     = x
    model.xdot  = xdot
    model.u     = u
    model.z     = vertcat([])
    model.p     = vertcat([])
    model.name  = model_name

    return model, constraint
