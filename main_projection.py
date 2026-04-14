#
# main_projection.py
#
# NMPC feasible trajectory projection for a unicycle with obstacle avoidance.
# Six scenarios: 4 obstacle-free, 2 obstacle scenarios.
#
# Fixes vs previous version:
#   - Terminal cost weight is small (w_pos_e = 1) -> no endpoint pinning
#   - Straight+obstacle: larger radius, offset so path MUST deviate
#   - Slalom: obstacles overlap the straight-line path so weaving is mandatory
#
# Usage:
#   python main_projection.py
#

import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

from projection_settings import projection_settings


# ======================================================================
#  Reference generators
# ======================================================================

def make_straight(N, dt):
    """3 m straight line along x — feasible baseline."""
    s = np.linspace(0, 3.0, N + 1)
    return s, np.zeros(N + 1)


def make_sharp_turn(N, dt):
    """Abrupt 90-deg corner at midpoint — discontinuous heading."""
    half = N // 2
    xr = np.concatenate([np.linspace(0, 1.5, half + 1),
                          1.5 * np.ones(N - half)])
    yr = np.concatenate([np.zeros(half + 1),
                          np.linspace(0, 1.5, N - half)])
    return xr, yr


def make_zigzag(N, dt):
    """3 m forward with 3 full lateral oscillations of ±0.5 m."""
    t  = np.linspace(0, 1.0, N + 1)   # normalised [0,1]
    xr = 3.0 * t
    yr = 0.5 * np.sin(2 * np.pi * 3 * t)
    return xr, yr


def make_curved(N, dt):
    """Quarter-circle arc R=2 m — nearly feasible."""
    a  = np.linspace(0, np.pi / 2, N + 1)
    xr = 2.0 * np.sin(a)
    yr = 2.0 * (1 - np.cos(a))
    return xr, yr


def make_straight_blocked(N, dt):
    """
    Straight line along x with ONE large obstacle sitting squarely on it.
    Radius 0.45 m ensures the path cannot slip past even with small lateral
    errors — the solver must execute a clear lateral detour.
    The obstacle is placed at x=1.5, y=0 (dead centre of the 3 m path).
    """
    xr, yr = make_straight(N, dt)
    # Single large obstacle centred on the path
    obstacles = [
        (1.5, 0.0, 0.45),
    ]
    return xr, yr, obstacles


def make_slalom(N, dt):
    """
    Straight reference along x with THREE obstacles whose edges OVERLAP
    the centreline, so a straight path is blocked.  The only clear route
    weaves left-right-left between the gates.

    Gate geometry (all at radius 0.30 m):
      Gate 1 at x=0.75: two obstacles at y=+0.35 and y=-0.35.
              Clear corridor is centred at y=0 with width ~0.10 m — tight.
      Gate 2 at x=1.50: shifted up — obstacles at y=+0.65 and y=-0.05.
              Clear corridor is around y=+0.30.
      Gate 3 at x=2.25: shifted down — obstacles at y=-0.65 and y=+0.05.
              Clear corridor is around y=-0.30.

    A straight line at y=0 clips both obstacles at gates 2 and 3, forcing
    the solver to weave.
    """
    xr = np.linspace(0, 3.0, N + 1)
    yr = np.zeros(N + 1)               # perfectly straight reference

    r = 0.30
    obstacles = [
        # Gate 1 — straddles y=0, corridor at y=0 (tight but passable straight)
        (0.75,  0.42, r),
        (0.75, -0.42, r),
        # Gate 2 — corridor shifted to y≈+0.32 (straight line clipped below)
        (1.50,  0.65, r),
        (1.50, -0.05, r),
        # Gate 3 — corridor shifted to y≈-0.32 (straight line clipped above)
        (2.25, -0.65, r),
        (2.25,  0.05, r),
    ]
    return xr, yr, obstacles


# ======================================================================
#  Scenario registry
# ======================================================================

SCENARIOS = [
    ("Straight",             make_straight,        False),
    ("Sharp Turn",           make_sharp_turn,       False),
    ("Zigzag",               make_zigzag,           False),
    ("Curved Arc",           make_curved,           False),
    ("Straight + Obstacle",  make_straight_blocked, True),
    ("Slalom (gates)",       make_slalom,           True),
]

N  = 20
Tf = 2.0
dt = Tf / N


# ======================================================================
#  Solve
# ======================================================================

def run_scenario(label, gen_fn, has_obs):
    if has_obs:
        xr, yr, obstacles = gen_fn(N, dt)
    else:
        xr, yr = gen_fn(N, dt)
        obstacles = []

    x0 = np.array([xr[0], yr[0], 0.0, 0.0, 0.0])

    solver, model, constraint = projection_settings(
        N, Tf, xr, yr, x0, obstacles=obstacles
    )

    # Set per-node references
    for k in range(N):
        solver.set(k, "yref", np.array([xr[k], yr[k], 0.0, 0.0]))
    solver.set(N, "yref", np.array([xr[N], yr[N]]))

    # 20 RTI iterations (warm-start chain)
    comp_times = []
    for _ in range(20):
        t0     = time.time()
        status = solver.solve()
        comp_times.append((time.time() - t0) * 1e3)
        if status not in (0, 2):
            print(f"  [{label}] solver status {status}")

    # Extract solution
    xp = np.zeros(N + 1)
    yp = np.zeros(N + 1)
    vp = np.zeros(N + 1)
    tp = np.zeros(N + 1)
    for k in range(N + 1):
        s      = solver.get(k, "x")
        xp[k]  = s[0]
        yp[k]  = s[1]
        tp[k]  = s[2]
        vp[k]  = s[3]

    ct  = np.array(comp_times)
    dev = np.sqrt((xp - xr)**2 + (yp - yr)**2)

    return dict(label=label, x_ref=xr, y_ref=yr,
                x_proj=xp, y_proj=yp, v_proj=vp, theta_proj=tp,
                obstacles=obstacles, comp_times=ct,
                avg_time=ct.mean(), max_time=ct.max(), dev=dev)


# ======================================================================
#  Plotting
# ======================================================================

def plot_traj(ax, res, show_cb=True):
    # Reference
    ax.plot(res["x_ref"], res["y_ref"], "r--",
            lw=1.8, label="Reference", zorder=2)

    # Projected path coloured by speed
    xp, yp, vp = res["x_proj"], res["y_proj"], res["v_proj"]
    pts  = np.array([xp, yp]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vmax = max(vp.max(), 0.05)
    lc   = LineCollection(segs, cmap="cool",
                          norm=plt.Normalize(0, vmax),
                          linewidth=2.5, zorder=3)
    lc.set_array(vp[:-1])
    ax.add_collection(lc)
    if show_cb:
        cb = plt.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("speed [m/s]", fontsize=8)

    # Obstacles
    for (ox, oy, r) in res["obstacles"]:
        ax.add_patch(patches.Circle((ox, oy), r,
                     color="tomato", alpha=0.35, zorder=4))
        ax.plot(ox, oy, "rx", ms=8, mew=2, zorder=5)

    ax.plot(xp[0],  yp[0],  "go", ms=7, label="Start",      zorder=6)
    ax.plot(xp[-1], yp[-1], "bs", ms=7, label="End (proj.)", zorder=6)

    ax.set_xlabel("x [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    ax.set_title(res["label"], fontsize=10, fontweight="bold")
    ax.set_aspect("equal", "datalim")
    ax.margins(0.18)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7, loc="best")


def plot_timing(ax, res):
    ct = res["comp_times"]
    ax.bar(np.arange(1, len(ct)+1), ct,
           color="steelblue", alpha=0.75, width=0.7)
    ax.axhline(ct.mean(), color="firebrick", ls="--", lw=1.5,
               label=f"avg {ct.mean():.2f} ms")
    ax.axhline(ct.max(),  color="black",     ls=":",  lw=1.2,
               label=f"max {ct.max():.2f} ms")
    ax.set_xlabel("RTI iteration", fontsize=9)
    ax.set_ylabel("Time [ms]",     fontsize=9)
    ax.set_title("Solve time — " + res["label"], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)


# ======================================================================
#  Main
# ======================================================================

def main():
    os.makedirs("results_projection", exist_ok=True)

    results = []
    hdr = (f"{'Scenario':<26} {'Avg[ms]':>9} {'Max[ms]':>9}"
           f" {'AvgDev[m]':>11} {'MaxDev[m]':>11}")
    print(hdr); print("-" * len(hdr))

    for (label, fn, has_obs) in SCENARIOS:
        print(f"  Solving: {label} ...", end="", flush=True)
        res = run_scenario(label, fn, has_obs)
        results.append(res)
        print(f"\r{label:<26} {res['avg_time']:>9.2f} {res['max_time']:>9.2f}"
              f" {res['dev'].mean():>11.4f} {res['dev'].max():>11.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        plot_traj(axes[0], res, show_cb=True)
        plot_timing(axes[1], res)
        plt.tight_layout()
        safe = (label.lower()
                .replace(" ", "_").replace("(", "")
                .replace(")", "").replace("+", "plus"))
        plt.savefig(f"results_projection/traj_{safe}.png", dpi=150)
        plt.close()

    # Combined 2x3 overview
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, res in zip(axes.flatten(), results):
        plot_traj(ax, res, show_cb=False)
    plt.suptitle("NMPC Feasible Trajectory Projection — All Scenarios",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("results_projection/all_scenarios.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results_projection/all_scenarios.png")

    # Timing bar chart
    labels = [r["label"] for r in results]
    xpos   = np.arange(len(labels))
    w      = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(xpos - w/2, [r["avg_time"] for r in results], w,
           label="Average", color="steelblue", alpha=0.85)
    ax.bar(xpos + w/2, [r["max_time"] for r in results], w,
           label="Maximum", color="tomato", alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Computation time [ms]", fontsize=10)
    ax.set_title("RTI Solve Time Across Scenarios", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig("results_projection/timing_comparison.png", dpi=150)
    plt.close()
    print("Saved: results_projection/timing_comparison.png")

    # Deviation box plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bp = ax.boxplot([r["dev"] for r in results],
                    labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_ylabel("Position deviation [m]", fontsize=10)
    ax.set_title("Reference-to-Projection Deviation by Scenario", fontsize=11)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig("results_projection/deviation_boxplot.png", dpi=150)
    plt.close()
    print("Saved: results_projection/deviation_boxplot.png")

    # Summary
    print("\n" + "=" * len(hdr))
    print("Summary"); print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for res in results:
        print(f"{res['label']:<26} {res['avg_time']:>9.2f}"
              f" {res['max_time']:>9.2f}"
              f" {res['dev'].mean():>11.4f} {res['dev'].max():>11.4f}")
    print("=" * len(hdr))
    print(f"\nOverall average : {np.mean([r['avg_time'] for r in results]):.2f} ms")
    print(f"Overall maximum : {np.max( [r['max_time'] for r in results]):.2f} ms")


if __name__ == "__main__":
    main()