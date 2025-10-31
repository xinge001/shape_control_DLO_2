
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def total_length(xs):
    return np.sum(np.linalg.norm(xs[1:] - xs[:-1], axis=1))

def bending_energy(xs):
    dif2 = xs[2:] - 2 * xs[1:-1] + xs[:-2]
    return np.sum(np.sum(dif2**2, axis=1))

def spacing_energy(xs, d0):
    dists = np.linalg.norm(xs[1:] - xs[:-1], axis=1)
    return np.sum((dists - d0)**2)

def data_fidelity(xs, pred_pos):
    return np.sum(np.sum((xs - pred_pos)**2, axis=1))

def correct_shape(pred_pos, fixed_idx=None, alpha=50.0, beta=10.0, gamma=10.0):
    """
    Enforce smoothness + spacing but preserve S-shape using data fidelity.
    pred_pos: (N,2)
    alpha: smoothness weight
    beta: spacing weight
    gamma: data fidelity weight
    """
    N = pred_pos.shape[0]
    d0 = np.mean(np.linalg.norm(pred_pos[1:] - pred_pos[:-1], axis=1))

    x0 = pred_pos.copy().ravel()

    fixed_idx = [] if fixed_idx is None else list(fixed_idx)
    fixed_mask = np.zeros(N, dtype=bool)
    fixed_mask[fixed_idx] = True
    free_idx = [i for i in range(N) if not fixed_mask[i]]

    def unpack(vec):
        xs = np.zeros((N, 2))
        xs[fixed_mask] = pred_pos[fixed_mask]
        xs[free_idx] = vec.reshape((-1, 2))
        return xs

    def pack(xs):
        return xs[free_idx].ravel()

    def obj(free_vec):
        xs = unpack(free_vec)
        return (gamma * data_fidelity(xs, pred_pos)
                + alpha * bending_energy(xs)
                + beta * spacing_energy(xs, d0))

    x0_free = pack(pred_pos)
    res = minimize(obj, x0_free, method='L-BFGS-B', options={'maxiter': 1000})
    xs_corr = unpack(res.x)
    return xs_corr, res

# # --- Example: noisy S curve ---
# N = 10
# x = np.linspace(0, 10, N)
# y = np.sin(x / 1.5) + 0.15 * np.random.randn(N)
# pred_pos = np.stack([x, y], axis=1)
#
# # Add a few outliers
# pred_pos[3] += np.array([0.0, 1.0])
# pred_pos[6] += np.array([0.5, -1.2])
#
# pred_pos = np.array([[0.00399,-0.09217],
# [0.00556,-0.05834],
# [-0.02530,-0.03136], [-0.06688,-0.01571],
# [-0.11066,0.01051],
# [-0.12461,0.05522],
# [-0.10443,0.09935],
# [-0.07128,0.12991],
# [-0.31817,0.15169],
# [-0.01142,0.20132]]
# )
#
# fixed_idx = [0, N - 1]
#
# xs_corr, res = correct_shape(pred_pos, fixed_idx=[0, 9, 1, 2, 3, 4, 5, 6, 7],
#                              alpha=50, beta=10.0, gamma=10.0)
# print(xs_corr)
#
# print("Optimization success:", res.success)
#
# # --- Visualization ---
# plt.figure(figsize=(7, 5))
# plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'o--b', label="Predicted (noisy S)")
# plt.plot(xs_corr[:, 0], xs_corr[:, 1], 'o-g', label="Corrected (smooth S)")
# plt.scatter(pred_pos[fixed_idx, 0], pred_pos[fixed_idx, 1],
#             c='red', s=80, label="Fixed endpoints")
#
# plt.axis('equal')
# plt.title("Smooth S-Curve Correction with Uniform Spacing")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
from scipy.optimize import minimize

def correct_shape(pred_pos, fixed_idx=None, alpha=10.0, beta=50.0, gamma=10.0, max_spacing=0.05):
    """
    Enforce smoothness + spacing + max distance constraint (< max_spacing)
    while preserving S-shape using data fidelity.
    """
    N = pred_pos.shape[0]
    d0 = np.mean(np.linalg.norm(pred_pos[1:] - pred_pos[:-1], axis=1))

    x0 = pred_pos.copy().ravel()

    fixed_idx = [] if fixed_idx is None else list(fixed_idx)
    fixed_mask = np.zeros(N, dtype=bool)
    fixed_mask[fixed_idx] = True
    free_idx = [i for i in range(N) if not fixed_mask[i]]

    def unpack(vec):
        xs = np.zeros((N, 2))
        xs[fixed_mask] = pred_pos[fixed_mask]
        xs[free_idx] = vec.reshape((-1, 2))
        return xs

    def pack(xs):
        return xs[free_idx].ravel()

    def obj(free_vec):
        xs = unpack(free_vec)
        return (gamma * data_fidelity(xs, pred_pos)
                + alpha * bending_energy(xs)
                + beta * spacing_energy(xs, d0))

    x0_free = pack(pred_pos)
    res = minimize(obj, x0_free, method='L-BFGS-B', options={'maxiter': 1000})
    xs_corr = unpack(res.x)

    # Final enforcement: clip any remaining violations
    dists = np.linalg.norm(xs_corr[1:] - xs_corr[:-1], axis=1)
    too_far = np.where(dists > max_spacing)[0]
    # Sequential enforcement: update using latest corrected positions
    print(fixed_mask)
    for i in range(len(xs_corr) - 1):
        if fixed_mask[i+1]:  # skip fixed point
            continue
        print(i)
        p1, p2 = xs_corr[i], xs_corr[i + 1]
        dist = np.linalg.norm(p2 - p1)
        print(dist)
        if dist > max_spacing:
            direction = (p2 - p1) / (dist + 1e-12)
            new_p2 = p1 + direction * max_spacing

            # ensure the next point (if fixed) remains correct
            xs_corr[i + 1] = new_p2

    return xs_corr, res
#

def visualize_rectification(pred_pos, xs_corr, max_spacing=0.1):
    """
    Visualize predicted vs corrected feature point positions
    and highlight distances exceeding max_spacing before correction.
    """
    pred_pos = np.array(pred_pos)
    xs_corr = np.array(xs_corr)
    N = len(pred_pos)

    # Compute segment lengths
    dist_before = np.linalg.norm(pred_pos[1:] - pred_pos[:-1], axis=1)
    dist_after = np.linalg.norm(xs_corr[1:] - xs_corr[:-1], axis=1)

    # Identify segments that were too long before correction
    too_far = np.where(dist_before > max_spacing)[0]

    plt.figure(figsize=(8, 6))
    plt.title("Feature Point Rectification")

    # Original shape
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'o--', color='gray', label='Before correction')
    for i, d in enumerate(dist_before):
        plt.text((pred_pos[i, 0] + pred_pos[i+1, 0]) / 2,
                 (pred_pos[i, 1] + pred_pos[i+1, 1]) / 2 + 0.01,
                 f"{d:.2f}", color='gray', fontsize=8, ha='center')

    # Highlight segments > max_spacing
    for i in too_far:
        plt.plot(pred_pos[i:i+2, 0], pred_pos[i:i+2, 1], 'r-', lw=2, label='Too long' if i == too_far[0] else None)

    # Corrected shape
    plt.plot(xs_corr[:, 0], xs_corr[:, 1], 'o-', color='dodgerblue', label='After correction')
    for i, d in enumerate(dist_after):
        plt.text((xs_corr[i, 0] + xs_corr[i+1, 0]) / 2,
                 (xs_corr[i, 1] + xs_corr[i+1, 1]) / 2 - 0.02,
                 f"{d:.2f}", color='blue', fontsize=8, ha='center')

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# 0.25017,-0.22113
# 0.20964,-0.32570
# 0.17368,-0.89150
# 0.18435,-1.63230
# 0.11935,-0.64233
# 0.12342,-2.18787
# 0.01702,-1.97716
# 0.08124,-2.16756
# 0.15318,-0.29132
# 0.12297,-0.02485


# Example data
# pred_pos = np.array([[ 0.23846142, -0.2518216 ],
#  [ 0.1922356 , -0.3092568 ],
#  [ 0.16498034 ,-0.39607978],
#  [ 0.13706501, -0.46536988],
#  [ 0.08633662, -0.5959293 ],
#  [ 0.03847078 ,-0.76597196],
#  [ 0.04023035 ,-0.66732204],
#  [ 0.05565862 ,-0.65864575],
#  [ 0.07530048 ,-0.60467154],
#  [ 0.12286823 ,-0.05098236]]
# )
#
# pred_pos = np.array([
#     [0.2505423,  -0.19469127],
#     [0.2114978,  -0.26718575],
#     [0.1813188,  -0.5732832],
#     [0.16591537, -0.63305694],
#     [0.12633821, -0.55969614],
#     [0.08349445, -1.2577811],
#     [0.06443337, -1.3495734],
#     [0.07065633, -1.423554],
#     [0.15473604, -0.21846072],
#     [0.12001045, -0.02024299],
# ])


# # Run correction
# # xs_corr, res = correct_shape(pred_pos,fixed_idx=[0,9,1,4,8])
# xs_corr, res = correct_shape(pred_pos,fixed_idx=[0,9,1,8])
# print(xs_corr)
#
# # Visualize before vs after
# visualize_rectification(pred_pos, xs_corr)






















import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Smoothness / S-shape check
# -------------------------

def discrete_curvature(points, normalize=True):
    """
    Discrete curvature proxy using second finite differences.
    points: (N,2) or (N,3)
    normalize: divide by (avg spacing)^2 to make measure scale-invariant
    returns: curv_mag (N-2,) for i=1..N-2
    """
    pts = np.asarray(points, dtype=float)
    d1 = pts[1:] - pts[:-1]               # (N-1, dim)
    d2 = d1[1:] - d1[:-1]                 # (N-2, dim)
    curv = np.linalg.norm(d2, axis=1)     # magnitude of second difference
    if normalize:
        avg_seg = np.mean(np.linalg.norm(d1, axis=1)) + 1e-12
        curv = curv / (avg_seg ** 2)
    return curv


def tangent_angle_changes(points):
    """
    Angle changes between consecutive segments (radians).
    returns angles array of length N-2 (angles at interior vertices).
    """
    pts = np.asarray(points, dtype=float)
    v = pts[1:] - pts[:-1]
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    v_unit = v / norms
    # dot product between consecutive unit vectors
    dots = np.sum(v_unit[1:] * v_unit[:-1], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # between 0 and pi
    return angles  # (N-2,)


def smoothness_check(points,
                     curv_med_thresh=0.08,      # was 0.03
                     curv_pct95_thresh=0.25,     # was 0.12
                     angle_med_thresh=0.25,      # was 0.12 rad (~14°)
                     angle_pct95_thresh=0.8,     # was 0.4 rad (~23°)
                     verbose=False):
    """
    Decide whether the curve is 'smooth enough' while allowing S-shapes.
    Uses robust statistics (median, 95th percentile) so local kinks get detected.
    - curv_med_thresh: median curvature threshold (scale-inv)
    - curv_pct95_thresh: 95th-percentile curvature threshold (for local spikes)
    - angle*_thresh: in radians (angle between segments)
    returns: (is_smooth:bool, diagnostics:dict)
    """
    pts = np.asarray(points)
    N = len(pts)
    if N < 4:
        # too few points to evaluate curvature properly; treat as smooth
        return True, {"reason": "too_few_points"}

    curv = discrete_curvature(pts, normalize=True)  # length N-2
    angles = tangent_angle_changes(pts)             # length N-2

    curv_med = float(np.median(curv))
    curv_pct95 = float(np.percentile(curv, 95))
    angle_med = float(np.median(angles))
    angle_pct95 = float(np.percentile(angles, 95))

    # Decision logic: allow S-shape (global curvature okay) but forbid large local spikes
    smooth = (curv_med <= curv_med_thresh and
              curv_pct95 <= curv_pct95_thresh and
              angle_med <= angle_med_thresh and
              angle_pct95 <= angle_pct95_thresh)

    diag = {
        "N": N,
        "curv_median": curv_med,
        "curv_95pct": curv_pct95,
        "angle_median": angle_med,
        "angle_95pct": angle_pct95,
        "thresholds": {
            "curv_med_thresh": curv_med_thresh,
            "curv_pct95_thresh": curv_pct95_thresh,
            "angle_med_thresh": angle_med_thresh,
            "angle_pct95_thresh": angle_pct95_thresh,
        },
        "is_smooth": bool(smooth),
        "curv_array": curv,
        "angle_array": angles
    }

    if verbose:
        print(f"curv_med={curv_med:.4f} (th={curv_med_thresh}), curv_95%={curv_pct95:.4f} (th={curv_pct95_thresh})")
        print(f"angle_med={angle_med:.4f} rad (th={angle_med_thresh}), angle_95%={angle_pct95:.4f} rad (th={angle_pct95_thresh})")
        print("=> smooth" if smooth else "=> not smooth")
    return bool(smooth), diag


def check_distance_anomalies(fp_pos, min_dist=0.2, max_dist=1.5, verbose=True):
    """
    Check if any consecutive feature points are too close or too far apart.

    Args:
        fp_pos: (N,2) array of feature point coordinates
        min_dist: minimum allowed distance (too small → overlap or compression)
        max_dist: maximum allowed distance (too large → stretching or break)
        verbose: print diagnostic info

    Returns:
        has_anomaly: bool
        too_close_idx: list of indices where distance < min_dist
        too_far_idx: list of indices where distance > max_dist
        distances: array of all consecutive distances
    """
    fp_pos = np.asarray(fp_pos, dtype=float)
    diffs = fp_pos[1:] - fp_pos[:-1]
    distances = np.linalg.norm(diffs, axis=1)

    too_close_idx = np.where(distances < min_dist)[0]
    too_far_idx = np.where(distances > max_dist)[0]
    has_anomaly = len(too_close_idx) > 0 or len(too_far_idx) > 0

    if verbose:
        for i in too_close_idx:
            print(f"⚠️ Too close between point {i} and {i+1}: {distances[i]:.3f} < {min_dist}")
        for i in too_far_idx:
            print(f"⚠️ Too far between point {i} and {i+1}: {distances[i]:.3f} > {max_dist}")
        if not has_anomaly:
            print("✅ All consecutive distances within thresholds.")

    return has_anomaly, too_close_idx.tolist(), too_far_idx.tolist(), distances
# -------------------------
# Diagnostic plot
# -------------------------

def plot_smoothness_diagnostics(points, diag=None, title="Curve smoothness diagnostic"):
    """
    Shows curve, curvature magnitudes and tangent-angle changes along index.
    diag: optional diagnostics dict (if None it will be recomputed)
    """
    pts = np.asarray(points)
    if diag is None:
        _, diag = smoothness_check(points, verbose=False)

    curv = diag["curv_array"]
    angles = diag["angle_array"]
    N = diag["N"]
    idx = np.arange(1, N-1)  # curvature/angle index aligns to interior vertices

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), gridspec_kw={"height_ratios": [2, 1, 1]})
    ax0, ax1, ax2 = axes

    # Curve plot
    ax0.plot(pts[:, 0], pts[:, 1], '-o', label="points")
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_title(title)
    ax0.grid(True)

    # Curvature plot
    ax1.plot(idx, curv, '-o', label="discrete curvature (norm of 2nd diff)")
    ax1.axhline(diag["thresholds"]["curv_med_thresh"], color='C1', linestyle='--', label='curv_med_thresh')
    ax1.axhline(diag["thresholds"]["curv_pct95_thresh"], color='C2', linestyle=':', label='curv_95pct_thresh')
    ax1.set_ylabel("curvature (scale-inv)")
    ax1.grid(True)
    ax1.legend()

    # Angle-change plot
    ax2.plot(idx, angles, '-o', label="tangent angle change (rad)")
    ax2.axhline(diag["thresholds"]["angle_med_thresh"], color='C1', linestyle='--', label='angle_med_thresh')
    ax2.axhline(diag["thresholds"]["angle_pct95_thresh"], color='C2', linestyle=':', label='angle_95pct_thresh')
    ax2.set_ylabel("angle (rad)")
    ax2.set_xlabel("vertex index (interior)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# -------------------------
# Example usage
# -------------------------


