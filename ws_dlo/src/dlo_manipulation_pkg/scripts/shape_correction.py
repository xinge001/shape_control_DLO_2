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

def correct_shape(pred_pos, fixed_idx=None, alpha=10.0, beta=50.0, gamma=10.0, max_spacing=0.07, min_spacing=0.02, new_spacing=0.05):
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

    if alpha>0 or beta>0 or gamma>0:
        print("Correct using alpha/beta/gamma")
        x0_free = pack(pred_pos)
        res = minimize(obj, x0_free, method='L-BFGS-B', options={'maxiter': 1000})
        xs_corr = unpack(res.x)
    else:
        print("Correct with distance only")
        xs_corr = pred_pos
        res=None

    # Final enforcement: clip any remaining spacing violations
    dists = np.linalg.norm(xs_corr[1:] - xs_corr[:-1], axis=1)
    for i in range(len(xs_corr) - 1):
        if fixed_mask[i + 1]:  # skip fixed point
            continue

        p1, p2 = xs_corr[i], xs_corr[i + 1]
        dist = np.linalg.norm(p2 - p1)

        # too far apart or too close → bring closer
        if dist > max_spacing or dist < min_spacing:
            direction = (p2 - p1) / (dist + 1e-12)
            new_p2 = p1 + direction * new_spacing
            xs_corr[i + 1] = new_p2

    return xs_corr, res

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