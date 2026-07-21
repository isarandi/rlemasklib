"""Tests for warp_distorted with the 12- and 14-coefficient (tilted sensor) models.

The reference is a dense backward warp implemented in numpy with the same OpenCV-style
model: pixel = K @ tilt(distort_12(normalized)). Boundary rounding differs between the
run-based warp and dense nearest sampling, so masks are compared by IoU.
"""

import numpy as np
import pytest

from rlemasklib import RLEMask


def distort_12(p, d):
    x, y = p[..., 0], p[..., 1]
    r2 = x * x + y * y
    a = (1 + r2 * (d[0] + r2 * (d[1] + d[4] * r2))) / (
        1 + r2 * (d[5] + r2 * (d[6] + d[7] * r2)))
    b = 2 * (x * d[3] + y * d[2])
    cx = r2 * (d[3] + d[8] + d[9] * r2)
    cy = r2 * (d[2] + d[10] + d[11] * r2)
    return np.stack([x * (a + b) + cx, y * (a + b) + cy], axis=-1)


def undistort_12(p, d, iters=100):
    pu = p.copy()
    for _ in range(iters):
        x, y = pu[..., 0], pu[..., 1]
        r2 = x * x + y * y
        a = (1 + r2 * (d[0] + r2 * (d[1] + d[4] * r2))) / (
            1 + r2 * (d[5] + r2 * (d[6] + d[7] * r2)))
        b = 2 * (x * d[3] + y * d[2])
        cx = r2 * (d[3] + d[8] + d[9] * r2)
        cy = r2 * (d[2] + d[10] + d[11] * r2)
        pu = np.stack([(p[..., 0] - cx - x * b) / a, (p[..., 1] - cy - y * b) / a], axis=-1)
    return pu


def tilt_matrix(tau_x, tau_y):
    cx, sx = np.cos(tau_x), np.sin(tau_x)
    cy, sy = np.cos(tau_y), np.sin(tau_y)
    return np.array([
        [cx, 0, 0],
        [-sx * sy, cy, 0],
        [sy, -cy * sx, cx * cy]])


def apply_h(H, p):
    ph = p @ H[:, :2].T + H[:, 2]
    return ph[..., :2] / ph[..., 2:3]


def rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def dense_reference_warp(mask, K1, K2, R1, R2, d1, d2, out_shape):
    """Backward-map every output pixel to the input image and sample nearest."""
    d1 = np.pad(np.asarray(d1, np.float64), (0, max(0, 14 - len(d1))))
    d2 = np.pad(np.asarray(d2, np.float64), (0, max(0, 14 - len(d2))))
    h_out, w_out = out_shape
    ys, xs = np.mgrid[:h_out, :w_out]
    p_out = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)

    pn2 = apply_h(np.linalg.inv(K2), p_out)
    q2 = apply_h(np.linalg.inv(tilt_matrix(d2[12], d2[13])), pn2)
    pu2 = undistort_12(q2, d2)
    pu1 = apply_h(R1 @ R2.T, pu2)
    q1 = distort_12(pu1, d1)
    pn1 = apply_h(tilt_matrix(d1[12], d1[13]), q1)
    p1 = apply_h(K1, pn1)

    xi = np.round(p1[:, 0]).astype(np.int64)
    yi = np.round(p1[:, 1]).astype(np.int64)
    h, w = mask.shape
    inside = (0 <= xi) & (xi < w) & (0 <= yi) & (yi < h)
    out = np.zeros(h_out * w_out, dtype=np.uint8)
    out[inside] = mask[yi[inside], xi[inside]]
    return out.reshape(h_out, w_out)


def open_polar_tables(radius=10.0, n=360):
    thetas = np.linspace(-np.pi, np.pi, n).astype(np.float32)
    radii = np.full(n, radius, dtype=np.float32)
    return (radii, thetas), (radii, thetas)


def blob_mask(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    mask = np.zeros((h, w), np.uint8)
    for _ in range(4):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        r = rng.uniform(0.08, 0.22) * min(h, w)
        mask |= ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2).astype(np.uint8)
    return mask


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 1.0


D_RADIAL = [-0.25, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
D_FULL12 = [-0.2, 0.04, 0.004, -0.003, -0.002, 0.01, 0.002, -0.001,
            0.002, -0.001, 0.0015, -0.002]
D_TILT = D_FULL12 + [0.03, -0.02]
D_PURE_TILT = [0.0] * 12 + [0.04, 0.025]

CASES = [
    ('radial-no-tilt', D_RADIAL, D_RADIAL, 0),
    ('full12-no-tilt', D_FULL12, D_RADIAL, 0),
    ('tilt-both', D_TILT, [0.0] * 12 + [-0.02, 0.03], 0),
    ('tilt-old-only', D_TILT, D_RADIAL, 0),
    ('tilt-new-only', D_RADIAL, D_TILT, 0),
    ('pure-tilt', D_PURE_TILT, [0.0] * 12, 0),
    ('tilt-roll90', D_TILT, D_RADIAL, 90),
    ('tilt-roll-90', D_TILT, D_TILT, -90),
    ('tilt-roll180', D_TILT, D_RADIAL, 180),
    ('tilt-roll37', D_TILT, D_RADIAL, 37),
]


@pytest.mark.parametrize('name,d1,d2,roll_deg', CASES, ids=[c[0] for c in CASES])
def test_warp_distorted_vs_dense_reference(name, d1, d2, roll_deg):
    h, w = 120, 160
    if roll_deg in (90, -90):
        out_shape = (w, h)
    else:
        out_shape = (h, w)
    mask = blob_mask(h, w, seed=hash(name) % 2 ** 32)

    K1 = np.array([[95.0, 0, (w - 1) / 2], [0, 90.0, (h - 1) / 2], [0, 0, 1]])
    K2 = np.array([[90.0, 0, (out_shape[1] - 1) / 2],
                   [0, 95.0, (out_shape[0] - 1) / 2], [0, 0, 1]])
    R1 = np.eye(3)
    R2 = rot_z(roll_deg)

    ref = dense_reference_warp(mask, K1, K2, R1, R2, d1, d2, out_shape)
    assert ref.sum() > 100, 'degenerate test case, reference output nearly empty'

    result = RLEMask.from_array(mask).warp_distorted(
        R1, R2, K1, K2,
        np.asarray(d1, np.float64), np.asarray(d2, np.float64),
        open_polar_tables(), open_polar_tables(), out_shape)

    score = iou(np.array(result), ref)
    assert score > 0.95, f'{name}: IoU {score:.4f} vs dense reference'


def test_zero_tilt_matches_12_param_path():
    # explicit zero taus must give the identical result as the plain 12-coefficient call
    h, w = 100, 140
    mask = blob_mask(h, w, seed=7)
    K = np.array([[90.0, 0, (w - 1) / 2], [0, 90.0, (h - 1) / 2], [0, 0, 1]])
    args = (np.eye(3), rot_z(20), K, K)
    tables = (open_polar_tables(), open_polar_tables())
    m = RLEMask.from_array(mask)
    res12 = m.warp_distorted(*args, np.asarray(D_FULL12), np.asarray(D_RADIAL),
                             *tables, (h, w))
    res14 = m.warp_distorted(*args, np.asarray(D_FULL12 + [0.0, 0.0]),
                             np.asarray(D_RADIAL + [0.0, 0.0]), *tables, (h, w))
    assert res12 == res14
