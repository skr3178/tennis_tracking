"""2D pixel track -> plane-points P (paper §3.1.1, Eqs. 1-2).

Vectorized numpy implementation. Supports both OpenGL and OpenCV ray
conventions (set per-dataset from camera.json's "convention" field).

Output convention (paper):
    P_t = (p_g.x, p_g.z, p_v.x, p_v.y) ∈ R^4
where
    p_g = ray ∩ {y = 0}   (drop y)
    p_v = ray ∩ {z = 0}   (drop z)
"""

from __future__ import annotations

import numpy as np


def _ray_dir_camera(uv: np.ndarray, intrinsics: np.ndarray,
                    convention: str) -> np.ndarray:
    """Return ray directions in CAMERA coords, shape (N, 4) homogeneous (w=0).

    OpenGL: cam looks down -Z, image y increases downward
            -> d_cam = (u-cx, -(v-cy), -f, 0)
    OpenCV: cam looks down +Z, image y increases downward
            -> d_cam = (u-cx,  (v-cy),  f, 0)
    """
    f, cx, cy = intrinsics
    u, v = uv[:, 0], uv[:, 1]
    if convention.lower().startswith("opengl"):
        return np.stack([u - cx, -(v - cy),
                         -np.full_like(u, f), np.zeros_like(u)], axis=1)
    if convention.lower().startswith("opencv"):
        return np.stack([u - cx, (v - cy),
                          np.full_like(u, f), np.zeros_like(u)], axis=1)
    raise ValueError(f"Unknown ray convention: {convention!r}")


def pixel_to_plane_points(uv: np.ndarray, intrinsics: np.ndarray,
                          extrinsic: np.ndarray,
                          convention: str = "opengl") -> np.ndarray:
    """uv: (L, 2) pixel coords (any float).
    intrinsics: (3,) = (f, cx, cy).
    extrinsic: (4, 4) world->camera.
    Returns P: (L, 4) float32 = (p_g.x, p_g.z, p_v.x, p_v.y).

    Both plane intersections are computed in WORLD coords. Frames where the
    ray is parallel to a plane (denominator near zero) get NaN — caller
    should reject or repair such sequences before training.
    """
    uv = np.asarray(uv, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    E = np.asarray(extrinsic, dtype=np.float64)

    E_inv = np.linalg.inv(E)
    c = (E_inv @ np.array([0., 0., 0., 1.]))[:3]                    # camera center, world
    d_cam = _ray_dir_camera(uv, intrinsics, convention)             # (L, 4)
    d_world = (E_inv @ d_cam.T).T[:, :3]                            # (L, 3)

    # Ray ∩ {y = 0}:  c.y + t*d.y = 0  ->  t = -c.y / d.y
    t_g = np.where(np.abs(d_world[:, 1]) > 1e-12,
                   -c[1] / d_world[:, 1], np.nan)
    p_g = c[None, :] + t_g[:, None] * d_world                       # (L, 3)

    # Ray ∩ {z = 0}:  c.z + t*d.z = 0  ->  t = -c.z / d.z
    t_v = np.where(np.abs(d_world[:, 2]) > 1e-12,
                   -c[2] / d_world[:, 2], np.nan)
    p_v = c[None, :] + t_v[:, None] * d_world                       # (L, 3)

    P = np.stack([p_g[:, 0], p_g[:, 2], p_v[:, 0], p_v[:, 1]], axis=1)
    return P.astype(np.float32)
