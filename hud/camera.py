"""Full pinhole camera (K, R, C) recovered from the ground homography alone, so the
shuttle, which is in the air where the ground homography means nothing, can be placed in 3D.

Assumes square pixels, principal point at the image centre and no lens distortion
(true enough for phones and broadcast cameras). Focal length comes from the two
rotation-column constraints on the homography (Zhang's method, single plane).
"""
import numpy as np


class Camera:
    def __init__(self, H, frame_size):
        w, h = frame_size
        self.c = np.array([w / 2, h / 2])
        G = np.linalg.inv(np.asarray(H))  # court -> image
        G = np.array([[1, 0, -self.c[0]], [0, 1, -self.c[1]], [0, 0, 1]]) @ G
        h1, h2, h3 = G.T
        # u = 1/f^2:  a*u + b = 0 for (r1 . r2 = 0) and (|r1| = |r2|); least squares over both
        a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
        b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
        u = -(a @ b) / (a @ a)
        if u <= 0:
            raise ValueError("Calibration gives no valid focal length; are the corners clicked in order?")
        self.f = 1 / np.sqrt(u)
        Kinv = np.diag([1 / self.f, 1 / self.f, 1])
        lam = 1 / np.linalg.norm(Kinv @ h1)
        if (Kinv @ h3)[2] * lam < 0:  # court must be in front of the camera
            lam = -lam
        r1, r2, t = Kinv @ h1 * lam, Kinv @ h2 * lam, Kinv @ h3 * lam
        U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
        self.R = U @ Vt
        self.C = -self.R.T @ t  # camera centre, court metres (z up)

    def ray(self, px):
        """Unit direction (court frame) of the viewing ray through pixel px."""
        d = self.R.T @ np.array([(px[0] - self.c[0]) / self.f, (px[1] - self.c[1]) / self.f, 1.0])
        return d / np.linalg.norm(d)

    def at_height(self, px, z):
        d = self.ray(px)
        return self.C + d * (z - self.C[2]) / d[2]

    def near_vertical(self, px, xy):
        """Point on the ray through px closest to the vertical line above court point xy:
        where the shuttle is if it's being hit by a player standing at xy."""
        d, o = self.ray(px), self.C
        q, e = np.array([xy[0], xy[1], 0.0]), np.array([0, 0, 1.0])
        w0 = o - q
        b, dd, ee = d @ e, d @ w0, e @ w0
        s = (b * ee - dd) / (1 - b * b)
        return o + s * d

    def project(self, X):
        x = self.R @ (np.asarray(X, float) - self.C)
        return self.c + self.f * x[:2] / x[2]

    def on_line(self, px, a, b, min_sin=np.sin(np.radians(8))):
        """Point on the 3-D line a->b closest to the ray through px (where the shuttle is, if it
        flies straight from a towards b). None if the ray is nearly parallel to the line."""
        d, u = self.ray(px), np.asarray(b, float) - np.asarray(a, float)
        u = u / np.linalg.norm(u)
        c = d @ u
        if np.sqrt(max(0.0, 1 - c * c)) < min_sin:
            return None
        w0 = np.asarray(a, float) - self.C
        s = (c * (d @ w0) - (u @ w0)) / (1 - c * c)
        return np.asarray(a, float) + s * u
