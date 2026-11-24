"""Shared geometry for LARD V2: runway database, PnP pose from 4 corners, and guidance errors.

Corner order everywhere is the CSV order: TL, TR, BR, BL (image positions).
DB corners A, B are the far end; C, D are the landing threshold.
"""
import itertools, json, os
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILES = {'arcgis': 'ArcGIS', 'bingmaps': 'Bing', 'flsim': 'FLSim', 'ges': 'GEarth', 'xplane': 'XPlane'}
SUFFIX = {'ArcGIS': 'arcgis', 'BingMaps': 'bingmaps', 'FLSim': 'flsim', 'GES': 'ges', 'XPlane': 'xplane'}
CORNERS = ['TL', 'TR', 'BR', 'BL']
F = 512 / np.tan(np.radians(30))                       # 60 deg FOV on 1024 px
K = np.array([[F, 0, 512], [0, F, 512], [0, 0, 1]], float)
_DB = {}


def runway_db(source):
    if source not in _DB:
        _DB[source] = json.load(open(os.path.join(ROOT, 'LARD', 'data', f'runways_db_V2_{DB_FILES[source]}.json')))
    return _DB[source]


def ecef(lat, lon, h):
    a, e2 = 6378137.0, 6.69437999014e-3
    la, lo = np.radians(lat), np.radians(lon)
    n = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    return np.array([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo), (n * (1 - e2) + h) * np.sin(la)])


def corners_of(row):
    return np.array([[row[f'x_{c}'], row[f'y_{c}']] for c in CORNERS], float)


class RunwayFrame:
    """Local planar frame of one runway (z up), with the ground-truth camera position for one image row."""

    def __init__(self, source, row):
        rw = runway_db(source)[row['airport']][row['runway']]
        P = np.array([ecef(rw[c]['coordinate']['latitude'], rw[c]['coordinate']['longitude'],
                           rw[c]['coordinate']['altitude']) for c in 'ABCD'])
        o = P.mean(0); Pc = P - o
        u = Pc[1] - Pc[0]; u /= np.linalg.norm(u)
        n = np.cross(Pc[1] - Pc[0], Pc[3] - Pc[0]); n /= np.linalg.norm(n)
        M = np.stack([u, np.cross(n, u), n])
        gt = M @ (ecef(row['lat'], row['lon'], row['alt']) - o)
        if gt[2] < 0:
            M[1] *= -1; M[2] *= -1; gt = M @ (ecef(row['lat'], row['lon'], row['alt']) - o)
        self.Q = (M @ Pc.T).T; self.Q[:, 2] = 0
        self.gt = gt
        self.ltp = self.Q[2:].mean(0)                               # threshold centre
        ax = self.ltp - self.Q[:2].mean(0); self.axis = ax / np.linalg.norm(ax)

    def solve(self, img_pts):
        """IPPE PnP over corner orderings; keep camera above runway and on the threshold side."""
        full = self.solve_full(img_pts)
        return None if full is None else full[0]

    def solve_full(self, img_pts):
        """Like solve(), but returns (camera position, corner ordering, rvec, tvec)."""
        best = None
        for perm in itertools.permutations(range(4)):
            Qp = self.Q[list(perm)]
            try:
                _, rvs, tvs, _ = cv2.solvePnPGeneric(Qp, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
            except cv2.error:
                continue
            for rv, tv in zip(rvs, tvs):
                R, _ = cv2.Rodrigues(rv); cam = (-R.T @ tv).ravel()
                if cam[2] <= 0 or np.linalg.norm(cam - self.ltp) > np.linalg.norm(cam - self.Q[:2].mean(0)):
                    continue
                re = np.abs(cv2.projectPoints(Qp, rv, tv, K, None)[0].squeeze() - img_pts).mean()
                if best is None or re < best[0]:
                    best = (re, cam, perm, rv, tv)
        return None if best is None else best[1:]

    def solve_multi(self, img_pts, uv):
        """PnP on n runway keypoints. img_pts[:4] are the corners (TL, TR, BR, BL); uv are the keypoints' unit-square
        coordinates (u across, v along the runway). Corners give the ordering; IPPE then solves on all points."""
        img_pts = np.asarray(img_pts, float)
        full = self.solve_full(img_pts[:4])
        if full is None:
            return None
        _, perm, _, _ = full
        Qp = self.Q[list(perm)]                                         # 3D corners in TL, TR, BR, BL order
        uv = np.asarray(uv, float); u, v = uv[:, :1], uv[:, 1:]
        P3 = (1 - u) * (1 - v) * Qp[0] + u * (1 - v) * Qp[1] + u * v * Qp[2] + (1 - u) * v * Qp[3]
        best = None                                                      # IPPE on all (planar) points, same physical rules
        _, rvs, tvs, _ = cv2.solvePnPGeneric(P3, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
        for rv, tv in zip(rvs, tvs):
            R, _ = cv2.Rodrigues(rv); cam = (-R.T @ tv).ravel()
            if cam[2] <= 0 or np.linalg.norm(cam - self.ltp) > np.linalg.norm(cam - self.Q[:2].mean(0)):
                continue
            re = np.abs(cv2.projectPoints(P3, rv, tv, K, None)[0].squeeze() - img_pts).mean()
            if best is None or re < best[0]:
                best = (re, cam)
        return None if best is None else best[1]

    def guidance(self, cam):
        """(distance along centreline m, lateral angle deg, glide angle deg w.r.t. VRP 305 m past threshold)."""
        v = cam - self.ltp; side = np.cross([0, 0, 1.0], self.axis)
        along = v @ self.axis
        return along, np.degrees(np.arctan2(v @ side, along)), np.degrees(np.arctan2(v[2], along + 305))

    def errors(self, img_pts, uv=None):
        """Position error (m), distance error (m), lateral/glide angle errors (deg); NaNs if PnP fails.
        With uv (unit-square coords of each keypoint), uses all keypoints (solve_multi); else 4-corner PnP."""
        cam = self.solve(np.asarray(img_pts, float)) if uv is None else self.solve_multi(img_pts, uv)
        if cam is None:
            return dict(pos=np.nan, dist=np.nan, lat=np.nan, glide=np.nan)
        g, e = self.guidance(self.gt), self.guidance(cam)
        return dict(pos=float(np.linalg.norm(cam - self.gt)), dist=abs(e[0] - g[0]),
                    lat=abs(e[1] - g[1]), glide=abs(e[2] - g[2]))


def box_corners(pts):
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_POSE_LUT = None


def _stem(s):
    return s.str.extract(r'/([^/]+)_[A-Za-z]+\.jpg$')[0]


def load_labels(source, split):
    """Read a LARD V2 metadata CSV. Runway names stay strings ('05' not 5). GES / X-Plane CSVs ship without
    lat/lon/alt; all renderers share the same scenario poses, so those are filled from the ArcGIS/Bing/FLSim CSVs."""
    global _POSE_LUT
    import pandas as pd
    d = pd.read_csv(os.path.join(DATA, f'm_{source}_{split}.csv'), sep=';', dtype={'runway': str})
    d['stem'] = _stem(d.image)
    if 'lat' not in d.columns:
        if _POSE_LUT is None:
            parts = []
            for s in ('arcgis', 'bingmaps', 'flsim'):
                for sp in ('train', 'test'):
                    p = pd.read_csv(os.path.join(DATA, f'm_{s}_{sp}.csv'), sep=';', dtype={'runway': str},
                                    usecols=['image', 'runway', 'lat', 'lon', 'alt'])
                    p['stem'] = _stem(p.image); parts.append(p.drop(columns='image'))
            _POSE_LUT = pd.concat(parts).drop_duplicates(['stem', 'runway'])
        d = d.merge(_POSE_LUT, on=['stem', 'runway'], how='left')
    return d
