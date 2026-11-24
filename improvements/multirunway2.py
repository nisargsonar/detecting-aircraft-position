"""Multi-runway joint PnP + wrong-runway check, from any run's eval CSV (predicted corners, released-detector boxes).

    python3 improvements/multirunway2.py --run resnet50_bing --test bing_all

For every image with exactly 2 evaluated runways: position error from the first runway alone (4 corners) and from one
pose fitted to all 8 corners (IPPE start per runway, chosen by reprojection error, then iterative refinement).
Wrong-runway check: from the joint pose, the lateral angle to each runway; predicted 'lined-up' runway = smaller |angle|;
truth = the runway with the smaller |labelled lateral_path_angle|.
Writes results/multirunway.json (+ per-image CSV in the run folder).
"""
import argparse, itertools, json, os, sys
import cv2, numpy as np, pandas as pd
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, PROJ)
from lard_geom import load_labels, runway_db, ecef, RunwayFrame, K  # noqa: E402

SRC = {'bing_all': 'bingmaps', 'bing662': 'bingmaps', 'flsim': 'flsim', 'xplane': 'xplane'}


def cam_of(rv, tv):
    R, _ = cv2.Rodrigues(rv); return (-R.T @ tv).ravel()


def lateral(cam, Q):
    """Lateral path angle (deg) of a camera relative to runway Q (4 corners, A,B far end; C,D threshold)."""
    ltp = Q[2:].mean(0); ax = ltp - Q[:2].mean(0); ax /= np.linalg.norm(ax)
    up = np.cross(Q[1] - Q[0], Q[3] - Q[0]); up /= np.linalg.norm(up)
    if up @ (cam - ltp) < 0: up = -up
    v = cam - ltp
    return np.degrees(np.arctan2(v @ np.cross(up, ax), v @ ax))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--test', default='bing_all')
    a = ap.parse_args(); src = SRC[a.test]; db = runway_db(src)
    ev = pd.read_csv(os.path.join(PROJ, 'runs', a.run, f'eval_{a.test}.csv'), dtype={'runway': str})
    lab = load_labels(src, 'test')
    ev = ev.merge(lab[['stem', 'runway', 'lat', 'lon', 'alt', 'lateral_path_angle']], on=['stem', 'runway'])
    world = lambda ap_, rw: np.array([ecef(db[ap_][rw][c]['coordinate']['latitude'], db[ap_][rw][c]['coordinate']['longitude'],
                                           db[ap_][rw][c]['coordinate']['altitude']) for c in 'ABCD'])
    rows = []
    for img, g in ev.groupby('image_path'):
        if len(g) != 2: continue
        r1, r2 = g.iloc[0].to_dict(), g.iloc[1].to_dict()
        W1, W2 = world(r1['airport'], r1['runway']), world(r2['airport'], r2['runway'])
        o = W1.mean(0); u = W1[1] - W1[0]; u /= np.linalg.norm(u)
        n = np.cross(W1[1] - W1[0], W1[3] - W1[0]); n /= np.linalg.norm(n); M = np.stack([u, np.cross(n, u), n])
        gt = M @ (ecef(r1['lat'], r1['lon'], r1['alt']) - o)
        if gt[2] < 0: M[1] *= -1; M[2] *= -1; gt = M @ (ecef(r1['lat'], r1['lon'], r1['alt']) - o)
        Q1, Q2 = (M @ (W1 - o).T).T, (M @ (W2 - o).T).T
        p1, p2 = np.array(json.loads(r1['pred_xy']))[:4], np.array(json.loads(r2['pred_xy']))[:4]

        def single(Q, P):
            c0 = Q.mean(0); Qp = Q - c0; Qp[:, 2] = 0; best = None
            for perm in itertools.permutations(range(4)):
                _, rvs, tvs, _ = cv2.solvePnPGeneric(Qp[list(perm)], P, K, None, flags=cv2.SOLVEPNP_IPPE)
                for rv, tv in zip(rvs, tvs):
                    c = cam_of(rv, tv) + c0
                    if c[2] <= 0 or np.linalg.norm(c - Q[2:].mean(0)) > np.linalg.norm(c - Q[:2].mean(0)): continue
                    re = np.abs(cv2.projectPoints(Qp[list(perm)], rv, tv, K, None)[0].squeeze() - P).mean()
                    if best is None or re < best[0]:
                        R, _ = cv2.Rodrigues(rv); best = (re, perm, rv, tv + R @ (-c0).reshape(3, 1), c)
            return best
        b1, b2 = single(Q1, p1), single(Q2, p2)
        if b1 is None or b2 is None: continue
        init = b1 if b1[0] <= b2[0] else b2
        Wall = np.vstack([Q1[list(b1[1])], Q2[list(b2[1])]]); Pall = np.vstack([p1, p2])
        ok, rv, tv = cv2.solvePnP(Wall, Pall, K, None, init[2].copy(), init[3].copy(), True, cv2.SOLVEPNP_ITERATIVE)
        cj = cam_of(rv, tv)
        lat_pred = [lateral(cj, Q1), lateral(cj, Q2)]
        lat_true = [abs(r1['lateral_path_angle']), abs(r2['lateral_path_angle'])]
        rows.append(dict(image=img, airport=r1['airport'], runways=f"{r1['runway']} + {r2['runway']}", along_km=r1['along_km'],
                         single_m=float(np.linalg.norm(b1[4] - gt)), joint_m=float(np.linalg.norm(cj - gt)),
                         aligned_true=[r1['runway'], r2['runway']][int(np.argmin(lat_true))],
                         aligned_pred=[r1['runway'], r2['runway']][int(np.argmin(np.abs(lat_pred)))],
                         lat_gap_true_deg=abs(lat_true[0] - lat_true[1])))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(PROJ, 'runs', a.run, f'multirunway_{a.test}.csv'), index=False)
    rng = np.random.default_rng(0); x, y = d.joint_m.values, d.single_m.values; n = len(d)
    bs = [np.median(x[i]) - np.median(y[i]) for i in (rng.integers(0, n, n) for _ in range(2000))]
    res = dict(run=a.run, test=a.test, n_images=int(n),
               single_runway_median_m=round(float(np.median(y)), 1), joint_median_m=round(float(np.median(x)), 1),
               diff_ci95=[round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
               joint_better_pct=round(float((x < y).mean() * 100), 1),
               wrong_runway=dict(n_images=int(n), correct_pct=round(float((d.aligned_true == d.aligned_pred).mean() * 100), 1),
                                 median_true_lateral_gap_deg=round(float(d.lat_gap_true_deg.median()), 2)))
    os.makedirs(os.path.join(PROJ, 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(PROJ, 'results', f'multirunway_{a.run}_{a.test}.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
