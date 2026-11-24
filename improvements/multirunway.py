"""Phase 3 improvement A: joint PnP using every runway visible in the image (our model's predicted corners).
For each test image with 2 labelled runways: position error from runway 1 alone, runway 2 alone, and both jointly."""
import itertools, json, os, sys
import cv2, numpy as np, pandas as pd
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, PROJ)
from lard_geom import load_labels, DATA, runway_db, ecef, RunwayFrame, K, corners_of  # noqa: E402

split = json.load(open(os.path.join(DATA, 'split_bing.json')))
lab = load_labels('bingmaps', 'test')
lab = lab[lab.image.isin(set(split['test']))]
db = runway_db('bingmaps')
lab = lab[[a in db and r in db[a] for a, r in zip(lab.airport, lab.runway)]]
t = pd.read_csv(os.path.join(PROJ, 'runs', 'resnet50_bing', 'test_results.csv'), dtype={'runway': str})
g = lab.merge(t[['stem', 'runway', 'ours_xy', 'yolo_xy']], on=['stem', 'runway']).groupby('image').filter(lambda x: len(x) == 2)


def world(row):
    rw = db[row['airport']][row['runway']]
    return np.array([ecef(rw[c]['coordinate']['latitude'], rw[c]['coordinate']['longitude'], rw[c]['coordinate']['altitude']) for c in 'ABCD'])


def cam_of(rv, tv):
    R, _ = cv2.Rodrigues(rv); return (-R.T @ tv).ravel()


rows = []
for img, grp in g.groupby('image'):
    r1, r2 = grp.iloc[0].to_dict(), grp.iloc[1].to_dict()
    gt_e = ecef(r1['lat'], r1['lon'], r1['alt'])
    res = dict(image=img, airport=r1['airport'], runways=f"{r1['runway']} + {r2['runway']}", along_km=r1['along_track_distance'] * 1.852)
    W1, W2 = world(r1), world(r2); o = W1.mean(0)
    u = W1[1] - W1[0]; u /= np.linalg.norm(u); n = np.cross(W1[1] - W1[0], W1[3] - W1[0]); n /= np.linalg.norm(n)
    M = np.stack([u, np.cross(n, u), n]); gt = M @ (gt_e - o)
    if gt[2] < 0: M[1] *= -1; M[2] *= -1; gt = M @ (gt_e - o)
    Q1, Q2 = (M @ (W1 - o).T).T, (M @ (W2 - o).T).T
    for who in ('ours', 'yolo'):
        p1, p2 = np.array(json.loads(r1[f'{who}_xy'])), np.array(json.loads(r2[f'{who}_xy']))
        e1 = RunwayFrame('bingmaps', r1).errors(p1)['pos']; e2 = RunwayFrame('bingmaps', r2).errors(p2)['pos']

        def single(Q, P):
            c0 = Q.mean(0); Qp = Q - c0; Qp[:, 2] = 0; best = None          # IPPE needs a z=0 plane
            for perm in itertools.permutations(range(4)):
                _, rvs, tvs, _ = cv2.solvePnPGeneric(Qp[list(perm)], P, K, None, flags=cv2.SOLVEPNP_IPPE)
                for rv, tv in zip(rvs, tvs):
                    c = cam_of(rv, tv) + c0
                    if c[2] <= 0 or np.linalg.norm(c - Q[2:].mean(0)) > np.linalg.norm(c - Q[:2].mean(0)): continue
                    re = np.abs(cv2.projectPoints(Qp[list(perm)], rv, tv, K, None)[0].squeeze() - P).mean()
                    if best is None or re < best[0]:
                        R, _ = cv2.Rodrigues(rv); tv0 = tv + R @ (-c0).reshape(3, 1)   # shift pose back to the shared frame
                        best = (re, perm, rv, tv0)
            return best
        b1, b2 = single(Q1, p1), single(Q2, p2)
        if b1 is None or b2 is None: continue
        Wall = np.vstack([Q1[list(b1[1])], Q2[list(b2[1])]]); Pall = np.vstack([p1, p2])
        init = b1 if b1[0] <= b2[0] else b2                                    # pick start by reprojection error (no ground truth)
        ok, rv, tv = cv2.solvePnP(Wall, Pall, K, None, init[2].copy(), init[3].copy(), True, cv2.SOLVEPNP_ITERATIVE)
        res.update({f'{who}_rw1': e1, f'{who}_rw2': e2, f'{who}_joint': float(np.linalg.norm(cam_of(rv, tv) - gt))})
    rows.append(res)
df = pd.DataFrame(rows).dropna()
df['ours_single'] = df[['ours_rw1', 'ours_rw2']].min(axis=1)        # best single runway (optimistic baseline)
df.to_csv(os.path.join(PROJ, 'improvements', 'multirunway_results.csv'), index=False)
print('images', len(df))
print(df[['ours_rw1', 'ours_rw2', 'ours_single', 'ours_joint', 'yolo_rw1', 'yolo_joint']].median().round(0).to_string())
print('joint beats target-runway-only on', round((df.ours_joint < df.ours_rw1).mean() * 100), '% of images;',
      'beats best single runway on', round((df.ours_joint < df.ours_single).mean() * 100), '%')
