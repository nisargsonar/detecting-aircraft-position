"""General evaluator: any trained run, any test set, 4-corner and (if available) 16-point PnP.

    python3 -m corners.evaluate2 --run dinov2_bing --test bing662     # the exact 662 Nov-6 Bing runways
    python3 -m corners.evaluate2 --run runA --test bing_all           # all downloaded Bing test runways
    python3 -m corners.evaluate2 --run runA --test flsim              # unseen renderer (our models only)

Crops come from the released YOLO detector's boxes (IoU >= 0.3, score >= 0.25), as in the Nov 6 comparison.
Writes runs/<run>/eval_<test>.csv (one row per runway). YOLO keypoint columns are added for Bing only.
"""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

from .data import RunwayCrops, KP_UV
from .model import CornerNet
from .train import to_full

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, 'baselines'))
from lard_geom import load_labels, DATA, RunwayFrame, corners_of  # noqa: E402
import eval_yolov8pose as B  # noqa: E402

TESTS = {'bing662': ('bingmaps', 'BingMaps', 'bing_test_rest'), 'bing_all': ('bingmaps', 'BingMaps', 'bing_test_rest'),
         'flsim': ('flsim', 'FLSim', 'flsim_test'), 'xplane': ('xplane', 'XPlane', 'xplane_test')}


def pnp(args):
    src, row, pts, n_kp = args
    try:
        fr = RunwayFrame(src, row)
    except KeyError:
        return None
    out = {f'{k}4': v for k, v in fr.errors(pts[:4]).items()}
    out['true_pos'] = fr.errors(corners_of(row))['pos']
    if n_kp > 4:
        out.update({f'{k}{n_kp}': v for k, v in fr.errors(pts, KP_UV[n_kp]).items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--test', required=True, choices=list(TESTS))
    a = ap.parse_args()
    rdir = os.path.join(PROJ, 'runs', a.run)
    meta = json.load(open(os.path.join(rdir, 'history.json')))
    meta = meta if isinstance(meta, dict) else {'backbone': 'resnet50', 'n_kp': 4}          # Nov 5 run stored a bare list
    backbone, n_kp = meta.get('backbone', 'resnet50'), meta.get('n_kp', 4)
    src, suf, key = TESTS[a.test]
    imgs = set(json.load(open(os.path.join(DATA, 'split_multi.json')))[key])
    lab = load_labels(src, 'test').dropna(subset=['lat'])
    lab = lab[lab.image.isin(imgs) & lab.image.map(lambda p: os.path.exists(os.path.join(DATA, p)))]
    if a.test == 'bing662':                                                     # exactly the Nov 6 runway set
        ref = pd.read_csv(os.path.join(PROJ, 'runs', 'resnet50_bing', 'test_results.csv'), dtype={'runway': str})
        ref = ref[ref.odd == 'IN_ODD']
        lab = lab.merge(ref[['stem', 'runway']], on=['stem', 'runway'])
    preds = B.load_preds()
    rows, boxes = [], {}
    for r in lab.to_dict('records'):
        gt = corners_of(r); gb = (*gt.min(0), *gt.max(0))
        c = [(B.iou(gb, b), b, kp) for s, b, kp in preds.get(r['stem'] + '_' + suf, [])]
        c = [x for x in c if x[0] >= B.IOU_MIN]
        if c:
            _, bb, kp = max(c, key=lambda x: x[0])
            boxes[len(rows)] = bb; r['yolo_kp'] = kp; rows.append(r)
    print(f'{a.run} ({backbone}, {n_kp} kp) on {a.test}: {len(rows)} runways with a detector box', flush=True)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = CornerNet(pretrained=False, backbone=backbone, n_kp=n_kp).to(dev)
    model.load_state_dict(torch.load(os.path.join(rdir, 'best.pt'), map_location=dev)); model.eval()
    pred = np.zeros((len(rows), n_kp, 2), np.float32)
    with torch.no_grad():
        for x, _, A, idx in DataLoader(RunwayCrops(rows, boxes=boxes, n_kp=n_kp), 32, num_workers=0):
            pred[idx.numpy()] = to_full(model(x.to(dev))[0].cpu(), A).numpy()
    with Pool(9) as pool:
        errs = pool.map(pnp, [(src, r, pred[i].astype(float), n_kp) for i, r in enumerate(rows)])
    out = []
    for i, (r, e) in enumerate(zip(rows, errs)):
        if e is None: continue
        gt = corners_of(r)
        d = dict(stem=r['stem'], runway=r['runway'], airport=r['airport'], image_path=r['image'], odd=r['runway_in_cone'],
                 along_km=r['along_track_distance'] * 1.852, corner_px=float(np.linalg.norm(pred[i, :4] - gt, axis=1).mean()),
                 pred_xy=json.dumps(pred[i].round(2).tolist()), gt_xy=json.dumps(gt.round(2).tolist()), **e)
        if src == 'bingmaps':                                                   # YOLO keypoints, same runway + box
            yk = np.asarray(r['yolo_kp']); fr = RunwayFrame(src, r)
            ye = fr.errors(yk)
            d.update(yolo_corner_px=float(np.linalg.norm(yk - gt, axis=1).mean()), yolo_xy=json.dumps(yk.round(2).tolist()),
                     **{f'yolo_{k}': v for k, v in ye.items()})
        out.append(d)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(rdir, f'eval_{a.test}.csv'), index=False)
    i = df[df.odd == 'IN_ODD']
    cols = ['corner_px', 'pos4'] + ([f'pos{n_kp}'] if n_kp > 4 else []) + (['yolo_corner_px', 'yolo_pos'] if src == 'bingmaps' else [])
    print(f'IN_ODD n={len(i)} medians:', i[cols].median().round(2).to_dict(), flush=True)


if __name__ == '__main__':
    main()
