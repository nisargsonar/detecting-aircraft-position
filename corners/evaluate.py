"""Evaluate the corner model on the Bing test subset against the released YOLOv8-Pose baseline (same runways).

Crops come from the YOLOv8 detector's own boxes (real detector in the loop). Usage: python3 -m corners.evaluate
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

from .data import RunwayCrops
from .model import CornerNet
from .train import to_full, OUT

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, 'baselines'))
from lard_geom import load_labels, DATA, RunwayFrame, corners_of  # noqa: E402
import eval_yolov8pose as B  # noqa: E402


def pnp(args):
    row, pts = args
    try:
        return RunwayFrame('bingmaps', row).errors(pts)
    except KeyError:
        return None


def main():
    split = json.load(open(os.path.join(DATA, 'split_bing.json')))
    lab = load_labels('bingmaps', 'test')
    lab = lab[lab.image.isin(set(split['test'])) & lab.image.map(lambda p: os.path.exists(os.path.join(DATA, p)))]
    base = pd.read_csv(os.path.join(PROJ, 'baselines', 'results_yolov8pose.csv'), dtype={'runway': str})
    base = base[(base.source == 'bingmaps') & base.detected]
    base['stem'] = base.image.str.replace('_BingMaps$', '', regex=True)
    lab = lab.merge(base.drop(columns=['image']), on=['stem', 'runway'], suffixes=('', '_b'))
    preds = B.load_preds()
    rows, boxes = [], {}
    for r in lab.to_dict('records'):
        gt = corners_of(r); gb = (*gt.min(0), *gt.max(0))
        c = [(B.iou(gb, b), b, kp) for s, b, kp in preds.get(r['stem'] + '_BingMaps', [])]
        c = [x for x in c if x[0] >= B.IOU_MIN]
        if c:
            _, bb, kp = max(c, key=lambda x: x[0])
            boxes[len(rows)] = bb; r['yolo_kp'] = kp; rows.append(r)
    print('test runway labels (detected by YOLOv8, both methods comparable):', len(rows), flush=True)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = CornerNet(pretrained=False).to(dev)
    model.load_state_dict(torch.load(os.path.join(OUT, 'best.pt'), map_location=dev)); model.eval()
    pred = np.zeros((len(rows), 4, 2), np.float32)
    with torch.no_grad():
        for x, _, A, idx in DataLoader(RunwayCrops(rows, boxes=boxes), 32, num_workers=0):
            pred[idx.numpy()] = to_full(model(x.to(dev))[0].cpu(), A).numpy()
    with Pool(9) as pool:
        errs = pool.map(pnp, [(r, pred[i].astype(float)) for i, r in enumerate(rows)])
    out = []
    for i, (r, e) in enumerate(zip(rows, errs)):
        if e is None: continue
        out.append(dict(stem=r['stem'], runway=r['runway'], image_path=r['image'],
                        gt_xy=json.dumps(corners_of(r).round(2).tolist()), ours_xy=json.dumps(pred[i].round(2).tolist()),
                        yolo_xy=json.dumps(np.round(r['yolo_kp'], 2).tolist()), along_nm=r['along_track_distance'], odd=r['runway_in_cone'],
                        ours_corner_px=float(np.linalg.norm(pred[i] - corners_of(r), axis=1).mean()),
                        **{f'ours_{k}': v for k, v in e.items()},
                        yolo_corner_px=r['pose_corner_px'], yolo_pos=r['pose_pos'], yolo_dist=r['pose_dist'],
                        yolo_lat=r['pose_lat'], yolo_glide=r['pose_glide'], true_pos=r['true_pos'], box_pos=r['boxpred_pos']))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, 'test_results.csv'), index=False)
    np.save(os.path.join(OUT, 'test_pred_corners.npy'), pred)
    i = df[df.odd == 'IN_ODD']
    print(i[['ours_corner_px', 'yolo_corner_px', 'ours_pos', 'yolo_pos', 'true_pos', 'box_pos',
             'ours_lat', 'yolo_lat', 'ours_glide', 'yolo_glide']].median().round(2).to_string())
    print('n IN_ODD', len(i))


if __name__ == '__main__':
    main()
