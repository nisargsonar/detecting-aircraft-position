"""Phase 1 baseline: true corners vs released YOLOv8-Pose corners vs predicted-box corners -> PnP, on the LARD V2 test set.

Uses the LARD authors' released predictions (yolov8pose_IN_ODD_predictions.json, deel-ai-papers/Yolo_models_LARD_V2).
Output: baselines/results_yolov8pose.csv (one row per test runway label).
"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from lard_geom import RunwayFrame, SUFFIX, corners_of, load_labels  # noqa: E402

SCORE_MIN, IOU_MIN = 0.25, 0.3


def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter + 1e-9)


def load_preds():
    by_img = defaultdict(list)
    for p in json.load(open(os.path.join(HERE, 'yolov8pose_IN_ODD_predictions.json'))):
        if p['score'] >= SCORE_MIN:
            x, y, w, h = p['bbox']
            by_img[p['image_id']].append((p['score'], (x, y, x + w, y + h), np.array(p['keypoints'], float).reshape(4, 2)))
    return by_img


PREDS = None


def work(args):
    source, row = args
    stem = os.path.splitext(os.path.basename(row['image']))[0]
    gt = corners_of(row)
    gbox = (*gt.min(0), *gt.max(0))
    out = dict(image=stem, source=source, airport=row['airport'], runway=row['runway'], odd=row['runway_in_cone'],
               along_nm=row['along_track_distance'], box_w=gbox[2] - gbox[0], box_h=gbox[3] - gbox[1])
    try:
        fr = RunwayFrame(source, row)
    except KeyError:                                   # runway missing from this renderer's database
        return None
    out.update({f'true_{k}': v for k, v in fr.errors(gt).items()})
    cands = [(iou(gbox, b), s, b, kp) for s, b, kp in PREDS.get(stem, [])]
    cands = [c for c in cands if c[0] >= IOU_MIN]
    out['detected'] = bool(cands)
    if cands:
        _, s, b, kp = max(cands, key=lambda c: c[0])
        out['score'] = s
        out['pose_corner_px'] = float(np.linalg.norm(kp - gt, axis=1).mean())
        out.update({f'pose_{k}': v for k, v in fr.errors(kp).items()})
        bc = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], float)
        out['boxpred_corner_px'] = float(np.linalg.norm(bc - gt, axis=1).mean())
        out.update({f'boxpred_{k}': v for k, v in fr.errors(bc).items()})
    return out


def init():
    global PREDS
    PREDS = load_preds()


if __name__ == '__main__':
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0          # optional subsample for a quick run
    data = os.path.join(os.path.dirname(HERE), 'data')
    jobs = []
    for src in SUFFIX.values():
        d = load_labels(src, 'test').dropna(subset=['lat'])
        if frac < 1: d = d.sample(frac=frac, random_state=0)
        jobs += [(src, r) for r in d.to_dict('records')]
    print('runway labels', len(jobs), flush=True)
    with Pool(9, initializer=init) as pool:
        rows = [r for r in pool.imap_unordered(work, jobs, chunksize=64) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, 'results_yolov8pose.csv'), index=False)
    print('saved', len(df), 'rows')
