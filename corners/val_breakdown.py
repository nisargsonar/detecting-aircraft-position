"""Validation corner error of a run, split by renderer (why mean >> median?).
Uses the GT-box crops of training-time validation (no detector), 4 corners, full-image px.
Usage: python3 -m corners.val_breakdown --run runA  ->  results/val_by_renderer_<run>.json"""
import argparse, json, os
import numpy as np, torch
from torch.utils.data import DataLoader

from .data import RunwayCrops
from .model import CornerNet
from .train import PROJ, multi_rows, to_full


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='runA'); a = ap.parse_args()
    h = json.load(open(os.path.join(PROJ, 'runs', a.run, 'history.json')))
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = CornerNet(pretrained=False, backbone=h['backbone'], n_kp=h['n_kp']).to(dev)
    model.load_state_dict(torch.load(os.path.join(PROJ, 'runs', a.run, 'best.pt'), map_location=dev)); model.eval()
    out = {}
    for r in ('bingmaps', 'xplane'):
        _, va = multi_rows((r,))
        e = []
        with torch.no_grad():
            for x, kp, A, _ in DataLoader(RunwayCrops(va, n_kp=h['n_kp']), 32, num_workers=0):
                pk, _ = model(x.to(dev))
                e.append((to_full(pk.cpu(), A) - to_full(kp, A)).norm(dim=-1)[:, :4].mean(1))
        e = torch.cat(e).numpy()
        out[r] = dict(n=int(len(e)), corner_px_median=round(float(np.median(e)), 2), corner_px_mean=round(float(e.mean()), 2),
                      pck1=round(float((e <= 1).mean() * 100), 1), over_5px_pct=round(float((e > 5).mean() * 100), 1),
                      over_20px_pct=round(float((e > 20).mean() * 100), 1),
                      share_of_total_error_from_worst_5pct=round(float(np.sort(e)[-max(1, len(e) // 20):].sum() / e.sum() * 100), 1))
        print(r, out[r], flush=True)
    p = os.path.join(PROJ, 'results', f'val_by_renderer_{a.run}.json')
    json.dump(dict(run=a.run, note='validation split (held-out training airports), true box with fixed small jitter, 4 corners', **out), open(p, 'w'), indent=1)


if __name__ == '__main__':
    main()
