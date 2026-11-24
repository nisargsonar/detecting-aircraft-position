"""Train a corner model.  Usage:
    python3 -m corners.train --epochs 10                                   # ResNet-50, 4 corners, Bing subset (Nov 5 run)
    python3 -m corners.train --backbone dinov2_vits14 --out dinov2_bing --deadline 07:00
Checkpoints every epoch (last.pt, resumable) and keeps the best-validation model (best.pt)."""
import argparse, datetime, json, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader

from .data import RunwayCrops
from .model import CornerNet, loss_fn

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from lard_geom import load_labels, DATA  # noqa: E402

OUT = os.path.join(PROJ, 'runs', 'resnet50_bing')          # the Nov 5 ResNet-50 run (used by evaluate.py)


def rows_for(split_imgs, labels):
    keep = set(split_imgs)
    d = labels[labels.image.isin(keep)]
    d = d[[os.path.exists(os.path.join(DATA, p)) for p in d.image]]
    return d.to_dict('records')


def to_full(kp, A):
    """crop px -> full-image px (A is the 2x3 full->crop affine)."""
    s = A[:, 0, 0][:, None, None]; t = A[:, :, 2][:, None, :]
    return (kp - t) / s


def bing_subset_rows():
    """The exact 4,599 train / 723 val crops of the Nov 5 ResNet-50 run (Bing subset, split_bing.json)."""
    split = json.load(open(os.path.join(DATA, 'split_bing.json')))
    labels = load_labels('bingmaps', 'train')
    return rows_for(split['train'], labels), rows_for(split['val'], labels)


def multi_rows(renderers=('bingmaps', 'xplane')):
    """Run-A data from split_multi.json (same airport split for every renderer); only images present on disk."""
    sp = json.load(open(os.path.join(DATA, 'split_multi.json')))
    keys = {'bingmaps': ('bing_train_all', 'bing_val'), 'xplane': ('xplane_train', 'xplane_val')}
    tr, va = [], []
    for r in renderers:
        lab = load_labels(r, 'train').dropna(subset=['lat'])
        tr += rows_for(sp[keys[r][0]], lab); va += rows_for(sp[keys[r][1]], lab)
    return tr, va


def deadline_ts(hhmm):
    if not hhmm: return None
    now = datetime.datetime.now(); h, m = map(int, hhmm.split(':'))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now: t += datetime.timedelta(days=1)
    return t.timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='resnet50'); ap.add_argument('--out', default='resnet50_bing')
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--backbone-lr', type=float, default=None)
    ap.add_argument('--deadline', default=None, help='HH:MM; do not start an epoch that would end after this')
    ap.add_argument('--n-kp', type=int, default=4); ap.add_argument('--data', default='bing_subset', choices=['bing_subset', 'bing_xplane'])
    a = ap.parse_args()
    out = os.path.join(PROJ, 'runs', a.out); os.makedirs(out, exist_ok=True)
    tr, va = bing_subset_rows() if a.data == 'bing_subset' else multi_rows()
    print(f'{a.backbone} n_kp={a.n_kp} data={a.data}: train labels {len(tr)}  val labels {len(va)}', flush=True)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(0)
    model = CornerNet(backbone=a.backbone, n_kp=a.n_kp).to(dev)
    blr = a.backbone_lr if a.backbone_lr is not None else (1e-5 if a.backbone.startswith('dinov2') else a.lr)
    groups = [{'params': model.backbone.parameters(), 'lr': blr},
              {'params': list(model.head.parameters()) + [model.beta], 'lr': a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [blr, a.lr], total_steps=steps, pct_start=0.15)
    hist, best, start = [], 1e9, 0
    last = os.path.join(out, 'last.pt')
    if os.path.exists(last):                                                  # resume
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        hist, best, start = ck['hist'], ck['best'], ck['epoch']
        print(f'resumed after epoch {start}', flush=True)
    dl_tr = DataLoader(RunwayCrops(tr, train=True, seed=start, n_kp=a.n_kp), a.bs, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(RunwayCrops(va, n_kp=a.n_kp), a.bs, num_workers=0)
    dl_ts, ep_sec = deadline_ts(a.deadline), None
    for ep in range(start, a.epochs):
        if dl_ts and ep_sec and time.time() + ep_sec > dl_ts:
            print(f'deadline {a.deadline}: stopping before epoch {ep + 1} ({ep} of {a.epochs} epochs done)', flush=True); break
        model.train(); t0 = time.time(); ls = []
        for x, kp, _, _ in dl_tr:
            x, kp = x.to(dev), kp.to(dev)
            pk, prob = model(x)
            loss, _, _ = loss_fn(pk, prob, kp)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ls.append(loss.item())
        model.eval(); errs, vls = [], []
        with torch.no_grad():
            for x, kp, A, _ in dl_va:
                pk, prob = model(x.to(dev))
                vls.append(loss_fn(pk, prob, kp.to(dev))[0].item())
                errs.append((to_full(pk.cpu(), A) - to_full(kp, A)).norm(dim=-1)[:, :4].mean(1))   # 4 corners: comparable across runs
        e = torch.cat(errs).numpy(); ep_sec = time.time() - t0
        hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(ls)), val_loss=float(np.mean(vls)),
                         val_corner_px_median=float(np.median(e)), val_corner_px_mean=float(e.mean()), sec=ep_sec))
        print(json.dumps(hist[-1]), flush=True)
        if np.median(e) < best:
            best = float(np.median(e)); torch.save(model.state_dict(), os.path.join(out, 'best.pt'))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        hist=hist, best=best, epoch=ep + 1), last)
        json.dump(dict(backbone=a.backbone, n_kp=a.n_kp, data=a.data, n_train=len(tr), n_val=len(va), epochs_planned=a.epochs, epochs_done=ep + 1, history=hist),
                  open(os.path.join(out, 'history.json'), 'w'), indent=1)
    print('best val median corner error (px, full image):', round(best, 3), flush=True)


if __name__ == '__main__':
    main()
