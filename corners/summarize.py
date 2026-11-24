"""Build results/results.json from the per-runway eval CSVs. Every comparison is paired on identical runways
(same detector boxes); every number carries its n; CIs are 2,000-sample paired bootstraps of the median difference.
Usage: python3 -m corners.summarize"""
import json, os
import numpy as np, pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS, RES = os.path.join(PROJ, 'runs'), os.path.join(PROJ, 'results')
SEG = [('near <2.5 km', 0, 2.5), ('mid 2.5-4.5 km', 2.5, 4.5), ('far >4.5 km', 4.5, 99)]
KEY = ['stem', 'runway']


def load(run, test):
    p = os.path.join(RUNS, run, f'eval_{test}.csv')
    if not os.path.exists(p): return None
    d = pd.read_csv(p, dtype={'runway': str})
    return d[d.odd == 'IN_ODD']


def r(x, k=2): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def stats(d, corner='corner_px', pos='pos4', lat='lat4', glide='glide4'):
    s = dict(n=int(len(d)), corner_px_median=r(d[corner].median()), corner_px_mean=r(d[corner].mean()),
             pck1=r((d[corner] <= 1).mean() * 100, 1), pck2=r((d[corner] <= 2).mean() * 100, 1),
             pos_median_m=r(d[pos].median(), 1), pos_over_1km_pct=r((d[pos] > 1000).mean() * 100, 1),
             lat_median_deg=r(d[lat].median(), 3), glide_median_deg=r(d[glide].median(), 3), by_distance={})
    for name, lo, hi in SEG:
        q = d[(d.along_km >= lo) & (d.along_km < hi)]
        s['by_distance'][name] = dict(n=int(len(q)), pos_median_m=r(q[pos].median(), 1), corner_px_median=r(q[corner].median()))
    return s


def paired(a, ca, b, cb, seed=0):
    m = a[KEY + [ca]].merge(b[KEY + [cb]], on=KEY, suffixes=('_a', '_b')).dropna()
    x, y = m.iloc[:, 2].values, m.iloc[:, 3].values
    rng = np.random.default_rng(seed); n = len(m)
    bs = [np.median(x[i]) - np.median(y[i]) for i in (rng.integers(0, n, n) for _ in range(2000))]
    return dict(n=int(n), median_a=r(np.median(x)), median_b=r(np.median(y)), diff=r(np.median(x) - np.median(y)),
                ci95=[r(np.percentile(bs, 2.5)), r(np.percentile(bs, 97.5))], a_better_pct=r((x < y).mean() * 100, 1))


def yolo_view(d):
    """YOLO columns of an eval CSV, renamed so stats() can read them (same runways and boxes as ours)."""
    m = {'yolo_corner_px': 'corner_px', 'yolo_pos': 'pos4', 'yolo_lat': 'lat4', 'yolo_glide': 'glide4'}
    return d.drop(columns=[c for c in m.values() if c in d.columns]).rename(columns=m)


def main():
    os.makedirs(RES, exist_ok=True)
    out = dict(notes=['All numbers measured; IN_ODD runways only; n stated per entry.',
                      'Crops use the released YOLO detector boxes (IoU>=0.3, score>=0.25) for every model.',
                      'YOLO keypoint comparisons use Bing only; Flight Simulator compares our own models (released YOLO may have seen it).',
                      'Released YOLO training config uses the LARD test split as validation (possible optimistic bias in YOLO numbers).',
                      '16-point PnP returned no physically valid pose on 6 of 1,086 Bing and 12 of 1,088 Flight Simulator runways; '
                      'paired 16-vs-4 comparisons drop those (n = 1,080 / 1,076), the per-model medians do not.',
                      'Run A is the only model that saw X-Plane in training; ResNet-50 and DINOv2 rows trained on the 4,599-crop Bing subset.',
                      'Validation corner errors use the true box with a fixed small jitter; test errors use detector-box crops.'])
    # 1. CNN vs ViT vs YOLO on the exact Nov 6 runways
    res, dino = load('resnet50_bing', 'bing662'), load('dinov2_bing', 'bing662')
    if res is not None:
        sec = dict(test='Bing Maps, the 662 Nov-6 test runways', yolov8_pose=stats(yolo_view(res)), resnet50=stats(res))
        hist = os.path.join(RUNS, 'dinov2_bing', 'history.json')
        if dino is not None:
            h = json.load(open(hist))
            sec['dinov2_vits14'] = stats(dino); sec['dinov2_epochs'] = f"{h['epochs_done']} of {h['epochs_planned']}"
            sec['dinov2_vs_resnet50_corner_px'] = paired(dino, 'corner_px', res, 'corner_px')
            sec['dinov2_vs_resnet50_pos_m'] = paired(dino, 'pos4', res, 'pos4')
            sec['dinov2_vs_yolo_corner_px'] = paired(dino, 'corner_px', res.assign(c=res.yolo_corner_px), 'c')
            sec['dinov2_vs_yolo_pos_m'] = paired(dino, 'pos4', res.assign(p=res.yolo_pos), 'p')
        sec['resnet50_vs_yolo_corner_px'] = paired(res, 'corner_px', res.assign(c=res.yolo_corner_px), 'c')
        sec['resnet50_vs_yolo_pos_m'] = paired(res, 'pos4', res.assign(p=res.yolo_pos), 'p')
        out['cnn_vs_vit_bing662'] = sec
    # 2. Run A: same model, 4-corner vs 16-point PnP (identical runways), then vs YOLO on the same runways
    for test in ('bing_all', 'flsim'):
        a = load('runA', test)
        if a is None: continue
        sec = dict(test=test, runA_4pt=stats(a), runA_16pt=stats(a, pos='pos16', lat='lat16', glide='glide16'),
                   pnp16_vs_pnp4_pos_m=paired(a.assign(p16=a.pos16), 'p16', a.assign(p4=a.pos4), 'p4'))
        if test == 'bing_all':
            sec['yolov8_pose_same_runways'] = stats(yolo_view(a))
            sec['runA16_vs_yolo_pos_m'] = paired(a, 'pos16', a.assign(p=a.yolo_pos), 'p')
            sec['runA_vs_yolo_corner_px'] = paired(a, 'corner_px', a.assign(c=a.yolo_corner_px), 'c')
        out[f'runA_{test}'] = sec
    hp = os.path.join(RUNS, 'runA', 'history.json')
    if os.path.exists(hp):
        h = json.load(open(hp))
        out['runA_training'] = dict(epochs=f"{h['epochs_done']} of {h['epochs_planned']}", n_train=h['n_train'], n_val=h['n_val'],
                                    history=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k != 'sec'} for e in h['history']])
    vp = os.path.join(RES, 'val_by_renderer_runA.json')
    if os.path.exists(vp): out['runA_val_by_renderer'] = json.load(open(vp))
    # 2c. All three of our models + YOLO on the same 1,086 Bing runways (larger n than the 662)
    ba = {k: load(k, 'bing_all') for k in ('resnet50_bing', 'dinov2_bing', 'runA')}
    ba = {k: v for k, v in ba.items() if v is not None}
    if len(ba) == 3:
        keys = set.intersection(*[set(map(tuple, v[KEY].values)) for v in ba.values()])
        ba = {k: v[[t in keys for t in map(tuple, v[KEY].values)]] for k, v in ba.items()}
        ref = ba['resnet50_bing']
        sec = dict(test='Bing Maps, all downloaded test runways with a detector box (same runways for every row)',
                   yolov8_pose=stats(yolo_view(ref)), **{k: stats(v) for k, v in ba.items()},
                   runA_16pt=stats(ba['runA'], pos='pos16', lat='lat16', glide='glide16'))
        for k in ba:
            sec[f'{k}_vs_yolo_corner_px'] = paired(ba[k], 'corner_px', ref.assign(c=ref.yolo_corner_px), 'c')
            sec[f'{k}_vs_yolo_pos_m'] = paired(ba[k], 'pos4', ref.assign(p=ref.yolo_pos), 'p')
        sec['dinov2_vs_resnet50_corner_px'] = paired(ba['dinov2_bing'], 'corner_px', ref, 'corner_px')
        sec['dinov2_vs_resnet50_pos_m'] = paired(ba['dinov2_bing'], 'pos4', ref, 'pos4')
        sec['runA_vs_resnet50_corner_px'] = paired(ba['runA'], 'corner_px', ref, 'corner_px')
        sec['runA_vs_resnet50_pos_m'] = paired(ba['runA'], 'pos4', ref, 'pos4')
        out['models_bing_all'] = sec
    # 3. Unseen renderer (Flight Simulator): our models only
    fl = {k: load(k, 'flsim') for k in ('resnet50_bing', 'dinov2_bing', 'runA')}
    fl = {k: v for k, v in fl.items() if v is not None}
    if fl:
        out['unseen_renderer_flsim'] = {k: stats(v) for k, v in fl.items()}
        if 'resnet50_bing' in fl and 'dinov2_bing' in fl:
            out['unseen_renderer_flsim']['dinov2_vs_resnet50_corner_px'] = paired(fl['dinov2_bing'], 'corner_px', fl['resnet50_bing'], 'corner_px')
            out['unseen_renderer_flsim']['dinov2_vs_resnet50_pos_m'] = paired(fl['dinov2_bing'], 'pos4', fl['resnet50_bing'], 'pos4')
        if 'resnet50_bing' in fl and 'runA' in fl:
            out['unseen_renderer_flsim']['runA_vs_resnet50_corner_px'] = paired(fl['runA'], 'corner_px', fl['resnet50_bing'], 'corner_px')
            out['unseen_renderer_flsim']['runA_vs_resnet50_pos_m'] = paired(fl['runA'], 'pos4', fl['resnet50_bing'], 'pos4')
    import glob
    out['multirunway_and_wrong_runway'] = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, 'multirunway_*.json')))]
    json.dump(out, open(os.path.join(RES, 'results.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == '__main__':
    main()
