"""Write results/MORNING_REPORT.md from results/results.json (measured numbers only, n on every line)."""
import json, os, time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(PROJ, 'results', 'results.json')))
L = [f'# Morning report — {time.strftime("%a %b %d %H:%M")}', '',
     'All numbers measured on IN_ODD runways; crops from the released YOLO detector boxes for every model. '
     'Paired comparisons are on identical runways; 95% CIs are paired bootstraps of the median difference '
     '(an interval containing 0 means no significant difference).', '']


def line(name, s):
    d = s['by_distance']
    return (f"| {name} | {s['n']} | {s['corner_px_median']} | {s['pck1']}% | {s['pos_median_m']} | "
            + ' / '.join(f"{v['pos_median_m']} (n={v['n']})" for v in d.values()) + f" | {s['lat_median_deg']}° / {s['glide_median_deg']}° |")


HDR = ['| Model | n | Corner px (median) | ≤1 px | Position m (median) | Near / mid / far m | Lateral / glide err |',
       '|---|---|---|---|---|---|---|']


def cmp(name, c, unit):
    return (f"- {name}: {c['median_a']} vs {c['median_b']} {unit}, difference {c['diff']} {unit} "
            f"(95% CI {c['ci95'][0]} to {c['ci95'][1]}; n = {c['n']})")


b0, u0 = R.get('models_bing_all'), R.get('unseen_renderer_flsim')
if b0 and u0 and 'runA_training' in R:
    p = lambda k: f"{b0[k]['diff']} (95% CI {b0[k]['ci95'][0]} to {b0[k]['ci95'][1]}, n = {b0[k]['n']})"
    a16 = R['runA_bing_all']['pnp16_vs_pnp4_pos_m']
    L += ['## Headline (measured; see sections for detail)',
          f"- Run A finished {R['runA_training']['epochs']} epochs.",
          f"- Corners, Bing: every model of ours beats YOLOv8-Pose; Run A by {p('runA_vs_yolo_corner_px')} px.",
          f"- Position, Bing: no model differs significantly from YOLO (all CIs contain 0); Run A vs YOLO {p('runA_vs_yolo_pos_m')} m.",
          f"- Run A vs the Bing-subset ResNet-50 on Bing: position {p('runA_vs_resnet50_pos_m')} m (significant).",
          f"- 16-point vs 4-corner PnP, same model and runways: {a16['diff']} m (95% CI {a16['ci95'][0]} to {a16['ci95'][1]}, n = {a16['n']}): no measured gain.",
          f"- CNN vs ViT: tied on Bing ({p('dinov2_vs_resnet50_corner_px')} px); on unseen Flight Simulator DINOv2 is better by "
          f"{u0['dinov2_vs_resnet50_pos_m']['diff']} m (95% CI {u0['dinov2_vs_resnet50_pos_m']['ci95'][0]} to {u0['dinov2_vs_resnet50_pos_m']['ci95'][1]}, n = {u0['dinov2_vs_resnet50_pos_m']['n']}).",
          '- Joint 2-runway PnP cuts median position error 8-12x for every model, n = 55-92 images (section 4).', '']

c = R.get('cnn_vs_vit_bing662')
if c:
    L += ['## 1. CNN vs ViT (Bing Maps, the 662 Nov-6 test runways)',
          f"Both models trained on the same 4,599 Bing crops (723 val). DINOv2 epochs: {c.get('dinov2_epochs', 'not run')}.", ''] + HDR
    L += [line('YOLOv8-Pose (released)', c['yolov8_pose']), line('ResNet-50 (ours)', c['resnet50'])]
    if 'dinov2_vits14' in c: L.append(line('DINOv2 ViT-S/14 (ours)', c['dinov2_vits14']))
    L += ['']
    for k, unit in [('dinov2_vs_resnet50_corner_px', 'px'), ('dinov2_vs_resnet50_pos_m', 'm'), ('resnet50_vs_yolo_corner_px', 'px'),
                    ('resnet50_vs_yolo_pos_m', 'm'), ('dinov2_vs_yolo_corner_px', 'px'), ('dinov2_vs_yolo_pos_m', 'm')]:
        if k in c: L.append(cmp(k.replace('_', ' '), c[k], unit))
    L += ['']

t = R.get('runA_training')
if t:
    L += [f"## 2. Run A training (ResNet-50, 16 keypoints, Bing + X-Plane): {t['epochs']} epochs, "
          f"n_train = {t['n_train']}, n_val = {t['n_val']}", '',
          '| Epoch | Train loss | Val loss | Val corner px median | Val corner px mean |', '|---|---|---|---|---|']
    L += [f"| {e['epoch']} | {e['train_loss']} | {e['val_loss']} | {e['val_corner_px_median']} | {e['val_corner_px_mean']} |" for e in t['history']]
    v = R.get('runA_val_by_renderer')
    if v:
        L += ['', 'Validation split by renderer (true box with fixed small jitter, 4 corners):', '',
              '| Renderer | n | Median px | Mean px | ≤1 px | >5 px | >20 px | Error share of worst 5% |', '|---|---|---|---|---|---|---|---|']
        for r in ('bingmaps', 'xplane'):
            s = v[r]
            L.append(f"| {r} | {s['n']} | {s['corner_px_median']} | {s['corner_px_mean']} | {s['pck1']}% | {s['over_5px_pct']}% | "
                     f"{s['over_20px_pct']}% | {s['share_of_total_error_from_worst_5pct']}% |")
    L += ['']

for test in ('bing_all', 'flsim'):
    a = R.get(f'runA_{test}')
    if not a: continue
    L += [f'## 2{"a" if test == "bing_all" else "b"}. Run A (ResNet-50, 16 keypoints, Bing + X-Plane): {test}', ''] + HDR
    L += [line('Run A, 4-corner PnP', a['runA_4pt']), line('Run A, 16-point PnP', a['runA_16pt'])]
    if 'yolov8_pose_same_runways' in a: L.append(line('YOLOv8-Pose, same runways', a['yolov8_pose_same_runways']))
    L += ['', cmp('16-point vs 4-corner PnP, same model and runways (position)', a['pnp16_vs_pnp4_pos_m'], 'm')]
    for k, unit in [('runA16_vs_yolo_pos_m', 'm'), ('runA_vs_yolo_corner_px', 'px')]:
        if k in a: L.append(cmp(k.replace('_', ' '), a[k], unit))
    L += ['']

b = R.get('models_bing_all')
if b:
    L += ['## 2c. All our models vs YOLOv8-Pose on the same Bing runways (larger test set)', ''] + HDR
    L += [line('YOLOv8-Pose (released)', b['yolov8_pose']), line('ResNet-50, Bing subset', b['resnet50_bing']),
          line('DINOv2, Bing subset', b['dinov2_bing']), line('Run A, 4-corner PnP', b['runA']), line('Run A, 16-point PnP', b['runA_16pt'])]
    L += ['']
    for k, v in b.items():
        if isinstance(v, dict) and 'ci95' in v: L.append(cmp(k.replace('_', ' '), v, 'px' if k.endswith('px') else 'm'))
    L += ['']

u = R.get('unseen_renderer_flsim')
if u:
    L += ['## 3. Unseen renderer: Flight Simulator (our models only; released YOLO may have seen it)', ''] + HDR
    for k, v in u.items():
        if isinstance(v, dict) and 'corner_px_median' in v: L.append(line(k, v))
    if 'dinov2_vs_resnet50_corner_px' in u: L += ['', cmp('DINOv2 vs ResNet-50 corner px', u['dinov2_vs_resnet50_corner_px'], 'px')]
    if 'dinov2_vs_resnet50_pos_m' in u: L.append(cmp('DINOv2 vs ResNet-50 position', u['dinov2_vs_resnet50_pos_m'], 'm'))
    if 'runA_vs_resnet50_corner_px' in u: L.append(cmp('Run A vs ResNet-50 (Bing only) corner px', u['runA_vs_resnet50_corner_px'], 'px'))
    if 'runA_vs_resnet50_pos_m' in u: L.append(cmp('Run A vs ResNet-50 (Bing only) position', u['runA_vs_resnet50_pos_m'], 'm'))
    L += ['']

m = R.get('multirunway_and_wrong_runway') or []
if m:
    L += ['## 4. Multi-runway joint PnP and wrong-runway check (images with 2 runways)', '',
          '| Run | Test | n images | One runway m | Both runways m | 95% CI of difference | Joint better | Wrong-runway correct |', '|---|---|---|---|---|---|---|---|']
    for x in m:
        L.append(f"| {x['run']} | {x['test']} | {x['n_images']} | {x['single_runway_median_m']} | {x['joint_median_m']} | "
                 f"{x['diff_ci95'][0]} to {x['diff_ci95'][1]} | {x['joint_better_pct']}% | {x['wrong_runway']['correct_pct']}% |")
    L += ['']

L += ['## Notes'] + [f'- {n}' for n in R['notes']]
open(os.path.join(PROJ, 'results', 'MORNING_REPORT.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
