"""Assets for the progress-update webpage: multi-runway examples + model stage images for the simulation."""
import json, os, sys
import cv2, numpy as np, pandas as pd, torch
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, PROJ)
from lard_geom import DATA, corners_of, load_labels  # noqa: E402
from corners.data import RunwayCrops, crop_box, affine, CROP  # noqa: E402
from corners.model import CornerNet  # noqa: E402
OUT = os.path.join(PROJ, 'report', 'web', 'img')
WHITE, BLUE, ORANGE = (255, 255, 255), (42, 120, 214), (235, 104, 52)

def quad(img, pts, col, w):
    cv2.polylines(img, [np.round(pts * 8).astype(np.int32)], True, col, w, cv2.LINE_AA, shift=3)

# ---- 1. multi-runway examples
mr = pd.read_csv(os.path.join(PROJ, 'improvements', 'multirunway_results.csv'))
t = pd.read_csv(os.path.join(PROJ, 'runs', 'resnet50_bing', 'test_results.csv'), dtype={'runway': str})
want = [('KOAK', '28L + 28R'), ('CYYZ', '06R + 06L'), ('GMMN', '35R + 35L'), ('VECC', '19L + 19R')]
cards = []
for apt, rws in want:
    sub = mr[(mr.airport == apt) & (mr.runways == rws)]
    r = sub.iloc[0] if apt != 'VECC' else sub.sort_values('ours_joint').iloc[-1]   # include one case where joint is worse
    img = cv2.cvtColor(cv2.imread(os.path.join(DATA, r.image)), cv2.COLOR_BGR2RGB)
    stem = os.path.basename(r.image).rsplit('_', 1)[0]
    rr = t[t.stem == stem]
    allp = np.vstack([np.array(json.loads(x)) for x in rr.gt_xy])
    c = allp.mean(0); h = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) * 0.75 + 20
    x0, y0 = int(max(c[0] - h, 0)), int(max(c[1] - h, 0)); x1, y1 = int(min(c[0] + h, 1024)), int(min(c[1] + h, 1024))
    crop = np.ascontiguousarray(img[y0:y1, x0:x1]); s = 480 / crop.shape[1]
    crop = cv2.resize(crop, (480, int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    for _, q in rr.iterrows():
        quad(crop, (np.array(json.loads(q.gt_xy)) - [x0, y0]) * s, WHITE, 3)
        quad(crop, (np.array(json.loads(q.ours_xy)) - [x0, y0]) * s, BLUE, 2)
    fn = f'multi_{apt}.jpg'; cv2.imwrite(os.path.join(OUT, fn), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cards.append(dict(img=fn, airport=apt, runways=rws, km=round(r.along_km, 1), single=round(r.ours_rw1), joint=round(r.ours_joint)))

# ---- 2. simulation examples (near / mid / far) with model heatmaps
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = CornerNet(pretrained=False).to(dev); model.load_state_dict(torch.load(os.path.join(PROJ, 'runs', 'resnet50_bing', 'best.pt'), map_location=dev)); model.eval()
ti = t[t.odd == 'IN_ODD'].copy()
picks = []
for lo, hi in [(1.0, 2.0), (3.2, 4.2), (5.0, 6.2)]:
    s_ = ti[(ti.along_nm * 1.852 > lo) & (ti.along_nm * 1.852 < hi)]
    s_ = s_.iloc[(s_.ours_pos - s_.ours_pos.median()).abs().argsort()[:1]]      # a typical (median-error) case
    picks.append(s_.iloc[0])
labels = load_labels('bingmaps', 'test')
sims = []
for k, r in enumerate(picks):
    img = cv2.cvtColor(cv2.imread(os.path.join(DATA, r.image_path)), cv2.COLOR_BGR2RGB)
    gt, ours, yolo = (np.array(json.loads(r[c])) for c in ('gt_xy', 'ours_xy', 'yolo_xy'))
    box = (*yolo.min(0), *yolo.max(0))
    cx, cy, side = crop_box(box); A = affine(cx, cy, side)
    full = img.copy(); cv2.rectangle(full, (int(cx - side / 2), int(cy - side / 2)), (int(cx + side / 2), int(cy + side / 2)), (242, 169, 0), 4)
    cv2.imwrite(os.path.join(OUT, f'sim{k}_frame.jpg'), cv2.cvtColor(cv2.resize(full, (640, 640), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    crop = cv2.warpAffine(img, A, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    big = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, f'sim{k}_crop.jpg'), cv2.cvtColor(big, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    x = torch.from_numpy(((crop.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1).astype(np.float32))[None]
    with torch.no_grad(): _, prob = model(x.to(dev))
    hm = prob[0].cpu().numpy().sum(0); hm = (hm / hm.max()) ** 0.6
    hm = cv2.resize(hm, (512, 512), interpolation=cv2.INTER_CUBIC)
    col = cv2.applyColorMap((np.clip(hm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    over = (big * 0.45 + col * 0.75).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, f'sim{k}_heat.jpg'), cv2.cvtColor(over, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    tocrop = lambda p: (p @ A[:, :2].T + A[:, 2]) * 2                           # full px -> 512 display px
    lab = labels[(labels.stem == r.stem) & (labels.runway == r.runway)].iloc[0]
    sims.append(dict(airport=lab.airport, runway=r.runway, km=round(r.along_nm * 1.852, 2),
                     lat_true=round(float(lab.lateral_path_angle), 2), glide_true=round(float(lab.vertical_path_angle), 2),
                     height_ft=round(float(lab.height_above_runway)),
                     gt=tocrop(gt).round(1).tolist(), ours=tocrop(ours).round(1).tolist(), yolo=tocrop(yolo).round(1).tolist(),
                     ours_px=round(r.ours_corner_px, 2), yolo_px=round(r.yolo_corner_px, 2),
                     ours_pos=round(r.ours_pos), yolo_pos=round(r.yolo_pos), true_pos=round(r.true_pos),
                     ours_lat=round(r.ours_lat, 2), ours_glide=round(r.ours_glide, 2), side_px=round(side)))
curve = pd.read_csv(os.path.join(os.path.dirname(PROJ), 'lard_output_example', 'pixel_vs_landing.csv'))
json.dump(dict(multi=cards, sims=sims, curve=curve.to_dict('list')), open(os.path.join(PROJ, 'report', 'web', 'data.json'), 'w'), indent=1)
print(json.dumps(dict(multi=cards, sims=[{k: v for k, v in s.items() if k not in ('gt', 'ours', 'yolo')} for s in sims]), indent=1))
