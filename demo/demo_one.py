"""Live demo on ONE Bing test image (unseen airport): crop -> 3 trained models -> corners -> PnP -> guidance.

    python3 demo/demo_one.py --pick near        # also: mid, far, random  (--seed N picks a different image)

Prints true vs predicted distance / lateral / glide for YOLOv8-Pose, ResNet-50, DINOv2 and Run A,
and saves + opens a figure: full image with the YOLO box, the crop with corners, Run A's heatmaps.
"""
import argparse, json, os, subprocess, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import cv2

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, 'baselines'))
from corners.data import RunwayCrops, MEAN, STD  # noqa: E402
from corners.model import CornerNet  # noqa: E402
from corners.train import to_full  # noqa: E402
from lard_geom import load_labels, DATA, RunwayFrame, corners_of  # noqa: E402
import eval_yolov8pose as B  # noqa: E402

RUNS = [('ResNet-50', 'resnet50_bing'), ('DINOv2', 'dinov2_bing'), ('Run A', 'runA')]
SEG = {'near': (0, 2.5), 'mid': (2.5, 4.5), 'far': (4.5, 99), 'random': (0, 99)}


def load_model(run, dev):
    h = json.load(open(os.path.join(PROJ, 'runs', run, 'history.json')))
    h = h if isinstance(h, dict) else {'backbone': 'resnet50', 'n_kp': 4}
    m = CornerNet(pretrained=False, backbone=h.get('backbone', 'resnet50'), n_kp=h.get('n_kp', 4)).to(dev)
    m.load_state_dict(torch.load(os.path.join(PROJ, 'runs', run, 'best.pt'), map_location=dev)); m.eval()
    return m, h.get('n_kp', 4)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pick', default='near', choices=list(SEG)); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    ev = pd.read_csv(os.path.join(PROJ, 'runs', 'runA', 'eval_bing_all.csv'), dtype={'runway': str})
    lo, hi = SEG[a.pick]
    ev = ev[(ev.odd == 'IN_ODD') & (ev.along_km >= lo) & (ev.along_km < hi)]
    pick = ev.sample(1, random_state=a.seed).iloc[0]
    lab = load_labels('bingmaps', 'test')
    row = lab[(lab.stem == pick.stem) & (lab.runway == pick.runway)].iloc[0].to_dict()
    gt = corners_of(row); gb = (*gt.min(0), *gt.max(0))
    _, box, ykp = max(((B.iou(gb, b), b, kp) for s, b, kp in B.load_preds()[row['stem'] + '_BingMaps']), key=lambda x: x[0])
    ykp = np.asarray(ykp, float)
    print(f"\nImage: {row['image']}\nAirport {row['airport']}, runway {row['runway']}, {pick.along_km:.1f} km out (test airport, never seen in training)")
    print(f"YOLO box: x {box[0]:.0f}-{box[2]:.0f}, y {box[1]:.0f}-{box[3]:.0f} px  ->  runway is {box[2]-box[0]:.0f} x {box[3]-box[1]:.0f} px in a 1024 x 1024 image\n")

    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    fr = RunwayFrame('bingmaps', row)
    true_g = fr.guidance(fr.gt)
    results, heat = [('True (label)', gt, fr.gt)], None
    results.append(('YOLOv8-Pose', ykp, fr.solve(ykp)))
    for name, run in RUNS:
        model, n_kp = load_model(run, dev)
        x, _, A, _ = RunwayCrops([row], boxes={0: box}, n_kp=n_kp)[0]
        with torch.no_grad():
            kp, prob = model(x[None].to(dev))
        pts = to_full(kp.cpu(), A[None])[0, :4].numpy().astype(float)
        results.append((name, pts, fr.solve(pts)))
        if run == 'runA':
            heat, crop, Acrop = prob[0, :4].sum(0).cpu().numpy(), x.numpy(), A.numpy()

    print(f"{'Model':<14}{'corner err px':>14}{'distance m':>12}{'lateral °':>11}{'glide °':>9}{'position err m':>16}")
    for name, pts, cam in results:
        ce = np.linalg.norm(pts - gt, axis=1).mean()
        if cam is None:
            print(f'{name:<14}{ce:>14.2f}   PnP found no valid pose'); continue
        d, la, gl = fr.guidance(cam)
        print(f"{name:<14}{ce:>14.2f}{d:>12.0f}{la:>11.2f}{gl:>9.2f}{np.linalg.norm(cam - fr.gt):>16.0f}")
    print(f"\n(distance = metres to the threshold along the centre line; true values: {true_g[0]:.0f} m, {true_g[1]:.2f}°, {true_g[2]:.2f}°)")

    img = cv2.cvtColor(cv2.imread(os.path.join(DATA, row['image'])), cv2.COLOR_BGR2RGB)
    rgb = np.clip(crop.transpose(1, 2, 0) * STD + MEAN, 0, 1)
    to_crop = lambda p: p @ Acrop[:, :2].T + Acrop[:, 2]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    ax[0].imshow(img); ax[0].add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec='orange', lw=2))
    ax[0].set_title(f"1. Camera image 1024², YOLO box ({pick.along_km:.1f} km out)")
    ax[1].imshow(rgb)
    for (name, pts, _), c in zip([results[0], results[1], results[4]], ['white', 'orange', 'deepskyblue']):
        q = to_crop(pts); q = np.vstack([q, q[:1]]); ax[1].plot(q[:, 0], q[:, 1], '-o', color=c, lw=2, ms=4, label=name)
    ax[1].legend(loc='lower right', fontsize=10); ax[1].set_title('2. Crop 256², corners: true / YOLO / Run A')
    ax[2].imshow(rgb); ax[2].imshow(cv2.resize(heat, (256, 256)), alpha=0.6, cmap='magma')
    ax[2].set_title('3. Run A heatmaps (4 corners, summed)')
    for x_ in ax: x_.axis('off')
    os.makedirs(os.path.join(PROJ, 'demo', 'out'), exist_ok=True)
    out = os.path.join(PROJ, 'demo', 'out', f"demo_{a.pick}_{a.seed}.png")
    plt.tight_layout(); plt.savefig(out, dpi=110); print(f'\nFigure: {out}')
    if sys.platform == 'darwin' and not os.environ.get('NO_OPEN'): subprocess.run(['open', out])


if __name__ == '__main__':
    main()
