"""Figures for the progress update. Usage: python3 report/make_figures.py"""
import json, os, sys
import cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from lard_geom import DATA, corners_of, load_labels  # noqa: E402

OUT = os.path.join(PROJ, 'report', 'figs'); os.makedirs(OUT, exist_ok=True)
RUN = os.path.join(PROJ, 'runs', 'resnet50_bing')
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0'
C_TRUE, C_YOLO, C_OURS = '#8a8984', '#eb6834', '#2a78d6'          # neutral / slot-2 orange / slot-1 blue
SEG = [(0, 2500, 'Near\n< 2.5 km'), (2500, 4500, 'Mid\n2.5–4.5 km'), (4500, 1e5, 'Far\n> 4.5 km')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    ax.grid(axis='y', color=GRID, lw=1); ax.set_axisbelow(True)


def grouped(ax, groups, series, ymax, unit=' m'):
    n = len(series); w = 0.8 / n
    for k, (name, vals, col) in enumerate(series):
        xs = np.arange(len(groups)) + (k - (n - 1) / 2) * w
        ax.bar(xs, vals, w * 0.86, color=col, label=name, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + ymax * 0.012, f'{v:,.0f}{unit}', ha='center', va='bottom', fontsize=10.5, color=INK)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, color=INK2); ax.set_ylim(0, ymax)


def seg_medians(df, col, along_col='along_nm'):
    m = df[along_col] * 1852
    return [df.loc[(m >= a) & (m < b), col].median() for a, b, _ in SEG]


# ---- Figure 1: baseline across the whole LARD V2 test set
b = pd.read_csv(os.path.join(PROJ, 'baselines', 'results_yolov8pose.csv'))
b = b[(b.odd == 'IN_ODD') & b.detected]
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200); fig.patch.set_facecolor(SURF); style(ax)
grouped(ax, [s[2] for s in SEG], [('True corners (best possible)', seg_medians(b, 'true_pos'), C_TRUE),
                                  ('YOLOv8-Pose corners (released LARD model)', seg_medians(b, 'pose_pos'), C_YOLO)], 700)
ax.set_ylabel('Median aircraft-position error (m)', color=INK2)
ax.legend(frameon=False, loc='upper left', fontsize=11)
ax.text(0.99, 0.97, f'Box corners only: {b.boxpred_pos.median():,.0f} m (off chart)\n'
        f'n = {len(b):,} runways · 5 renderers · unseen airports', transform=ax.transAxes, ha='right', va='top', fontsize=10.5, color=INK2)
fig.text(0.07, 0.97, 'Baseline: corner errors become hundreds of metres far out', fontsize=15, fontweight='bold', color=INK, va='top')
fig.text(0.07, 0.915, f'PnP on the full LARD V2 test set · YOLOv8-Pose median corner error {b.pose_corner_px.median():.2f} px '
         f'→ {b.pose_pos.median():.0f} m overall', fontsize=10.5, color=INK2, va='top')
fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(os.path.join(OUT, 'fig1_baseline_by_distance.png'), facecolor=SURF); plt.close(fig)

# ---- Figure 2: training curve
if os.path.exists(os.path.join(RUN, 'history.json')):
    h = pd.DataFrame(json.load(open(os.path.join(RUN, 'history.json'))))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200); fig.patch.set_facecolor(SURF); style(ax)
    ax.plot(h.epoch, h.val_corner_px_median, color=C_OURS, lw=2.5, marker='o', ms=6, mec=SURF, mew=2, zorder=3)
    ax.axhline(b.pose_corner_px.median(), color=C_YOLO, lw=1.5, ls=(0, (4, 3)))
    ax.text(h.epoch.max(), b.pose_corner_px.median() + 0.03, 'YOLOv8-Pose (test set)', ha='right', va='bottom', color=INK2, fontsize=10)
    ax.set_xlabel('Epoch', color=INK2); ax.set_ylabel('Median corner error (px)', color=INK2)
    ax.set_title('ResNet-50 corner model: validation error (held-out airports)', loc='left', fontsize=12.5, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_training_curve.png'), facecolor=SURF); plt.close(fig)

# ---- Figure 3 + 4: ours vs YOLOv8-Pose on the same Bing test runways
tr = os.path.join(RUN, 'test_results.csv')
if os.path.exists(tr):
    t = pd.read_csv(tr); t = t[t.odd == 'IN_ODD']
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200); fig.patch.set_facecolor(SURF); style(ax)
    ymax = max(seg_medians(t, 'yolo_pos') + seg_medians(t, 'ours_pos')) * 1.25
    grouped(ax, [s[2] for s in SEG], [('True corners', seg_medians(t, 'true_pos'), C_TRUE),
                                      ('YOLOv8-Pose', seg_medians(t, 'yolo_pos'), C_YOLO),
                                      ('Ours: crop + ResNet-50 heatmaps (first run)', seg_medians(t, 'ours_pos'), C_OURS)], ymax)
    ax.set_ylabel('Median aircraft-position error (m)', color=INK2); ax.legend(frameon=False, loc='upper left', fontsize=11)
    fig.text(0.07, 0.97, 'First model vs. published baseline (same runways, same detector boxes)', fontsize=15, fontweight='bold', color=INK, va='top')
    fig.text(0.07, 0.915, f'Bing Maps test subset, n = {len(t):,} runways · corner error: ours {t.ours_corner_px.median():.2f} px vs '
             f'YOLOv8-Pose {t.yolo_corner_px.median():.2f} px', fontsize=10.5, color=INK2, va='top')
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(os.path.join(OUT, 'fig3_ours_vs_yolov8pose.png'), facecolor=SURF); plt.close(fig)

    # qualitative: test runways at increasing distance; true (white), YOLOv8-Pose (orange), ours (blue)
    picks = t.sort_values('along_nm').iloc[np.linspace(len(t) * 0.1, len(t) * 0.95, 4).astype(int)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=180); fig.patch.set_facecolor(SURF)
    for ax, (_, r) in zip(axes, picks.iterrows()):
        img = cv2.cvtColor(cv2.imread(os.path.join(DATA, r.image_path)), cv2.COLOR_BGR2RGB)
        g, y, o = (np.array(json.loads(r[c])) for c in ('gt_xy', 'yolo_xy', 'ours_xy'))
        c = g.mean(0); h = max(np.ptp(g[:, 0]), np.ptp(g[:, 1])) * 0.9 + 6
        ax.imshow(img, interpolation='nearest'); ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] + h, c[1] - h)
        for pts, col, lw in ((g, 'white', 2.2), (y, C_YOLO, 1.6), (o, C_OURS, 1.6)):
            q = np.vstack([pts, pts[:1]]); ax.plot(q[:, 0], q[:, 1], color=col, lw=lw)
        ax.set_title(f"{r.along_nm * 1.852:.1f} km out\nYOLO {r.yolo_pos:,.0f} m · ours {r.ours_pos:,.0f} m", fontsize=11, color=INK)
        ax.axis('off')
    fig.text(0.01, 0.02, 'White = true corners · orange = YOLOv8-Pose · blue = ours', fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT, 'fig4_examples.png'), facecolor=SURF, bbox_inches='tight', pad_inches=0.15); plt.close(fig)
print('figures in', OUT)
