"""Show what one LARD V2 label looks like: the image, the 4 labelled corners and the true distance.

    python3 demo/show_label.py
"""
import os, subprocess, sys, warnings
warnings.filterwarnings('ignore')
import cv2
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from lard_geom import load_labels, DATA, corners_of, RunwayFrame  # noqa: E402

STEM = 'KPBI-14-32-10L-28R__10-smpl__10h40_051'
row = load_labels('bingmaps', 'test').query('stem == @STEM').iloc[0].to_dict()
c = corners_of(row)
print(f"Image   : {row['image']}  (1024 x 1024 colour)")
print(f"Airport : {row['airport']}   Runway: {row['runway']}")
for name, (x, y) in zip(['TL (far left)', 'TR (far right)', 'BR (near right)', 'BL (near left)'], c):
    print(f"  {name:<16} x = {x:4.0f}  y = {y:4.0f}")
fr = RunwayFrame('bingmaps', row)
print(f"True distance to threshold: {fr.guidance(fr.gt)[0]:.0f} m   (runway is {c[:, 1].max() - c[:, 1].min():.0f} px tall in the image)")
img = cv2.imread(os.path.join(DATA, row['image']))
x0, y0 = c.min(0).astype(int) - 40; x1, y1 = c.max(0).astype(int) + 40
for (x, y), n in zip(c.astype(int), ['TL', 'TR', 'BR', 'BL']):
    cv2.circle(img, (x, y), 2, (0, 200, 255), -1)
zoom = cv2.resize(img[y0:y1, x0:x1], None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
for (x, y), n in zip(c, ['TL', 'TR', 'BR', 'BL']):
    p = (int((x - x0) * 8), int((y - y0) * 8))
    cv2.circle(zoom, p, 9, (0, 200, 255), -1)
    cv2.putText(zoom, n, (p[0] + 12, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
os.makedirs(os.path.join(PROJ, 'demo', 'out'), exist_ok=True)
out = os.path.join(PROJ, 'demo', 'out', 'label_example.png')
cv2.imwrite(out, zoom)
print(f"Zoomed view of the labelled corners (8x): {out}")
if sys.platform == 'darwin' and not os.environ.get('NO_OPEN'):
    subprocess.run(['open', os.path.join(DATA, row['image']), out])
