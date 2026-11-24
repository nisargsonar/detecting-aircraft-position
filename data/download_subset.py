"""Download a Bing Maps subset of LARD V2: train / val (held-out train airports) / test, split by airport."""
import os, json, urllib.request, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
HF = 'https://huggingface.co/datasets/DEEL-AI/LARD_V2/resolve/main/'
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
tr = pd.read_csv(f'{HERE}/m_bingmaps_train.csv', sep=';'); te = pd.read_csv(f'{HERE}/m_bingmaps_test.csv', sep=';')
airports = np.array(sorted(tr.airport.unique())); rng.shuffle(airports)
val_ap = set(airports[:10])                                   # 10 held-out training airports -> validation
pick = lambda d, n: rng.choice(sorted(d.image.unique()), min(n, d.image.nunique()), replace=False)
split = {'train': pick(tr[~tr.airport.isin(val_ap)], 4000),
         'val':   pick(tr[tr.airport.isin(val_ap)], 600),
         'test':  pick(te, 1200)}
json.dump({k: list(v) for k, v in split.items()} | {'val_airports': sorted(val_ap)}, open(f'{HERE}/split_bing.json', 'w'), indent=1)
todo = [im for v in split.values() for im in v]
def get(rel):
    out = os.path.join(HERE, rel)
    if os.path.exists(out) and os.path.getsize(out) > 0: return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for _ in range(3):
        try: urllib.request.urlretrieve(HF + rel, out + '.part'); os.replace(out + '.part', out); return 1
        except Exception: pass
    return -1
with ThreadPoolExecutor(2) as ex:
    res = list(ex.map(get, todo))
print({k: len(v) for k, v in split.items()}, 'downloaded', res.count(1), 'cached', res.count(0), 'failed', res.count(-1))
