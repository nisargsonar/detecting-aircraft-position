"""Night-1 data expansion: Bing remainder + Bing full train, X-Plane train/test, Flight Simulator test.
Same airport split for every renderer (val airports = split_bing.json['val_airports']). Resumable; retries with backoff."""
import json, os, sys, time, random, urllib.request, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
HF = 'https://huggingface.co/datasets/DEEL-AI/LARD_V2/resolve/main/'
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)
sb = json.load(open(f'{HERE}/split_bing.json')); val_ap = set(sb['val_airports'])
csv = lambda s, sp: pd.read_csv(f'{HERE}/m_{s}_{sp}.csv', sep=';', usecols=['image', 'airport'])
def pick(d, n):
    ims = sorted(d.image.unique()); return list(rng.choice(ims, min(n, len(ims)), replace=False))
bt = csv('bingmaps', 'train'); xt = csv('xplane', 'train')
split = {
  'bing_test_rest': list(sb['test']),                                         # the 1,200 planned (483 missing)
  'bing_train_all': sorted(bt[~bt.airport.isin(val_ap)].image.unique()),
  'bing_val': list(sb['val']),
  'xplane_train':   pick(xt[~xt.airport.isin(val_ap)], 6000),
  'xplane_val':     pick(xt[xt.airport.isin(val_ap)], 800),
  'xplane_test':    pick(csv('xplane', 'test'), 1200),
  'flsim_test':     pick(csv('flsim', 'test'), 1200),
}
json.dump(split | {'val_airports': sorted(val_ap)}, open(f'{HERE}/split_multi.json', 'w'), indent=1)
def get(rel):
    out = os.path.join(HERE, rel)
    if os.path.exists(out) and os.path.getsize(out) > 0: return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for k in range(6):
        try:
            urllib.request.urlretrieve(HF + rel, out + '.part'); os.replace(out + '.part', out); return 1
        except Exception:
            time.sleep(min(60, 2 ** k + random.random()))
    return -1
for name, ims in split.items():                                              # in priority order
    with ThreadPoolExecutor(3) as ex: res = list(ex.map(get, ims))
    print(time.strftime('%H:%M'), name, len(ims), 'new', res.count(1), 'cached', res.count(0), 'failed', res.count(-1), flush=True)
