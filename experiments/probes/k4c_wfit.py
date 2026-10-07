"""V55 refit weighted by each neuron's kappa4 lever on the mean closure, |sigma phi(alpha) (alpha^2 - 1)|, all neurons
(no decisive mask), features s3sq, y21, c2, chain, 1 (full layers >= 6). Reports weighted pattern corr and the lever-weighted
error of the rescaled pattern vs the chain's own (the quantity the mean closure sees). Prints the coefficients.
Usage: python experiments/probes/k4c_wfit.py   (env WPOW: weight power, default 2)"""
import numpy as np, os
from scipy.stats import norm
S = np.load("data/dumps/k4c_feats.npz")
NETS = {"mlp0": "data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "data/kaggle_out/kaggle_out_k4/k4teach"}
KEEP = [0, 1, 3, 5, 6]
P = float(os.environ.get("WPOW", "2"))
data = {n: [] for n in NETS}
for k in S.files:
    net, L, full = k.split("_"); l = int(L[1:])
    if full != "1" or l < 6: continue
    f = os.path.join(NETS[net], f"layer{l}_seed0.npz")
    if not os.path.exists(f): continue
    T = np.load(f); v = np.diag(T["C"]).astype(np.float64); mu = T["mean"].astype(np.float64); a = mu / np.sqrt(v)
    lev = np.abs(np.sqrt(v) * norm.pdf(a) * (a * a - 1))
    data[net].append((l, S[k][KEEP], T["K4d"].astype(np.float64) / v ** 2, lev, np.abs(a) < 1.5))
def fit(on, weighted):
    A, y, w = [], [], []
    for net in on:
        for l, X, t, lev, d in data[net]:
            m = np.ones_like(d) if weighted else d
            A.append(X[:, m].T); y.append(t[m]); w.append((lev[m] ** P) if weighted else np.ones(m.sum()))
    A = np.vstack(A); y = np.concatenate(y); sw = np.sqrt(np.concatenate(w))
    return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
def ev(b, net):
    e, e0 = [], []
    for l, X, t, lev, d in data[net]:
        ch = X[3]; p = b @ X; p = p * ch.mean() / p.mean(); tt = t * ch.mean() / t.mean()   # both at the chain level
        e.append(np.sum((lev * (p - tt)) ** 2)); e0.append(np.sum((lev * (ch - tt)) ** 2))
    return np.sum(e) / np.sum(e0)
for on in (["mlp0"], ["synth"], ["mlp0", "synth"]):
    for wtd in (False, True):
        b = fit(on, wtd)
        print(f"fit {'+'.join(on):10s} {'lever-wtd' if wtd else 'decisive ':9s}: lever-weighted err / chain  mlp0 {ev(b,'mlp0'):.3f}  synth {ev(b,'synth'):.3f}   beta {':'.join(f'{x:.6g}' for x in b)}")
