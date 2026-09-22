"""Residual-structure study (V33): is the final-layer residual truth - V31 prediction predictable from cheap per-neuron features?
Inputs: kaggle_out_dumps/*/dumps/{<name>_<seed>.npz (feats0..15: n x 19, pred), truth_<name>_<seed>.npz (truth 16 x n, pred)}
Features per neuron (final layer li = 15 and the previous layer): [1, mu, var, sigma, alpha, |alpha|, phi, Phi, pk1v, D3, |D21| row sum,
K4_sigma, sigma*phi, dG, pk2v, pk3v, pk4v, g4row, delta].
Models: ridge on layer-15 features; ridge on layers 15+14 features; ridge with quadratic expansion; gradient boosting (sklearn) if available.
Validation: leave-MLPs-out (5-fold over MLPs). Reports the residual MSE before/after as a fraction (1 = no gain).
Usage: python residual_fit.py
"""
import glob, os, numpy as np
files = sorted(glob.glob("data/kaggle_out/kaggle_out_dumps/*/dumps/truth_*.npz"))
X15, X14, Y, G = [], [], [], []
for i, tf in enumerate(files):
    key = os.path.basename(tf)[len("truth_"):]
    ff = os.path.join(os.path.dirname(tf), key)
    if not os.path.exists(ff): continue
    t = np.load(tf); f = np.load(ff)
    truth = t["truth"][-1]; pred = t["pred"][-1]
    X15.append(f["feats15"]); X14.append(f["feats14"]); Y.append(truth - pred); G.append(np.full(len(truth), i))
X15 = np.concatenate(X15); X14 = np.concatenate(X14); Y = np.concatenate(Y); G = np.concatenate(G)
n_mlp = len(set(G)); print(f"{n_mlp} MLPs, {len(Y)} neurons; residual MSE {np.mean(Y**2):.4e} (this is the raw final-layer MSE)")
names = ["1","mu","var","sigma","alpha","|alpha|","phi","Phi","pk1v","D3","|D21|row","K4sig","sig*phi","dG","pk2v","pk3v","pk4v","g4row","delta"]
def standardize(X):
    m = X.mean(0); s = X.std(0) + 1e-12; return (X - m) / s
def cv_ridge(X, y, groups, lam=1e-3, folds=5):
    ids = np.array(sorted(set(groups))); rng = np.random.default_rng(0); rng.shuffle(ids)
    sse = 0.0
    for k in range(folds):
        test = np.isin(groups, ids[k::folds]); tr = ~test
        Xt = X[tr]; A = Xt.T @ Xt + lam * len(Xt) * np.eye(X.shape[1]); b = Xt.T @ y[tr]
        w = np.linalg.solve(A, b); sse += np.sum((y[test] - X[test] @ w) ** 2)
    return sse / len(y)
base = np.mean(Y ** 2)
def report(label, X, lam=1e-3):
    X = np.hstack([np.ones((len(X), 1)), standardize(X)])
    m = cv_ridge(X, Y, G, lam); print(f"{label:55s} CV residual MSE {m:.4e}  = {m/base:.4f} of baseline", flush=True)
report("ridge: layer-15 features (19)", X15)
report("ridge: layer-15 + layer-14 features (38)", np.hstack([X15, X14]))
# quadratic expansion of the layer-15 features (excluding the constant)
Z = X15[:, 1:]; Zs = standardize(Z)
quad = np.hstack([Zs] + [Zs[:, i:i+1] * Zs[:, i:] for i in range(Zs.shape[1])])
report(f"ridge: layer-15 quadratic ({quad.shape[1]})", quad, lam=1e-2)
# single-feature correlations
print("single-feature |corr| with the residual:", " ".join(f"{nm}:{abs(np.corrcoef(X15[:, j], Y)[0,1]):.3f}" for j, nm in enumerate(names) if X15[:, j].std() > 0))
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    ids = np.array(sorted(set(G))); rng = np.random.default_rng(0); rng.shuffle(ids); sse = 0.0
    XX = np.hstack([X15, X14])
    for k in range(5):
        test = np.isin(G, ids[k::5]); tr = ~test
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0)
        mdl.fit(XX[tr], Y[tr]); sse += np.sum((Y[test] - mdl.predict(XX[test])) ** 2)
    print(f"{'HistGBM: layers 15+14 features':55s} CV residual MSE {sse/len(Y):.4e}  = {sse/len(Y)/base:.4f} of baseline")
except ImportError:
    print("sklearn not available: GBM skipped")
# is the residual per-MLP structured at all? fraction of residual variance explained by a per-MLP constant offset
off = np.array([Y[G == g].mean() for g in sorted(set(G))]); print(f"per-MLP mean residual: rms {np.sqrt(np.mean(off**2)):.3e} (vs residual rms {np.sqrt(base):.3e}) -> offsets explain {np.mean(off**2)/base:.3%}")
