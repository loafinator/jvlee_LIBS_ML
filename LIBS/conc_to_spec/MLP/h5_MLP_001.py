from __future__ import annotations

import sys
import json
import time
import gc
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# Add project root setup matching directory layout
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import get_h5_ds


# =============================================================================
# Global Configuration & File Paths
# =============================================================================
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/max12_cts_noleak_crossval.h5"
SPLIT_DIR = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits_NoExtraDopants')
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/conc_to_spec/MLP/h5_plots")

NUM_FOLDS = 7
THRESH = 2.8
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
EXP_DOPANTS = {'Ce', 'Gd', 'Sm', 'U'}

# CTS MLP Hyperparameters
EPOCHS = 80
EARLY_STOP_PATIENCE = 15
BATCH_SIZE = 1024          # was 256; bigger batches keep the GPU busy
LEARNING_RATE = 2e-3       # scaled up a bit for the larger batch
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS = (512, 2048, 5120)
DROPOUT = 0.1
MAKE_PLOT = True

# Rows are L2-normalised, so mean(y^2) = 1/C. Multiplying targets by sqrt(C)
# gives unit mean-square targets (a predict-zero model then has loss == 1.0).
# All reported metrics are converted back to the original L2-normalised units.
SCALE = 100.0

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
if USE_AMP:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# =============================================================================
# Model
# =============================================================================
class PyTorchCTSMLP(nn.Module):
    """Deep MLP mapping elemental concentrations to LIBS spectra.

    NOTE: no ReLU on the output layer. The original final ReLU let the whole
    output die (all-zero predictions, zero gradient). Predictions are clamped
    at >= 0 at evaluation time instead.
    """
    def __init__(self, in_features=7, out_wavelengths=10000,
                 hidden_dims=(512, 2048, 5120), dropout=0.1):
        super().__init__()
        layers = []
        in_dim = in_features
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim),
                           nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h_dim
        self.out = nn.Linear(in_dim, out_wavelengths)
        layers.append(self.out)
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.out:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Data
# =============================================================================
def load_crossval_data(h5_path, spike_thresh=THRESH, exp_dopants=EXP_DOPANTS,
                       host_elements=HOST_ELEMENTS) -> Dict[str, Any]:
    """Same filtering as before, but with far fewer full-size copies of y."""
    keep_elements = host_elements.union(exp_dopants)

    with h5py.File(h5_path, 'r') as hf:
        y = np.asarray(get_h5_ds(hf, 'y')[:], dtype=np.float32)   # no copy if already f32
        X = np.asarray(get_h5_ds(hf, 'X')[:], dtype=np.float32)
        is_synth = get_h5_ds(hf, 'is_synthetic')[:]
        is_lockbox = get_h5_ds(hf, 'is_lockbox')[:].astype(bool) if 'is_lockbox' in hf else None
        group_id = get_h5_ds(hf, 'group_id')[:]
        wavelengths = get_h5_ds(hf, 'wavelengths')[:]
        names = [n.decode('utf-8') if isinstance(n, bytes) else str(n)
                 for n in get_h5_ds(hf, 'feature_cols')[:]]

    keep_idx = [i for i, n in enumerate(names) if n in keep_elements]
    extra_dopant_idx = [i for i, n in enumerate(names) if n not in keep_elements]

    clean_spike_mask = y.max(axis=1) <= spike_thresh
    has_extra = (X[:, extra_dopant_idx] > 0).any(axis=1)
    valid_synth_mask = (is_synth == 0) | (~has_extra)
    final_row_mask = clean_spike_mask & valid_synth_mask

    n_total = len(y)
    X_clean = X[final_row_mask][:, keep_idx]
    y_clean = y[final_row_mask]
    del y
    gc.collect()

    # L2 normalisation, in place (no extra 8 GB temporaries)
    y_t = torch.from_numpy(y_clean)
    y_t.div_(y_t.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-8))

    names_clean = [names[i] for i in keep_idx]
    n_dropped_spikes = n_total - int(clean_spike_mask.sum())
    n_dropped_synth = int((clean_spike_mask & ~valid_synth_mask).sum())
    print("Data loading summary:")
    print(f"  - Dropped {n_dropped_spikes} corrupted saturation shots (max > {spike_thresh}).")
    print(f"  - Filtered {n_dropped_synth} synthetic shots containing non-experimental dopants.")
    print(f"  - Retained {len(y_t)} total shots across {len(names_clean)} features: {names_clean}")

    out = dict(X=torch.from_numpy(X_clean), y=y_t,
               is_synth=torch.from_numpy(is_synth[final_row_mask]).long(),
               group_id=group_id[final_row_mask], wavelengths=wavelengths, names=names_clean)
    if is_lockbox is not None:
        out['is_lockbox'] = is_lockbox[final_row_mask]
    return out


def assign_group_folds(X, group_id, is_synthetic, dopant_idx, elem_names=None,
                       n_splits=5, n_trials=2000, min_groups_per_dopant=1, seed=42):
    """Stratified group fold assignment matching original script logic."""
    exp = np.where(is_synthetic == 0)[0]
    groups, inv = np.unique(group_id[exp], return_inverse=True)
    inv = inv.ravel()
    n_groups = len(groups)
    shots = np.bincount(inv, minlength=n_groups)

    first = np.zeros(n_groups, dtype=int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    present = X[exp[first]][:, dopant_idx] > 0
    total_g_per_dopant = present.sum(axis=0)
    active_dopant_mask = total_g_per_dopant > 0
    present = present[:, active_dopant_mask]
    active_dopant_idx = [d for i, d in enumerate(dopant_idx) if active_dopant_mask[i]]
    sig = (present.astype(int) * (1 << np.arange(len(active_dopant_idx)))).sum(axis=1)
    classes = np.unique(sig)

    best_fog = None
    best_score = None
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        fog = np.empty(n_groups, dtype=int)
        offset = int(rng.integers(n_splits))
        for s in rng.permutation(classes):
            members = rng.permutation(np.where(sig == s)[0])
            fog[members] = (offset + np.arange(len(members))) % n_splits
            offset += len(members)
        fold_shots = np.bincount(fog, weights=shots, minlength=n_splits)
        imbalance = np.abs(fold_shots / shots.sum() - 1.0 / n_splits).max()
        cov = np.array([present[fog == k].sum(axis=0) for k in range(n_splits)])
        score = (-min(int(cov.min()), min_groups_per_dopant), imbalance)
        if best_score is None or score < best_score:
            best_fog, best_score = fog, score

    if best_fog is None:
        raise ValueError("Failed to assign fold groups: n_trials must be > 0.")

    fold_id = np.full(len(X), -1, dtype=np.int32)
    fold_id[exp] = best_fog[inv]
    return fold_id


def build_splits_once(data, split_path, num_folds, seed=42):
    split_path = Path(split_path).with_suffix('.npz')
    if split_path.exists():
        print(f"Loading cached splits from: {split_path}")
        d = np.load(split_path)
        return d['test_idx'], d['cv_idx'], d['cv_fold_id']

    split_path.parent.mkdir(parents=True, exist_ok=True)
    test_idx = np.where(data['is_lockbox'])[0] if 'is_lockbox' in data else np.array([], dtype=int)
    cv_idx = np.where(~data['is_lockbox'])[0] if 'is_lockbox' in data else np.arange(len(data['X']), dtype=int)

    X_cv = data['X'][cv_idx].numpy()
    is_synth_cv = data['is_synth'][cv_idx]
    dopant_idx = [i for i, n in enumerate(data['names'])
                  if n not in HOST_ELEMENTS and (X_cv[:, i] > 0).any()]
    cv_fold_id = assign_group_folds(
        X=X_cv, group_id=data['group_id'][cv_idx], is_synthetic=is_synth_cv,
        dopant_idx=dopant_idx, elem_names=data['names'], n_splits=num_folds, seed=seed)
    np.savez(split_path, test_idx=test_idx, cv_idx=cv_idx, cv_fold_id=cv_fold_id,
             num_folds=num_folds, seed=seed)
    return test_idx, cv_idx, cv_fold_id


def to_gpu_fp16_scaled(Y_cpu: torch.Tensor, idx: np.ndarray, chunk: int = 16384) -> torch.Tensor:
    """Copy selected rows to the GPU as scaled fp16 in chunks (no big CPU copies)."""
    idx_t = torch.from_numpy(np.asarray(idx)).long()
    out = torch.empty((len(idx_t), Y_cpu.shape[1]), dtype=torch.float16, device=DEVICE)
    for s in range(0, len(idx_t), chunk):
        sel = idx_t[s:s + chunk]
        out[s:s + chunk] = (Y_cpu[sel] * SCALE).to(DEVICE).half()
    return out


# =============================================================================
# Streaming metrics (R2 / MAE / RMSE) -- accumulated on GPU, nothing big in RAM
# =============================================================================
class MetricAcc:
    def __init__(self, C: int):
        self.n = 0
        self.sy = torch.zeros(C, dtype=torch.float64, device=DEVICE)
        self.syy = torch.zeros(C, dtype=torch.float64, device=DEVICE)
        self.see = torch.zeros(C, dtype=torch.float64, device=DEVICE)
        self.sae = torch.zeros((), dtype=torch.float64, device=DEVICE)

    def update(self, y: torch.Tensor, p: torch.Tensor):
        """y, p in ORIGINAL (L2-normalised) units, float32, shape (n, C)."""
        d = (p - y).double()
        yd = y.double()
        self.n += y.shape[0]
        self.sy += yd.sum(0)
        self.syy += (yd * yd).sum(0)
        self.see += (d * d).sum(0)
        self.sae += d.abs().sum()

    def merge(self, other: "MetricAcc"):
        self.n += other.n
        self.sy += other.sy; self.syy += other.syy
        self.see += other.see; self.sae += other.sae

    def result(self) -> Dict[str, float]:
        C = self.sy.numel()
        total = self.n * C
        sst_c = self.syy - self.sy ** 2 / self.n
        valid = sst_c > 1e-6 * sst_c.mean()
        r2_c = 1.0 - self.see / sst_c.clamp(min=1e-30)
        mse = (self.see.sum() / total).item()
        return dict(
            n_spectra=int(self.n),
            mse=mse,
            rmse=mse ** 0.5,
            mae=(self.sae / total).item(),
            # 1 - total SSE / total per-channel variance (sklearn 'variance_weighted')
            r2_variance_weighted=(1.0 - self.see.sum() / sst_c.sum()).item(),
            # plain average of per-channel R2 (sklearn 'uniform_average'), low-variance channels excluded
            r2_uniform_channels=r2_c[valid].mean().item(),
            # R2 against a single global mean (peaks dominate -> optimistic)
            r2_global=(1.0 - self.see.sum() / (self.syy.sum() - self.sy.sum() ** 2 / total)).item(),
            baseline_mse_predict_zero=(self.syy.sum() / total).item(),
            baseline_mse_predict_channel_mean=(sst_c.sum() / total).item(),
        )


def print_metrics(title: str, m: Dict[str, float]):
    print(f" {title:<18}| R2(var-wtd): {m['r2_variance_weighted']:.4f} | R2(chan-avg): {m['r2_uniform_channels']:.4f} "
          f"| R2(global): {m['r2_global']:.4f} | MAE: {m['mae']:.6e} | RMSE: {m['rmse']:.6e} | MSE: {m['mse']:.6e}")


# =============================================================================
# GPU-resident helpers
# =============================================================================
def forward_eval(model, bx):
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=USE_AMP):
        p = model(bx)
    return p.float().clamp_min_(0.0)        # intensities are non-negative


@torch.no_grad()
def val_loss_scaled(model, X, Y, idx, bs=4096) -> float:
    sse = torch.zeros((), dtype=torch.float64, device=DEVICE)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        p = forward_eval(model, X[b])
        sse += ((p - Y[b].float()) ** 2).sum().double()
    return (sse / (len(idx) * Y.shape[1])).item()   # 1.0 == predict-zero


@torch.no_grad()
def eval_into(model, X, Y, idx, acc: MetricAcc, lock_sum=None, bs=2048):
    """Update `acc` with predictions for rows `idx`; optionally add preds to lock_sum."""
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        p = forward_eval(model, X[b])
        if lock_sum is not None:
            lock_sum[s:s + len(b)] += p
        acc.update(Y[b].float() / SCALE, p / SCALE)


def plot_spectral_overlay_sample(wavelengths, y_true, y_pred, sample_idx=0, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(wavelengths, y_true, label="Measured / Target", color="black", alpha=0.8, lw=1)
    ax1.plot(wavelengths, y_pred, label="MLP Predicted", color="crimson", alpha=0.8, lw=1)
    ax1.set_ylabel("Normalized Intensity (a.u.)")
    ax1.set_title(f"CTS MLP Forward Prediction - Sample #{sample_idx}", fontweight="bold")
    ax1.legend(loc="upper right"); ax1.grid(True, linestyle=":", alpha=0.5)
    ax2.plot(wavelengths, y_true - y_pred, label="Residual (True - Pred)", color="darkblue", lw=0.8)
    ax2.axhline(0, color="gray", linestyle="--", lw=1)
    ax2.set_xlabel("Wavelength (nm)"); ax2.set_ylabel("Error (a.u.)")
    ax2.legend(loc="upper right"); ax2.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Spectral Overlay Plot to: {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    global SCALE
    print(f"=== Starting GPU-Accelerated CTS MLP Forward Pipeline [{DEVICE}] (bf16 autocast: {USE_AMP}) ===")
    t0_start = time.perf_counter()

    data = load_crossval_data(H5_FILE)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS, seed=SEED)

    C = data['y'].shape[1]
    SCALE = float(C) ** 0.5
    input_dim = data['X'].shape[1]
    wavelengths_grid = data['wavelengths']

    # Everything lives on the GPU from here on (fp16 targets: ~4 GB for the CV pool)
    X_cv = data['X'][torch.from_numpy(cv_idx).long()].to(DEVICE)
    X_test = data['X'][torch.from_numpy(test_idx).long()].to(DEVICE)
    Y_cv = to_gpu_fp16_scaled(data['y'], cv_idx)
    Y_test = to_gpu_fp16_scaled(data['y'], test_idx)
    del data
    gc.collect()

    fold_t = torch.from_numpy(cv_fold_id).to(DEVICE)
    print(f"Dataset Loaded | CV Pool: {len(cv_idx)} samples | Lockbox Test Set: {len(test_idx)} samples")
    print(f"Inputs: {input_dim} concentration features | Target Output: {C} spectral channels")
    print(f"(Validation uses experimental shots only: fold_id >= 0; synthetic shots are train-only)")

    oof_acc = MetricAcc(C)
    lock_sum = torch.zeros((len(test_idx), C), device=DEVICE)
    test_idx_gpu = torch.arange(len(test_idx), device=DEVICE)
    fold_results = []

    print("\n--- Training Deep CTS MLP across K Folds ---")
    for k in range(NUM_FOLDS):
        t_fold = time.perf_counter()
        torch.manual_seed(SEED + k)
        train_idx = (fold_t != k).nonzero().squeeze(1)
        val_idx = (fold_t == k).nonzero().squeeze(1)

        model = PyTorchCTSMLP(input_dim, C, HIDDEN_DIMS, DROPOUT).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY, fused=USE_AMP)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val, best_state, bad_epochs = float('inf'), None, 0
        n_tr = len(train_idx)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            perm = train_idx[torch.randperm(n_tr, device=DEVICE)]
            tr_loss = torch.zeros((), device=DEVICE)
            for s in range(0, n_tr, BATCH_SIZE):
                b = perm[s:s + BATCH_SIZE]
                if len(b) < 2:          # BatchNorm needs >1 sample
                    continue
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=USE_AMP):
                    pred = model(X_cv[b])
                loss = F.mse_loss(pred.float(), Y_cv[b].float())
                loss.backward()
                optimizer.step()
                tr_loss += loss.detach() * len(b)      # no per-step .item() sync

            model.eval()
            v = val_loss_scaled(model, X_cv, Y_cv, val_idx)
            scheduler.step(v)

            if v < best_val:
                best_val, bad_epochs = v, 0
                best_state = {n: t.detach().clone() for n, t in model.state_dict().items()}  # stays on GPU
            else:
                bad_epochs += 1

            if epoch <= 3 or epoch % 5 == 0:
                print(f"  fold {k+1} ep {epoch:02d} | train {tr_loss.item()/n_tr:.4f} | "
                      f"val {v:.4f} (1.0 = predict-zero) | best {best_val:.4f}")
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"  fold {k+1}: early stop at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            raise ValueError("Best_state freaking sharted itself, try again boyo!")
        model.eval()

        fold_acc = MetricAcc(C)
        eval_into(model, X_cv, Y_cv, val_idx, fold_acc)
        eval_into(model, X_test, Y_test, test_idx_gpu, MetricAcc(C), lock_sum=lock_sum)  # accumulates ensemble
        fm = fold_acc.result()
        oof_acc.merge(fold_acc)
        fold_results.append(fm)

        dt = time.perf_counter() - t_fold
        print(f"Fold {k+1}/{NUM_FOLDS} Complete | Best Val Spectral MSE: {fm['mse']:.6e} | Time: {dt:.1f}s")
        print_metrics(f"Fold {k+1} (val)", fm)

        # Save progress after every fold so a kill can't wipe everything out
        with open(OUTPUT_DIR / f"cts_mlp_fold_metrics_{TIMESTAMP}.json", "w") as f:
            json.dump(fold_results, f, indent=2)

        del model, optimizer, scheduler, best_state
        torch.cuda.empty_cache()

    # Lockbox: average the 7 fold models' predictions
    lock_pred = lock_sum / NUM_FOLDS
    lock_acc = MetricAcc(C)
    for s in range(0, len(test_idx), 2048):
        sl = slice(s, s + 2048)
        lock_acc.update(Y_test[sl].float() / SCALE, lock_pred[sl] / SCALE)

    oof_m, lock_m = oof_acc.result(), lock_acc.result()

    print("\n" + "=" * 100)
    print(" Concentration-to-Spectra (CTS) MLP Performance Summary (original L2-normalised units)")
    print("=" * 100)
    print_metrics("OOF (CV)", oof_m)
    print_metrics("Lockbox (ens.)", lock_m)
    print(f" OOF baselines      | MSE predict-zero: {oof_m['baseline_mse_predict_zero']:.6e} "
          f"| MSE predict-channel-mean: {oof_m['baseline_mse_predict_channel_mean']:.6e}")
    print("=" * 100)

    with open(OUTPUT_DIR / f"cts_mlp_metrics_{TIMESTAMP}.json", "w") as f:
        json.dump(dict(oof=oof_m, lockbox=lock_m, folds=fold_results), f, indent=2)

    if MAKE_PLOT:
        i = int(np.random.randint(0, len(test_idx)))
        plot_spectral_overlay_sample(
            wavelengths_grid, (Y_test[i].float() / SCALE).cpu().numpy(),
            (lock_pred[i] / SCALE).cpu().numpy(), i,
            OUTPUT_DIR / f"cts_mlp_sample_{i}_reconstruction_{TIMESTAMP}.png")

    print(f"CTS Pipeline completed in {time.perf_counter() - t0_start:.1f} seconds.")


if __name__ == "__main__":
    main()