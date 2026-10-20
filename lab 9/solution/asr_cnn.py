"""
Lab 9 - CNN-based speech recognition from mel-spectral images
=============================================================

Task: "Implement CNN based speech recognition using melspectral images."

Pipeline
    1. dataset    : spoken digits, synthesised by `speech_commands.py` (or loaded from a
                    real folder of WAVs with `--data-dir`) with speaker-disjoint splits
    2. front-end  : log-mel spectrogram images (64 mels x 64 frames, 0.8 s clips),
                    computed from scratch (pre-emphasis -> framing -> Hann -> |FFT|^2 ->
                    mel filterbank -> log -> per-image standardisation)
    3. models     : a small VGG-style CNN (~190 k parameters) and an MLP baseline over the
                    flattened image, both trained with the same optimiser/schedule
    4. evaluation : test accuracy, per-class precision/recall/F1, confusion matrix,
                    accuracy-vs-SNR robustness sweep, and misclassified examples

Run
    python asr_cnn.py                 # full run, writes everything to ./results
    python asr_cnn.py --quick         # tiny run for a fast check
    python asr_cnn.py --speakers 40 --epochs 12
    python asr_cnn.py --predict some.wav
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from speech_commands import SR, DIGITS, add_noise, generate_dataset, load_audio_dir

# --------------------------------------------------------------------------- front-end
CLIP_S = 0.8                                  # longer than the longest digit (0.68 s at rate 1.25)
CLIP_LEN = int(SR * CLIP_S)                   # 12800 samples
N_FFT = 512                                   # 32 ms window
HOP = 195                                     # -> exactly 64 frames per clip
N_MELS = 64
FMIN, FMAX = 50.0, 7600.0
PREEMPH = 0.97


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=FMIN, fmax=FMAX):
    """Triangular mel filterbank, HTK style: (n_mels, n_fft//2+1), rows sum to 1."""
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
        fb[i] /= fb[i].sum() + 1e-12
    return fb


_MEL_FB = None


def log_mel(wave, fb=None):
    """Waveform (any length) -> log-mel image, shape (N_MELS, 64), float32."""
    global _MEL_FB
    if fb is None:
        if _MEL_FB is None:
            _MEL_FB = mel_filterbank()
        fb = _MEL_FB
    wave = np.asarray(wave)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if np.issubdtype(wave.dtype, np.integer):
        wave = wave / np.iinfo(wave.dtype).max
    wave = wave.astype(np.float64)
    if len(wave) >= CLIP_LEN:
        start = (len(wave) - CLIP_LEN) // 2
        wave = wave[start:start + CLIP_LEN]
    else:
        pad = (CLIP_LEN - len(wave)) // 2
        wave = np.pad(wave, (pad, CLIP_LEN - len(wave) - pad))
    wave = np.append(wave[0], wave[1:] - PREEMPH * wave[:-1])          # pre-emphasis
    frames = np.lib.stride_tricks.sliding_window_view(wave, N_FFT)[::HOP]
    frames = frames * np.hanning(N_FFT)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = spec @ fb.T                                                  # (frames, mels)
    mel = np.log(mel + 1e-6)
    img = mel.T.astype(np.float32)                                     # (mels, frames)
    img = (img - img.mean()) / (img.std() + 1e-6)                      # per-image standardisation
    return img


def mel_images(items, rng=None, snr_db=None):
    """List of dataset items -> (X float32 [n,1,64,64], y int64 [n])."""
    X = np.zeros((len(items), 1, N_MELS, 64), dtype=np.float32)
    for i, it in enumerate(items):
        w = it["wave"]
        if snr_db is not None and rng is not None:
            w = add_noise(w, snr_db, rng)
        X[i, 0] = log_mel(w)
    y = np.array([DIGITS.index(it["label"]) for it in items], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

# --------------------------------------------------------------------------- models
class MelCNN(nn.Module):
    """Small VGG-style CNN: 4 conv blocks (2x conv 3x3 + BN + ReLU + max-pool) + classifier."""

    def __init__(self, n_classes=10):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(1, 8), block(8, 16), block(16, 32), block(32, 48))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(48 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class MelMLP(nn.Module):
    """Baseline: two hidden layers over the flattened 64x64 image."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(N_MELS * 64, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())

# --------------------------------------------------------------------------- training
def train_model(model, Xtr, ytr, Xdv, ydv, epochs=20, batch=64, lr=1e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "dev_loss": [], "dev_acc": []}
    n = len(ytr)
    g = torch.Generator().manual_seed(seed)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=g)
        tot_loss = correct = 0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == ytr[idx]).sum().item()
        sched.step()

        model.eval()
        with torch.no_grad():
            dv_out = model(Xdv)
            dv_loss = lossf(dv_out, ydv).item()
            dv_acc = (dv_out.argmax(1) == ydv).mean().item()
        hist["train_loss"].append(tot_loss / n)
        hist["train_acc"].append(correct / n)
        hist["dev_loss"].append(dv_loss)
        hist["dev_acc"].append(dv_acc)
        if verbose:
            print(f"    epoch {ep:2d}/{epochs}  train loss {tot_loss / n:.3f} acc {correct / n:.3f}"
                  f"  |  dev loss {dv_loss:.3f} acc {dv_acc:.3f}  ({time.time() - t0:.1f}s)")
    return hist


@torch.no_grad()
def predict_probs(model, X, batch=128):
    model.eval()
    return torch.cat([nn.functional.softmax(model(X[k:k + batch]), dim=1)
                      for k in range(0, len(X), batch)])


def classification_report(y_true, y_pred, labels=DIGITS):
    """Per-class precision / recall / F1 / support + macro averages."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows, f1s = [], []
    for k, name in enumerate(labels):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": name, "precision": prec, "recall": rec,
                     "f1": f1, "support": int((y_true == k).sum())})
        f1s.append(f1)
    rows.append({"class": "macro avg",
                 "precision": float(np.mean([r["precision"] for r in rows])),
                 "recall": float(np.mean([r["recall"] for r in rows])),
                 "f1": float(np.mean(f1s)),
                 "support": int(len(y_true))})
    return rows


def confusion(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[t, p] += 1
    return cm

# --------------------------------------------------------------------------- main flows
def build_data(speakers=80, seed=0, data_dir=None):
    if data_dir:
        data, info = load_audio_dir(data_dir)
    else:
        data, info = generate_dataset(n_speakers=speakers, seed=seed)
    packed = {}
    for split in ("train", "dev", "test"):
        packed[split] = mel_images(data[split])
    return packed, data, info


def robustness_sweep(model, test_items, snrs=(None, 20, 10, 5, 0, -5), seed=0):
    """Accuracy on the test speakers after adding white noise at each SNR (dB)."""
    rng = np.random.default_rng(seed)
    results = []
    for snr in snrs:
        X, y = mel_images(test_items, rng=rng, snr_db=snr)
        acc = (predict_probs(model, X).argmax(1) == y).float().mean().item()
        results.append({"snr_db": snr, "accuracy": acc})
        print(f"    SNR {'clean' if snr is None else f'{snr:>3} dB'}: acc {acc:.3f}")
    return results


def save_plots(out_dir, hist, cm, mel_examples, snr_results, X_test, y_test, preds, test_items):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # mel-spectral images of the ten digits
    fig, axes = plt.subplots(2, 5, figsize=(13, 5))
    for ax, k in zip(axes.ravel(), range(10)):
        ax.imshow(mel_examples[k], origin="lower", aspect="auto", cmap="magma")
        ax.set_title(DIGITS[k])
        ax.set_xlabel("frame"); ax.set_ylabel("mel")
    fig.suptitle("Log-mel spectrogram images (0.8 s clips, 64 mels x 64 frames)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "mel_images.png"), dpi=110); plt.close(fig)

    # training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    a1.plot(ep, hist["train_loss"], label="train"); a1.plot(ep, hist["dev_loss"], label="dev")
    a2.plot(ep, hist["train_acc"], label="train"); a2.plot(ep, hist["dev_acc"], label="dev")
    a1.set_title("loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.set_title("accuracy"); a2.set_xlabel("epoch"); a2.set_ylim(0, 1.02); a2.legend()
    fig.suptitle("CNN training")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=110); plt.close(fig)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(10), DIGITS, rotation=45); ax.set_yticks(range(10), DIGITS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix - test speakers")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=110); plt.close(fig)

    # accuracy vs SNR
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [("clean" if r["snr_db"] is None else r["snr_db"]) for r in snr_results]
    ax.plot(range(len(xs)), [r["accuracy"] for r in snr_results], "o-")
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_xlabel("SNR (dB), 'clean' = no added noise"); ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.set_title("Robustness to additive noise")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "accuracy_vs_snr.png"), dpi=110); plt.close(fig)

    # misclassified examples at 0 dB
    wrong = np.where(np.asarray(preds) != y_test.numpy())[0][:6]
    if len(wrong):
        rng = np.random.default_rng(0)
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        for ax, k in zip(axes.ravel(), wrong):
            w = add_noise(test_items[k]["wave"], 0.0, rng)
            ax.imshow(log_mel(w), origin="lower", aspect="auto", cmap="magma")
            ax.set_title(f"true {test_items[k]['label']} / pred {DIGITS[preds[k]]}")
            ax.axis("off")
        for ax in axes.ravel()[len(wrong):]:
            ax.axis("off")
        fig.suptitle("Misclassified test clips with 0 dB noise")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "misclassified.png"), dpi=110); plt.close(fig)
    else:
        print("    (no misclassified test clips - nothing to plot)")


def _fmt(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Lab 9 - CNN speech recognition on mel-spectral images")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--speakers", type=int, default=80, help="synthetic dataset speakers")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="tiny run (few speakers / epochs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", help="optional real dataset folder (sub-folder per word)")
    ap.add_argument("--out", default=os.path.join(here, "results"))
    ap.add_argument("--predict", metavar="WAV", help="classify one WAV file and exit")
    ap.add_argument("--model", default=None, help="checkpoint for --predict (default: <out>/model.pt)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = args.out
    os.makedirs(out, exist_ok=True)
    model_path = args.model or os.path.join(out, "model.pt")

    if args.predict:
        ckpt = torch.load(model_path, map_location="cpu")
        net = MelCNN(); net.load_state_dict(ckpt["state_dict"]); net.eval()
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        fs, w = wavfile.read(args.predict)
        w = np.asarray(w)
        if w.ndim > 1:
            w = w.mean(axis=1)
        if np.issubdtype(w.dtype, np.integer):
            w = w / np.iinfo(w.dtype).max
        w = w.astype(np.float64)
        if fs != SR:
            w = resample_poly(w, SR, fs)
        p = predict_probs(net, torch.from_numpy(log_mel(w)[None, None]))[0].numpy()
        order = np.argsort(p)[::-1]
        print(f"{args.predict}:")
        for k in order[:3]:
            print(f"  {DIGITS[k]:>6}  {p[k] * 100:5.1f}%")
        return

    speakers = 24 if args.quick else args.speakers
    epochs = 6 if args.quick else args.epochs
    print(f"Lab 9 - CNN speech recognition | {speakers} speakers, {epochs} epochs" +
          ("  (quick mode)" if args.quick else ""))

    t0 = time.time()
    print("[1/5] building dataset and mel-spectral images ...")
    packed, data, info = build_data(speakers=speakers, seed=args.seed, data_dir=args.data_dir)
    Xtr, ytr = packed["train"]; Xdv, ydv = packed["dev"]; Xte, yte = packed["test"]
    print(f"    speakers {info['speakers']} | train {len(ytr)} / dev {len(ydv)} / test {len(yte)} images"
          f"  ({time.time() - t0:.1f}s)")

    print(f"[2/5] training MelCNN ({n_params(MelCNN())} parameters) ...")
    cnn = MelCNN()
    hist_cnn = train_model(cnn, Xtr, ytr, Xdv, ydv, epochs=epochs, seed=args.seed)
    torch.save({"state_dict": cnn.state_dict(),
                "words": DIGITS,
                "frontend": {"sr": SR, "clip_s": CLIP_S, "n_fft": N_FFT, "hop": HOP,
                             "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX, "preemph": PREEMPH}},
               model_path)

    print(f"[3/5] training MLP baseline ({n_params(MelMLP())} parameters) ...")
    mlp = MelMLP()
    hist_mlp = train_model(mlp, Xtr, ytr, Xdv, ydv, epochs=epochs, seed=args.seed, verbose=False)

    cnn.eval(); mlp.eval()
    with torch.no_grad():
        cnn_acc = (cnn(Xte).argmax(1) == yte).mean().item()
        mlp_acc = (mlp(Xte).argmax(1) == yte).mean().item()
    print(f"    test accuracy - CNN {cnn_acc:.3f} | MLP {mlp_acc:.3f}")
    print(f"    best dev accuracy - CNN {max(hist_cnn['dev_acc']):.3f} | MLP {max(hist_mlp['dev_acc']):.3f}")

    print("[4/5] test evaluation (report, confusion matrix, error examples) ...")
    probs = predict_probs(cnn, Xte).numpy()
    preds = probs.argmax(1)
    report = classification_report(yte.numpy(), preds)
    cm = confusion(yte.numpy(), preds)
    for r in report:
        print(f"    {r['class']:>9}  P {_fmt(r['precision'])}  R {_fmt(r['recall'])}"
              f"  F1 {_fmt(r['f1'])}  n {r['support']}")

    mel_examples = {}
    for k, w in enumerate(DIGITS):
        mel_examples[k] = Xte[[i for i, it in enumerate(data["test"]) if it["label"] == w][0], 0].numpy()

    print("[5/5] robustness sweep (additive white noise) ...")
    snr_results = robustness_sweep(cnn, data["test"], seed=args.seed + 1)

    save_plots(out, hist_cnn, cm, mel_examples, snr_results, Xte, yte, preds, data["test"])
    metrics = {"accuracy": {"cnn": cnn_acc, "mlp": mlp_acc},
               "best_dev_accuracy": {"cnn": max(hist_cnn["dev_acc"]), "mlp": max(hist_mlp["dev_acc"])},
               "parameters": {"cnn": n_params(MelCNN()), "mlp": n_params(MelMLP())},
               "epochs": epochs, "speakers": speakers,
               "frontend": {"clip_s": CLIP_S, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS},
               "classification_report": report,
               "confusion_matrix": cm.tolist(),
               "robustness": snr_results,
               "history": hist_cnn}
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"done in {time.time() - t0:.1f}s - results in {out}")


if __name__ == "__main__":
    main()
