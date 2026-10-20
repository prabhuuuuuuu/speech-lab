"""
Lab 8 - Extraction and Comparison of MFCC, PLP, LPCC and GFCC Features from Speech Signals
==========================================================================================

Load and preprocess a speech signal, frame and window it, and extract four classic
feature sets for speech recognition, all implemented from scratch with NumPy / SciPy:

    1. MFCC : pre-emphasis -> framing -> Hamming -> FFT power spectrum ->
              Mel filterbank -> log -> DCT-II
    2. PLP  : power spectrum -> Bark critical-band integration -> equal-loudness
              pre-emphasis -> cube-root loudness compression -> IDFT ->
              all-pole model (LPC order 5, Levinson-Durbin) -> cepstral recursion
              (Hermansky, 1990)
    3. LPCC : autocorrelation LPC (order 12, Levinson-Durbin) -> cepstral recursion
    4. GFCC : power spectrum -> gammatone filterbank (ERB scale, 4th order) ->
              cube-root cochlear compression -> DCT-II

The script then visualises the four representations, compares them and measures how
much each feature set changes when white noise is added to the clean signal (10 dB
SNR) - a simple proxy for robustness in a speech recogniser.

Run
    python feature_extraction.py                                   # lab 1 harvard.wav if found, else synthetic
    python feature_extraction.py --wav my.wav --start 0.5 --duration 3.0
    python feature_extraction.py --synthetic                       # no audio file needed

Outputs (in results/): feature_comparison.png, noise_robustness.png, feature_stats.csv,
robustness.csv
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

SR = 16000                 # working sample rate
FRAME_MS, HOP_MS = 25.0, 10.0
N_FFT = 512
N_MELS = 26                # triangular Mel filters
N_MFCC = 13                # kept cepstral coefficients (c0..c12)
N_BARK = 21                # Bark critical bands for PLP
PLP_ORDER = 5              # all-pole order for PLP (Hermansky)
N_CEP = 13                 # cepstral coefficients for PLP / LPCC / GFCC
LPC_ORDER = 12             # all-pole order for LPCC
N_GAMMATONE = 32           # gammatone filters for GFCC
PREEMPH = 0.97


def banner(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


# =============================================================================
# Signal loading and preprocessing
# =============================================================================
def synthesize_utterance(fs=SR, duration=2.4, seed=0):
    """Speech-like fallback signal: /a/-/i/-/u/ vowel sequence with formant glides,
    a fricative /s/ burst and a pitch contour. Used only when no WAV file exists."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t) + 3 * rng.standard_normal(n).cumsum() * 0.02
    f0 = np.clip(f0, 80, 220)

    phase = 2 * np.pi * np.cumsum(f0) / fs
    pulses = np.zeros(n)
    wrap = np.diff(np.floor(phase / (2 * np.pi))) > 0
    pulses[1:][wrap] = 1.0

    targets = [(1.0, (700, 1200, 2500)), (1.8, (300, 2300, 3000)), (2.4, (350, 800, 2400))]
    formants = np.zeros((n, 3))
    for (t0, f), (t1, g) in zip([(0.0, targets[0][1])] + targets[:-1], targets):
        m = (t >= t0) & (t < t1)
        w = ((t[m] - t0) / max(t1 - t0, 1e-9))[:, None]
        formants[m] = (1 - w) * np.array(f) + w * np.array(g)

    sig = np.zeros(n)
    state = [None, None, None]
    step = int(0.005 * fs)
    for k in range(0, n - 1, step):
        sl = slice(k, min(k + step, n))
        exc = pulses[sl] - 0.5 * np.roll(pulses, 1)[sl]
        y = exc
        for depth, (f, bw) in enumerate(zip(formants[k], [90, 110, 170])):
            r = np.exp(-np.pi * bw / fs)
            b = [1 - r]
            a = [1, -2 * r * np.cos(2 * np.pi * f / fs), r * r]
            y, state[depth] = _lfilter_chunk(b, a, y, state[depth])
        sig[sl] = y

    fric = slice(int(0.62 * fs), int(0.78 * fs))
    sig[fric] += 0.15 * rng.standard_normal(fric.stop - fric.start)
    sig = pre_emphasis(sig, PREEMPH)
    return normalize(sig)


def _lfilter_chunk(b, a, x, zi):
    from scipy.signal import lfilter, lfilter_zi
    if zi is None:
        zi = lfilter_zi(b, a) * (x[0] if len(x) else 0.0)
    y, zo = lfilter(b, a, x, zi=zi)
    return y, zo


def load_speech(path=None, target_sr=SR, start=1.0, duration=3.0):
    """Load a WAV file (mono, resampled, trimmed, normalised).

    Returns (signal, sample_rate, source_description). Falls back to the synthetic
    utterance when no path is given or the file cannot be read."""
    if path and os.path.exists(path):
        sr, data = wavfile.read(path)
        data = np.asarray(data)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            data = data / np.iinfo(data.dtype).max
        data = data.astype(np.float64)
        if sr != target_sr:
            data = resample_poly(data, target_sr, sr)
            sr = target_sr
        i0, i1 = int(start * sr), int((start + duration) * sr)
        data = data[i0:min(i1, len(data))]
        data = normalize(pre_emphasis(data, PREEMPH))
        return data, sr, f"{os.path.basename(path)} [{start:.1f}-{start + duration:.1f}s], mono {sr} Hz"
    return synthesize_utterance(target_sr, duration=duration), target_sr, "synthetic /a-i-u/ utterance (no WAV file)"


def normalize(sig, peak=0.95):
    sig = np.asarray(sig, np.float64)
    sig = sig - sig.mean()
    m = np.max(np.abs(sig))
    return sig * (peak / m) if m > 0 else sig


def pre_emphasis(sig, coeff=PREEMPH):
    return np.append(sig[0], sig[1:] - coeff * sig[:-1])


def frame_signal(sig, sr=SR, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Split into overlapping frames -> (n_frames, frame_len)."""
    fl, hp = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(sig) < fl:
        sig = np.pad(sig, (0, fl - len(sig)))
    n = 1 + (len(sig) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return sig[idx]


def hamming_window(n):
    """Hamming window written from the definition: 0.54 - 0.46 cos(2 pi k /(N-1))."""
    k = np.arange(n)
    return 0.54 - 0.46 * np.cos(2 * np.pi * k / (n - 1))


def power_spectrum(frames, n_fft=N_FFT):
    """|FFT|^2 / n_fft of every windowed frame -> (n_frames, n_fft//2+1)."""
    spec = np.fft.rfft(frames, n_fft, axis=1)
    return (np.abs(spec) ** 2) / n_fft


# =============================================================================
# 1. MFCC
# =============================================================================
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20.0, fmax=None):
    """Triangular filters equally spaced on the Mel scale -> (n_mels, n_fft//2+1)."""
    fmax = fmax or sr / 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    filters = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, ce, hi = edges[i], edges[i + 1], edges[i + 2]
        filters[i] = np.maximum(0.0, np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce)))
    return filters


def extract_mfcc(sig, sr=SR, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT,
                 frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """MFCC pipeline. Returns (coefficients (n_frames, n_mfcc), intermediate dict)."""
    frames = frame_signal(sig, sr, frame_ms, hop_ms)
    windowed = frames * hamming_window(frames.shape[1])
    power = power_spectrum(windowed, n_fft)
    filters = mel_filterbank(sr, n_fft, n_mels)
    mel_energy = power @ filters.T
    log_mel = np.log(np.maximum(mel_energy, 1e-12))
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    info = {"frames": frames, "windowed": windowed, "power": power, "filters": filters,
            "mel_energy": mel_energy, "log_mel": log_mel}
    return coeffs, info


# =============================================================================
# 2. PLP  (Hermansky, 1990)
# =============================================================================
def hz_to_bark(f):
    f = np.asarray(f, np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def bark_to_hz(b):
    """Invert hz_to_bark numerically on a fine grid."""
    grid = np.linspace(0, SR / 2, 20001)
    return np.interp(b, hz_to_bark(grid), grid)


def bark_filterbank(sr=SR, n_fft=N_FFT, n_bark=N_BARK, fmin=20.0, fmax=None):
    """Triangular critical-band filters equally spaced on the Bark scale."""
    fmax = fmax or sr / 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = bark_to_hz(np.linspace(hz_to_bark(fmin), hz_to_bark(fmax), n_bark + 2))
    filters = np.zeros((n_bark, len(freqs)))
    for i in range(n_bark):
        lo, ce, hi = edges[i], edges[i + 1], edges[i + 2]
        filters[i] = np.maximum(0.0, np.minimum((freqs - lo) / max(ce - lo, 1e-9),
                                                (hi - freqs) / max(hi - ce, 1e-9)))
    return filters


def equal_loudness_curve(freqs):
    """E(w) from Hermansky (1990): E(w) = (w^2+56.8e6) w^4 / ((w^2+6.3e6)^2 (w^2+0.38e9))."""
    w = 2 * np.pi * np.maximum(freqs, 1.0)
    return ((w ** 2 + 56.8e6) * w ** 4) / (((w ** 2 + 6.3e6) ** 2) * (w ** 2 + 0.38e9))


def levinson_durbin(r, order):
    """Solve the normal equations from autocorrelation r -> (a, prediction error).
    a[0] = 1 and A(z) = 1 + a[1] z^-1 + ... (Levinson-Durbin recursion)."""
    r = np.asarray(r, np.float64)
    if r[0] <= 1e-12:
        return np.concatenate([[1.0], np.zeros(order)]), 1e-12
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / err
        a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        err *= max(1.0 - k * k, 1e-12)
    return a, err


def extract_plp(sig, sr=SR, order=PLP_ORDER, n_cep=N_CEP, n_bark=N_BARK,
                n_fft=N_FFT, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """PLP pipeline. Returns (coefficients (n_frames, n_cep), intermediate dict)."""
    frames = frame_signal(sig, sr, frame_ms, hop_ms)
    windowed = frames * hamming_window(frames.shape[1])
    power = power_spectrum(windowed, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    eq = equal_loudness_curve(freqs)
    power_eq = power * eq                                     # equal-loudness pre-emphasis
    filters = bark_filterbank(sr, n_fft, n_bark)
    bark_spec = power_eq @ filters.T                          # critical-band integration
    loudness = bark_spec ** 0.33                              # intensity -> loudness (cube root)

    r = np.fft.irfft(loudness, n=2 * (n_bark - 1), axis=1)[:, :n_bark]   # IDFT -> autocorrelation
    coeffs = np.zeros((len(frames), n_cep))
    a_matrix = np.zeros((len(frames), order + 1))
    for i, row in enumerate(r):
        a, err = levinson_durbin(row, order)
        a_matrix[i] = a
        coeffs[i] = lpc_to_cepstrum(a, n_cep, gain=np.sqrt(max(err, 1e-12)))
    info = {"frames": frames, "windowed": windowed, "power": power, "eq_curve": eq,
            "power_eq": power_eq, "filters": filters, "bark_spec": bark_spec,
            "loudness": loudness, "autocorr": r, "lpc": a_matrix}
    return coeffs, info


# =============================================================================
# 3. LPC / LPCC
# =============================================================================
def lpc_to_cepstrum(a, n_cep=N_CEP, gain=1.0):
    """Cepstral coefficients of the all-pole model 1/A(z) via the standard recursion:
    c0 = log(G), c1 = -a1, c_m = -a_m - sum_{k=1}^{m-1} (k/m) c_k a_{m-k}."""
    p = len(a) - 1
    c = np.zeros(n_cep)
    c[0] = np.log(max(gain, 1e-12))
    for m in range(1, n_cep):
        am = a[m] if m <= p else 0.0
        acc = sum((k / m) * c[k] * (a[m - k] if m - k <= p else 0.0) for k in range(1, m))
        c[m] = -am - acc
    return c


def lpc_spectrum(a, sr=SR, n_fft=N_FFT, db=True):
    """|1/A(e^{jw})| spectrum of an LPC model (for plotting)."""
    from scipy.signal import freqz
    w, h = freqz([1.0], a, worN=n_fft, fs=sr)
    mag = np.abs(h)
    return w, 20 * np.log10(np.maximum(mag, 1e-12)) if db else mag


def extract_lpcc(sig, sr=SR, order=LPC_ORDER, n_cep=N_CEP, n_fft=N_FFT,
                 frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """LPCC pipeline: autocorrelation LPC (Levinson-Durbin) + cepstral recursion.
    Returns (coefficients (n_frames, n_cep), intermediate dict)."""
    frames = frame_signal(sig, sr, frame_ms, hop_ms)
    windowed = frames * hamming_window(frames.shape[1])
    n_pad = 1 << int(np.ceil(np.log2(2 * windowed.shape[1])))
    spec = np.fft.rfft(windowed, n_pad, axis=1)
    autocorr = np.fft.irfft(np.abs(spec) ** 2, n_pad, axis=1)[:, :order + 1] / max(windowed.shape[1], 1)
    coeffs = np.zeros((len(frames), n_cep))
    a_matrix = np.zeros((len(frames), order + 1))
    errors = np.zeros(len(frames))
    for i, row in enumerate(autocorr):
        a, err = levinson_durbin(row, order)
        a_matrix[i], errors[i] = a, err
        coeffs[i] = lpc_to_cepstrum(a, n_cep, gain=np.sqrt(max(err, 1e-12)))
    info = {"frames": frames, "windowed": windowed, "autocorr": autocorr,
            "lpc": a_matrix, "errors": errors}
    return coeffs, info


# =============================================================================
# 4. GFCC
# =============================================================================
def hz_to_erb(f):
    return 21.4 * np.log10(1.0 + 4.37 * np.asarray(f, np.float64) / 1000.0)


def erb_to_hz(e):
    return (10 ** (e / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_filterbank(sr=SR, n_fft=N_FFT, n_filters=N_GAMMATONE, fmin=50.0, fmax=None):
    """4th-order gammatone magnitude responses, equally spaced on the ERB scale:
    |H(f)| = (1 + ((f-fc)/b)^2)^-2 with b = 1.019 * ERB(fc)."""
    fmax = fmax or min(sr / 2, 7800.0)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    centers = erb_to_hz(np.linspace(hz_to_erb(fmin), hz_to_erb(fmax), n_filters))
    filters = np.zeros((n_filters, len(freqs)))
    for i, fc in enumerate(centers):
        b = 1.019 * 24.7 * (4.37 * fc / 1000.0 + 1.0)
        filters[i] = (1.0 + ((freqs - fc) / b) ** 2) ** -2
        filters[i] /= filters[i].max()
    return filters


def extract_gfcc(sig, sr=SR, n_filters=N_GAMMATONE, n_cep=N_CEP, n_fft=N_FFT,
                 frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """GFCC pipeline: gammatone filterbank + cube-root compression + DCT.
    Returns (coefficients (n_frames, n_cep), intermediate dict)."""
    frames = frame_signal(sig, sr, frame_ms, hop_ms)
    windowed = frames * hamming_window(frames.shape[1])
    power = power_spectrum(windowed, n_fft)
    filters = gammatone_filterbank(sr, n_fft, n_filters)
    energies = power @ filters.T
    compressed = np.maximum(energies, 1e-12) ** 0.33           # cochlear compression
    coeffs = dct(np.log(compressed), type=2, axis=1, norm="ortho")[:, :n_cep]
    info = {"frames": frames, "windowed": windowed, "power": power,
            "filters": filters, "energies": energies, "compressed": compressed}
    return coeffs, info


# =============================================================================
# Comparison and robustness
# =============================================================================
def feature_stats(name, feat):
    """Summary numbers used in the comparison table."""
    return {"feature": name, "n_frames": len(feat), "n_dims": feat.shape[1],
            "mean_abs": float(np.mean(np.abs(feat))),
            "temporal_std": float(np.mean(feat.std(axis=0))),
            "dynamic_range": float(feat.max() - feat.min())}


def add_white_noise(sig, snr_db, rng):
    """White Gaussian noise scaled to the requested SNR (power ratio) of the signal."""
    p_sig = np.mean(sig ** 2)
    noise = rng.standard_normal(len(sig)) * np.sqrt(p_sig / 10 ** (snr_db / 10))
    return sig + noise


def frame_cosine(feat_a, feat_b, drop_first=True):
    """Mean per-frame cosine similarity between two feature matrices (optionally
    ignoring coefficient 0, which only carries frame energy)."""
    a = feat_a[:, 1:] if drop_first else feat_a
    b = feat_b[:, 1:] if drop_first else feat_b
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(a * b, axis=1)))


# =============================================================================
# Main demo
# =============================================================================
def _plot_all(sig, sr, mfcc_f, plp_f, lpcc_f, gfcc_f, infos, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(sig)) / sr
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, (name, feat) in zip(axes.ravel(), [("MFCC", mfcc_f), ("PLP", plp_f),
                                               ("LPCC", lpcc_f), ("GFCC", gfcc_f)]):
        show = feat[:, 1:].T                     # c0 carries frame energy and hides the rest
        lo, hi = np.percentile(show, [2, 98])
        lim = max(abs(lo), abs(hi))
        im = ax.imshow(show, aspect="auto", origin="lower", cmap="viridis",
                       vmin=-lim, vmax=lim,
                       extent=[0, len(sig) / sr, 0.5, feat.shape[1] - 0.5])
        ax.set_title(f"{name}  (c1..c{feat.shape[1] - 1} x {feat.shape[0]} frames, c0 dropped)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("coefficient")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"Feature comparison - {len(sig) / sr:.2f} s of speech")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_comparison.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    axes[0, 0].plot(np.fft.rfftfreq(N_FFT, 1 / sr), infos["mfcc"]["filters"].T)
    axes[0, 0].set_title(f"Mel filterbank ({N_MELS} filters)")
    axes[0, 1].plot(np.fft.rfftfreq(N_FFT, 1 / sr), infos["plp"]["filters"].T)
    axes[0, 1].set_title(f"Bark critical bands ({N_BARK} bands)")
    axes[1, 0].plot(np.fft.rfftfreq(N_FFT, 1 / sr), infos["gfcc"]["filters"].T)
    axes[1, 0].set_title(f"Gammatone filterbank ({N_GAMMATONE} filters, ERB)")
    for idx in [len(infos["lpcc"]["frames"]) // 3, 2 * len(infos["lpcc"]["frames"]) // 3]:
        w, h = lpc_spectrum(infos["lpcc"]["lpc"][idx], sr)
        axes[1, 1].plot(w, h, label=f"frame {idx}")
    axes[1, 1].set_title(f"LPC spectra (order {LPC_ORDER}, dB)")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "filterbanks.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Lab 8 - MFCC / PLP / LPCC / GFCC extraction and comparison")
    ap.add_argument("--wav", help="WAV file to analyse (default: lab 1 harvard.wav if present)")
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic /a-i-u/ utterance instead of a file")
    ap.add_argument("--start", type=float, default=1.0, help="start time in the file (seconds)")
    ap.add_argument("--duration", type=float, default=3.0, help="segment duration (seconds)")
    ap.add_argument("--snr", type=float, default=10.0, help="SNR of the noisy copy (dB)")
    ap.add_argument("--out", default=RESULTS_DIR, help="output directory")
    args = ap.parse_args()

    default_wav = os.path.normpath(os.path.join(
        HERE, "..", "..", "lab 1", "solution", "POS, parser and BOW",
        "1) POS, parser and BoW", "harvard.wav"))
    wav = None if args.synthetic else (args.wav or (default_wav if os.path.exists(default_wav) else None))

    banner("TASK 1 - Load and preprocess the speech signal")
    sig, sr, source = load_speech(wav, start=args.start, duration=args.duration)
    print(f"source         : {source}")
    print(f"sample rate    : {sr} Hz")
    print(f"length         : {len(sig)} samples ({len(sig) / sr:.2f} s)")
    print(f"pre-emphasis   : y[n] = x[n] - {PREEMPH} x[n-1], then DC removal + peak normalisation")
    print(f"RMS / peak     : {np.sqrt(np.mean(sig ** 2)):.4f} / {np.abs(sig).max():.4f}")

    banner("TASK 2 - Framing and windowing")
    frames = frame_signal(sig, sr)
    print(f"frame length   : {FRAME_MS:.0f} ms ({frames.shape[1]} samples), hop {HOP_MS:.0f} ms")
    print(f"frames         : {frames.shape[0]}")
    print(f"window         : Hamming, first/last value {hamming_window(frames.shape[1])[0]:.3f}/{hamming_window(frames.shape[1])[-1]:.3f}")

    banner("TASK 3 - MFCC (Mel-frequency cepstral coefficients)")
    mfcc_f, mfcc_info = extract_mfcc(sig, sr)
    print(f"pipeline       : pre-emphasis -> {frames.shape[0]} frames x Hamming -> {N_FFT}-point FFT")
    print(f"                 -> {N_MELS} triangular Mel filters -> log -> DCT-II -> keep c{N_MFCC - 1}")
    print(f"MFCC matrix    : {mfcc_f.shape}  (range {mfcc_f.min():.2f} .. {mfcc_f.max():.2f})")

    banner("TASK 4 - PLP (perceptual linear prediction, Hermansky 1990)")
    plp_f, plp_info = extract_plp(sig, sr)
    print(f"pipeline       : power spectrum -> equal-loudness pre-emphasis")
    print(f"                 -> {N_BARK} Bark critical bands -> cube-root loudness -> IDFT")
    print(f"                 -> LPC order {PLP_ORDER} (Levinson-Durbin) -> cepstral recursion, {N_CEP} coefficients")
    print(f"PLP matrix     : {plp_f.shape}  (range {plp_f.min():.2f} .. {plp_f.max():.2f})")

    banner("TASK 5 - LPC coefficients and LPCC")
    lpcc_f, lpcc_info = extract_lpcc(sig, sr)
    a_mid = lpcc_info["lpc"][len(lpcc_info["lpc"]) // 2]
    print(f"LPC order      : {LPC_ORDER}  (autocorrelation method, Levinson-Durbin)")
    print(f"example frame  : 1 + {np.array2string(a_mid[1: 6], precision=3)} ...  (a1..a5 of frame {len(lpcc_info['lpc']) // 2})")
    print(f"mean residual  : {lpcc_info['errors'].mean():.3e}")
    print(f"LPCC matrix    : {lpcc_f.shape}  (range {lpcc_f.min():.2f} .. {lpcc_f.max():.2f})")

    banner("TASK 6 - GFCC (gammatone frequency cepstral coefficients)")
    gfcc_f, gfcc_info = extract_gfcc(sig, sr)
    print(f"pipeline       : power spectrum -> {N_GAMMATONE} gammatone filters (ERB, 4th order)")
    print(f"                 -> cube-root compression -> log -> DCT-II -> {N_CEP} coefficients")
    print(f"GFCC matrix    : {gfcc_f.shape}  (range {gfcc_f.min():.2f} .. {gfcc_f.max():.2f})")

    banner("TASK 7 - Visualisation and comparison")
    os.makedirs(args.out, exist_ok=True)
    _plot_all(sig, sr, mfcc_f, plp_f, lpcc_f, gfcc_f,
              {"mfcc": mfcc_info, "plp": plp_info, "lpcc": lpcc_info, "gfcc": gfcc_info}, args.out)
    stats = [feature_stats("MFCC", mfcc_f), feature_stats("PLP", plp_f),
             feature_stats("LPCC", lpcc_f), feature_stats("GFCC", gfcc_f)]
    with open(os.path.join(args.out, "feature_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    for s in stats:
        print(f"{s['feature']:<5} {s['n_frames']:>4} frames x {s['n_dims']:>2} dims | "
              f"mean|v| {s['mean_abs']:>6.2f} | mean temporal std {s['temporal_std']:>6.2f} | "
              f"range {s['dynamic_range']:>6.2f}")
    print(f"figures        : {os.path.join(args.out, 'feature_comparison.png')}, {os.path.join(args.out, 'filterbanks.png')}")

    banner(f"TASK 8 - Suitability for speech recognition (noise robustness, SNR {args.snr:.0f} dB)")
    rng = np.random.default_rng(0)
    noisy = add_white_noise(sig, args.snr, rng)
    ext = {"MFCC": extract_mfcc, "PLP": extract_plp, "LPCC": extract_lpcc, "GFCC": extract_gfcc}
    clean_feats = {"MFCC": mfcc_f, "PLP": plp_f, "LPCC": lpcc_f, "GFCC": gfcc_f}
    rows = []
    for name, fn in ext.items():
        noisy_f, _ = fn(noisy, sr)
        n = min(len(clean_feats[name]), len(noisy_f))
        sim = frame_cosine(clean_feats[name][:n], noisy_f[:n])
        rows.append({"feature": name, "cosine_similarity": sim,
                     "mean_abs": float(np.mean(np.abs(clean_feats[name])))})
        print(f"{name:<5} cosine similarity (clean vs noisy, coefficients 1..): {sim:.4f}")
    with open(os.path.join(args.out, "robustness.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "cosine_similarity", "mean_abs"])
        w.writeheader()
        w.writerows(rows)
    print("\nInterpretation: higher cosine similarity means the feature vector survives")
    print("noise better. MFCC/GFCC use fine spectral detail; PLP's all-pole smoothing and")
    print("equal-loudness weighting usually make it the most stable, LPCC is compact but")
    print("sensitive to the noisy autocorrelation estimate. See the notebook for details.")


if __name__ == "__main__":
    main()
