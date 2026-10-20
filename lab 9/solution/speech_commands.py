"""
Lab 9 - Synthetic spoken-command dataset (spoken digits zero..nine)
===================================================================

The CNN of `asr_cnn.py` needs speech. To keep the lab completely offline and
reproducible, the dataset is generated in code with a small **formant (source-filter)
speech synthesizer**, the classic model of speech production:

    pitch (pulse train) + noise  ->  glottal shaping  ->  formant resonators  ->  digits

Every simulated **speaker** gets its own vocal-tract length (formant scaling), pitch,
speaking rate and noise level, so the held-out test speakers are voices the model has
never heard. The ten words are caricatures of the English digits, defined in `WORDS` as
phoneme sequences (`VOWELS`, `DIPHTHONGS`, `GLIDES`, `NASALS`, `FRICATIVES`, `STOPS`).

A real dataset can be used instead: `load_audio_dir(root)` expects one sub-folder per
word (as in the Google Speech Commands dataset) with `.wav` files.

Run (quick demo)
    python speech_commands.py                 # generate the default dataset, save 10 sample WAVs
    python speech_commands.py --speakers 40 --out samples/
"""

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, resample_poly

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SR = 16000
TWO_PI = 2 * np.pi
CHUNK_MS = 10                                     # filter coefficients update every 10 ms
VOICED_LEVEL = 8.0                                # excitation levels of the two sound sources
NOISE_LEVEL = 0.35

# --------------------------------------------------------------------------- phoneme table
VOWELS = {"AA": (730, 1090, 2440), "IY": (270, 2290, 3010), "UW": (300, 870, 2240),
          "AH": (640, 1190, 2390), "EH": (530, 1840, 2480), "IH": (390, 1990, 2550),
          "AO": (570, 840, 2410), "ER": (490, 1350, 1690)}
DIPHTHONGS = {"EY": ((400, 2200, 2800), (300, 2300, 3000)),
              "AY": ((700, 1200, 2500), (350, 2100, 2900)),
              "OW": ((500, 900, 2400), (350, 700, 2400))}
GLIDES = {"W": (290, 610, 2150), "Y": (260, 2070, 3020), "R": (310, 1060, 1380)}
NASALS = {"N": (280, 1400, 2300), "M": (280, 1100, 2200)}
FRICATIVES = {"S": (5500, 1600), "Z": (5000, 1800), "F": (4000, 2500),
              "V": (2800, 2000), "TH": (4800, 2200)}          # (noise band centre, bandwidth)
STOPS = {"P": 1200, "T": 3500, "K": 2200, "B": 1000, "D": 2500, "G": 1800}  # burst centre

WORDS = {"zero": "Z IY ER OW", "one": "W AH N", "two": "T UW", "three": "TH R IY",
         "four": "F AO R", "five": "F AY V", "six": "S IH K S", "seven": "S EH V AH N",
         "eight": "EY T", "nine": "N AY N"}
DIGITS = list(WORDS)


def phoneme_segments(word):
    """Expand a word's phoneme string into renderable segments (closure/burst included)."""
    segs = []
    for ph in word.split():
        if ph in VOWELS:
            segs.append(dict(dur=110, voiced=1.0, noise=0.0, f=VOWELS[ph], gain=1.0))
        elif ph in DIPHTHONGS:
            segs.append(dict(dur=160, voiced=1.0, noise=0.0, f=DIPHTHONGS[ph], gain=1.0))
        elif ph in GLIDES:
            segs.append(dict(dur=80, voiced=1.0, noise=0.0, f=GLIDES[ph], gain=0.85))
        elif ph in NASALS:
            segs.append(dict(dur=95, voiced=1.0, noise=0.0, f=NASALS[ph], gain=0.55))
        elif ph in FRICATIVES:
            center, bw = FRICATIVES[ph]
            voiced = 0.45 if ph in ("Z", "V") else 0.0
            segs.append(dict(dur=115, voiced=voiced, noise=0.85, f=(500, 1500, 2500),
                             band=(center, bw), gain=0.9))
        elif ph in STOPS:
            center = STOPS[ph]
            voicing = 0.06 if ph in ("B", "D", "G") else 0.0
            segs.append(dict(dur=45, voiced=voicing, noise=0.0, f=(300, 1300, 2200), gain=0.4))
            segs.append(dict(dur=12, voiced=0.0, noise=1.0, f=(500, 1500, 2500),
                             band=(center, 1500), gain=0.9))
            if voicing == 0.0:                            # aspiration after voiceless stops
                segs.append(dict(dur=30, voiced=0.0, noise=0.6, f=(500, 1500, 2500),
                                 band=(max(center * 0.7, 1000), 2000), gain=0.25))
        else:
            raise ValueError(f"unknown phoneme: {ph}")
    return segs


def make_speaker(rng):
    """Random speaker profile: pitch, formant scaling (vocal-tract length), rate, level."""
    return {"f0": float(np.clip(rng.normal(135, 35), 80, 240)),
            "formant_scale": float(np.clip(rng.normal(1.0, 0.08), 0.82, 1.18)),
            "rate": float(np.clip(rng.normal(1.0, 0.09), 0.78, 1.25)),
            "gain": float(rng.uniform(0.5, 1.0))}


def _resonator(f, bw, sr=SR):
    """2nd-order all-pole resonator at frequency f with bandwidth bw (band-pass,
    b=[1-r]); used for the noise source of fricatives and bursts."""
    r = np.exp(-np.pi * bw / sr)
    a = [1.0, -2 * r * np.cos(TWO_PI * f / sr), r * r]
    return [1 - r], a


def _formant(f, bw, sr=SR):
    """Formant resonator normalised to unity gain at DC (classic Klatt cascade), so
    that cascading three formants shapes the vowel spectrum instead of muting it."""
    r = np.exp(-np.pi * bw / sr)
    a = [1.0, -2 * r * np.cos(TWO_PI * f / sr), r * r]
    return [1.0 + a[1] + a[2]], a


def synthesize_word(word, speaker, rng, sr=SR):
    """Render one word for one speaker -> waveform (float, peak 0.9)."""
    segs = phoneme_segments(WORDS[word])
    scale = speaker["formant_scale"]
    rate = speaker["rate"]
    durs = np.array([s["dur"] for s in segs]) * rate
    total_ms = float(durs.sum())
    n = int(sr * total_ms / 1000)
    t_ms = np.arange(n) / sr * 1000.0

    bounds = np.concatenate([[0.0], np.cumsum(durs)])
    seg_idx = np.clip(np.searchsorted(bounds, t_ms, side="right") - 1, 0, len(segs) - 1)
    prog = np.clip((t_ms - bounds[seg_idx]) / np.maximum(durs[seg_idx], 1e-9), 0.0, 1.0)

    def formants_at(k, p):
        f = segs[k]["f"]
        if isinstance(f[0], tuple):
            return tuple((1 - p) * a * scale + p * b * scale for a, b in zip(*f))
        return tuple(v * scale for v in f)

    # pitch contour: declination + vibrato, one value per sample (sampled per chunk)
    f0 = speaker["f0"] * (1.12 - 0.24 * t_ms / total_ms) * (1 + 0.015 * np.sin(TWO_PI * 2.5 * t_ms / 1000))

    # amplitude envelope from segment gains, smoothed so there are no clicks
    gain_env = np.array([s["gain"] for s in segs])[seg_idx]
    win = 0.5 * (1 - np.cos(np.linspace(0, TWO_PI, int(0.006 * sr), endpoint=False)))
    gain_env = np.convolve(gain_env, win / win.sum(), mode="same")
    fade = np.minimum(1.0, np.arange(n) / (0.01 * sr)) * np.minimum(1.0, (n - np.arange(n)) / (0.01 * sr))

    out = np.zeros(n)
    chunk = int(CHUNK_MS / 1000 * sr)
    glottal_b, glottal_a = [0.2], [1.0, -0.8]         # 2 x one-pole lowpass (glottal rolloff)
    radiation_b, radiation_a = [1.0, -0.85], [1.0]   # vocal-tract radiation: +6 dB/oct tilt
    st_g = np.zeros(1); st_g2 = np.zeros(1); st_r = np.zeros(1)
    st_f = [np.zeros(2) for _ in range(3)]; st_n = np.zeros(2)
    phase = 0.0

    for k in range(0, n, chunk):
        sl = slice(k, min(k + chunk, n))
        L = sl.stop - sl.start
        if L <= 0:
            break
        seg = segs[seg_idx[sl.start]]
        f = formants_at(seg_idx[sl.start], prog[sl.start])
        band = seg.get("band", (500, 1500))

        dphi = TWO_PI * f0[sl.start] / sr
        phases = phase + dphi * np.arange(1, L + 1)
        pulses = np.zeros(L)
        pulses[np.nonzero(np.floor(phases / TWO_PI) != np.floor((phases - dphi) / TWO_PI))[0]] = 1.0
        phase = phases[-1] % TWO_PI

        y, st_g = lfilter(glottal_b, glottal_a, seg["voiced"] * pulses, zi=st_g)  # glottal source
        y, st_g2 = lfilter(glottal_b, glottal_a, y, zi=st_g2)
        for depth in range(3):                                                    # vocal tract
            b, a = _formant(f[depth], (80, 110, 170)[depth])
            y, st_f[depth] = lfilter(b, a, y, zi=st_f[depth])
        y, st_r = lfilter(radiation_b, radiation_a, y, zi=st_r)                   # lip radiation tilt

        yn, st_n = lfilter(*_resonator(band[0], band[1]), rng.standard_normal(L), zi=st_n)
        yn /= np.sqrt(np.mean(yn ** 2)) + 1e-12                                   # unit-RMS noise

        out[sl] = (VOICED_LEVEL * y + NOISE_LEVEL * seg["noise"] * yn) * gain_env[sl] * speaker["gain"]

    out -= out.mean()
    out *= 0.9 / max(np.abs(out).max(), 1e-9) * fade
    return out.astype(np.float64)


def add_noise(wave, snr_db, rng):
    """Additive white Gaussian noise at the requested SNR (dB)."""
    p = np.mean(wave ** 2)
    return wave + rng.standard_normal(len(wave)) * np.sqrt(p / 10 ** (snr_db / 10))


def save_wav(path, wave, sr=SR):
    wavfile.write(path, sr, np.int16(np.clip(wave, -1, 1) * 32767))


def generate_dataset(n_speakers=120, words=None, seed=0, sr=SR,
                     splits=(0.70, 0.15, 0.15), snr_range=(25.0, 45.0)):
    """Speaker-disjoint dataset: train / dev / test speakers never overlap.

    Returns (data, info): data = {"train"|"dev"|"test": [{"wave", "label", "speaker"}, ...]},
    info = {"speakers": {...}, "words": [...], "sr": sr}."""
    rng = np.random.default_rng(seed)
    words = words or DIGITS
    speakers = [make_speaker(rng) for _ in range(n_speakers)]
    n_train = int(round(splits[0] * n_speakers))
    n_dev = int(round(splits[1] * n_speakers))
    groups = {"train": speakers[:n_train],
              "dev": speakers[n_train:n_train + n_dev],
              "test": speakers[n_train + n_dev:]}

    data, offsets = {}, {"train": 0, "dev": n_train, "test": n_train + n_dev}
    for name, sps in groups.items():
        items = []
        for j, sp in enumerate(sps):
            for w in words:
                wave = synthesize_word(w, sp, rng, sr)
                wave = add_noise(wave, rng.uniform(*snr_range), rng)   # breathing/room floor
                items.append({"wave": wave.astype(np.float32), "label": w,
                              "speaker": offsets[name] + j})
        data[name] = items
    info = {"speakers": {k: len(v) for k, v in groups.items()}, "words": words, "sr": sr}
    return data, info


def load_audio_dir(root, sr=SR, splits=(0.70, 0.15, 0.15), seed=0):
    """Load a real dataset: one sub-folder per word, `.wav` files (e.g. Google Speech
    Commands digits). Returns the same structure as generate_dataset()."""
    rng = np.random.default_rng(seed)
    words = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    files = []
    for w in words:
        for fn in os.listdir(os.path.join(root, w)):
            if fn.lower().endswith(".wav"):
                files.append((os.path.join(root, w, fn), w))
    rng.shuffle(files)
    n_train, n_dev = int(0.70 * len(files)), int(0.15 * len(files))
    data = {"train": [], "dev": [], "test": []}
    for k, (path, w) in enumerate(files):
        name = "train" if k < n_train else "dev" if k < n_train + n_dev else "test"
        fs, wave = wavfile.read(path)
        wave = np.asarray(wave)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if np.issubdtype(wave.dtype, np.integer):
            wave = wave / np.iinfo(wave.dtype).max
        if fs != sr:
            wave = resample_poly(wave, sr, fs)
        data[name].append({"wave": wave.astype(np.float32), "label": w, "speaker": None})
    info = {"speakers": {k: "n/a" for k in data}, "words": words, "sr": sr}
    return data, info


def main():
    ap = argparse.ArgumentParser(description="Lab 9 - synthetic spoken-digit dataset")
    ap.add_argument("--speakers", type=int, default=24, help="number of speakers to generate")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, info = generate_dataset(n_speakers=args.speakers, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    print(f"speakers train/dev/test: {info['speakers']}")
    for name in ("train", "dev", "test"):
        print(f"{name:5}: {len(data[name])} clips")
    for w in DIGITS:
        item = next(i for i in data['test'] if i['label'] == w)
        path = os.path.join(args.out, f"{w}.wav")
        save_wav(path, item["wave"])
        print(f"saved {path}  ({len(item['wave']) / SR:.2f} s)")
    print("now run:  python asr_cnn.py --quick")


if __name__ == "__main__":
    main()
