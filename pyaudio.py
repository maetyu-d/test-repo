#!/usr/bin/env python3
"""
Transcendent post-mastering pass:

✅ Harmonic spectral freezing ONLY on chords (not clicks/noise)
✅ Shifting formant resonators (Tim Hecker / Fennesz smear)
✅ Climax + release arc for emotional payoff

Run:
  python transcendent_master.py --in input.wav --out output.wav
"""

import argparse
import numpy as np
import soundfile as sf


def moving_average(x, win):
    """Extract harmonic bed via heavy smoothing."""
    if win <= 1:
        return x.copy()
    k = np.ones(win) / win
    y = np.zeros_like(x)
    for ch in range(x.shape[1]):
        y[:, ch] = np.convolve(x[:, ch], k, mode="same")
    return y


def freeze_harmonic(harmonic, win, hop, amount, phase_jitter, stride, seed):
    """Spectral freeze bloom applied ONLY to harmonic layer."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(harmonic)
    window = np.hanning(win)

    frame = 0
    for i in range(0, len(harmonic) - win, hop):
        frame += 1
        if stride > 1 and (frame % stride) != 0:
            continue

        chunk = harmonic[i:i+win] * window[:, None]
        spec = np.fft.rfft(chunk, axis=0)

        mag = np.abs(spec)
        ph = np.angle(spec)

        # phase diffusion → overtone cloud
        ph += rng.uniform(-phase_jitter, phase_jitter, size=ph.shape)

        frozen = mag * np.exp(1j * ph)
        recon = np.fft.irfft(frozen, axis=0).real

        out[i:i+win] += recon * amount

    return out


def add_formants(x, sr, amount):
    """Shifting formant resonator layer (Hecker/Fennesz style)."""
    n = x.shape[0]
    t = np.linspace(0, n / sr, n, endpoint=False)

    centers = (300, 800, 2000)
    sweep_rate = 0.0007
    sweep_depth = 0.4

    form = np.zeros_like(x)

    for c in centers:
        sweep = c * (1 + sweep_depth * np.sin(2*np.pi*sweep_rate*t))
        osc = np.sin(2*np.pi*sweep*t) * amount
        form[:, 0] += osc
        form[:, 1] += osc

    return x + form


def apply_arc(x):
    """Climax + release envelope arc."""
    n = x.shape[0]
    env = np.ones(n)

    cs = int(n * 0.55)
    ce = int(n * 0.78)

    env[:cs] = np.linspace(0.85, 1.0, cs)
    env[cs:ce] = np.linspace(1.0, 1.40, ce - cs)
    env[ce:] = np.linspace(1.40, 0.70, n - ce)

    return x * env[:, None]


def normalize(x, peak=0.95):
    p = np.max(np.abs(x))
    return x if p < 1e-12 else x / p * peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out",
                    default="sensual_alvanoto_transcendent_master.wav")

    ap.add_argument("--smooth_win", type=int, default=12000)
    ap.add_argument("--win", type=int, default=8192)
    ap.add_argument("--hop_mult", type=int, default=2)

    ap.add_argument("--freeze_amount", type=float, default=0.6)
    ap.add_argument("--phase_jitter", type=float, default=0.15)
    ap.add_argument("--freeze_stride", type=int, default=2)

    ap.add_argument("--formant_amount", type=float, default=0.015)

    args = ap.parse_args()

    x, sr = sf.read(args.inp)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)

    # 1) Separate chord bed vs clicks/noise
    harmonic = moving_average(x, args.smooth_win)
    residual = x - harmonic

    # 2) Harmonic-only freeze bloom
    hop = args.win // args.hop_mult
    frozen = freeze_harmonic(
        harmonic,
        args.win,
        hop,
        args.freeze_amount,
        args.phase_jitter,
        args.freeze_stride,
        seed=2,
    )

    bed = harmonic * 0.5 + frozen * 0.5

    # 3) Formant smear
    bed = add_formants(bed, sr, args.formant_amount)

    # 4) Climax + release arc
    bed = apply_arc(bed)

    # 5) Recombine residual (keeps clicks sharp)
    y = normalize(bed + residual)

    sf.write(args.out, y, sr, subtype="PCM_24")
    print("✅ Wrote:", args.out)


if __name__ == "__main__":
    main()
