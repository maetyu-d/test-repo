#!/usr/bin/env python3
"""
Python 3.9-compatible end-to-end generator + mastering script.

Generates a piece (default 3:40) blending:
- Sensual Embrace: lush pads + emotional harmony + slow melodic counterlines
- Alva Noto: crisp clicks + crackle
- Xerrox-ish: grid-locked spectral slicing on bed+crackle ONLY (clicks untouched)
Then applies modern spectral mastering:
- Harmonic-only spectral freezing (only on chords/bed, not noise/clicks)
- Shifting formant resonators (Hecker/Fennesz-ish coloration)
- Climax + release arc
Ensures last ~10s are clicks only and crackle drops out there.

Output: 48kHz / 24-bit stereo WAV

Run:
  python generate_transcendent_piece_py39.py --out out.wav

If slow:
  - increase --freeze_stride (3 or 4)
  - increase --hop_mult (3)
  - reduce --freeze_win (4096)
"""

import argparse
import numpy as np
import soundfile as sf


# ----------------------------
# Utility
# ----------------------------
def normalize(x, peak=0.95):
    p = float(np.max(np.abs(x)))
    if p < 1e-12:
        return x
    return (x / p) * peak


def pan_mono(sig, pos):
    """pos: array in [-1..1]"""
    left = sig * (0.5 * (1.0 - pos))
    right = sig * (0.5 * (1.0 + pos))
    return np.stack([left, right], axis=1)


def sine_env(n, power=1.0):
    env = np.sin(np.linspace(0.0, np.pi, n)) ** power
    return env.astype(np.float64)


def moving_average_stereo(x, win):
    """Heavy smoothing to approximate harmonic bed (stereo)."""
    if win <= 1:
        return x.copy()
    k = np.ones(int(win), dtype=np.float64) / float(win)
    y = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        y[:, ch] = np.convolve(x[:, ch], k, mode="same")
    return y


# ----------------------------
# Sound sources
# ----------------------------
def lush_pad(sr, t):
    """Lush pad bed with chord progression. Returns stereo."""
    sections = [
        (0,   55,  [220.00, 261.63, 329.63]),   # A minor-ish
        (55,  110, [196.00, 246.94, 293.66]),   # Gm-ish
        (110, 165, [174.61, 220.00, 261.63]),   # F-ish
        (165, 220, [146.83, 185.00, 220.00]),   # D-ish
    ]

    pad = np.zeros(len(t), dtype=np.float64)

    for start, end, freqs in sections:
        idx = (t >= start) & (t < end)
        n = int(np.sum(idx))
        if n <= 0:
            continue

        env = sine_env(n, power=1.2)
        seg = np.zeros(n, dtype=np.float64)

        for f in freqs:
            seg += np.sin(2*np.pi*f*t[idx])
            seg += 0.65*np.sin(2*np.pi*(f*1.0045)*t[idx])  # detune warmth

        seg *= (0.75 / max(1, len(freqs)))
        pad[idx] += seg * env

    shimmer = np.sin(2*np.pi*880.0*t) * (0.02 + 0.02*np.sin(2*np.pi*0.01*t))
    pad += shimmer

    fade = int(sr * 5.0)
    env = np.ones_like(pad)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    pad *= env

    pos = np.sin(2*np.pi*0.001*t)
    return pan_mono(pad * 0.75, pos)


def slow_counterlines(sr, t):
    """Slow yearning counterlines (legato). Returns stereo."""
    scale = np.array([220, 247, 262, 294, 330, 392], dtype=np.float64)
    counter = np.zeros(len(t), dtype=np.float64)

    rng = np.random.default_rng(3)
    pos = 8.0

    while pos < (t[-1] - 18.0):
        dur = float(rng.uniform(6.0, 12.0))
        f = float(rng.choice(scale))
        idx = (t >= pos) & (t < pos + dur)
        n = int(np.sum(idx))
        if n > 32:
            env = sine_env(n, power=1.0)
            vibr = 0.003*np.sin(2*np.pi*0.33*t[idx])
            counter[idx] += np.sin(2*np.pi*(f*(1.0+vibr))*t[idx]) * env * 0.10

        pos += float(rng.uniform(10.0, 18.0))

    pan_pos = np.sin(2*np.pi*0.0015*t)
    return pan_mono(counter, pan_pos)


def variable_soft_distort(stereo, sr):
    """Variable soft distortion on a stereo stem (gentle)."""
    n = stereo.shape[0]
    tt = np.linspace(0.0, n/sr, n, endpoint=False, dtype=np.float64)
    drive = 0.4 + 0.6*np.sin(2*np.pi*0.004*tt)

    driven = np.copy(stereo)
    for ch in range(2):
        driven[:, ch] = np.tanh(stereo[:, ch] * (1.0 + drive*2.0))

    return stereo*0.65 + driven*0.35


def noto_clicks_and_crackle(sr, t, end_clicks_only_sec):
    """Return (clicks_stereo, crackle_stereo). Crackle drops out for last N seconds."""
    n = len(t)
    rng = np.random.default_rng(0)

    clicks = np.zeros(n, dtype=np.float64)
    click_count = 12000
    click_len = int(sr * 0.001)
    win = np.hanning(click_len).astype(np.float64)

    click_times = rng.choice(n - (click_len+1), size=click_count, replace=False)
    for ct in click_times:
        amp = float(rng.uniform(0.4, 1.0))
        clicks[ct:ct+click_len] += win * amp

    pulse = (np.sin(2*np.pi*3.0*t) > 0.995).astype(np.float64) * 0.8
    clicks += pulse

    pan_pos = np.sin(2*np.pi*0.07*t)
    clicks_st = pan_mono(clicks * 0.35, pan_pos)

    # Crackle/noise bed
    crackle = rng.standard_normal(n).astype(np.float64)
    crackle = np.tanh(crackle * 0.8) * (0.03 + 0.02*np.sin(2*np.pi*0.11*t))
    crackle_st = pan_mono(crackle, np.sin(2*np.pi*0.02*t))

    # Drop crackle during last N seconds
    tail_n = int(end_clicks_only_sec * sr)
    crackle_st[-tail_n:, :] *= 0.0

    return clicks_st, crackle_st


# ----------------------------
# Xerrox-style slicing (bed+crackle ONLY)
# ----------------------------
def gridlocked_spectral_slice(stereo_bed, sr, grid_rate=4.0, win=2048, hop=None):
    if hop is None:
        hop = win

    n = stereo_bed.shape[0]
    out = np.zeros_like(stereo_bed, dtype=np.float64)

    step = int(sr / grid_rate)
    if step < 1:
        step = 1

    w = np.hanning(win).astype(np.float64)

    for i in range(0, n - win, hop):
        chunk = stereo_bed[i:i+win] * w[:, None]
        spec = np.fft.rfft(chunk, axis=0)

        if (i // step) % 2 == 0:
            keep = np.arange(spec.shape[0]) > (spec.shape[0] * 0.35)
        else:
            keep = np.arange(spec.shape[0]) < (spec.shape[0] * 0.22)

        spec *= keep[:, None]
        recon = np.fft.irfft(spec, axis=0).real
        out[i:i+win] += recon

    return out


# ----------------------------
# Modern spectral mastering (harmonic-only freeze + formants + arc)
# ----------------------------
def freeze_harmonic_only(bed_stereo, sr,
                         smooth_win=12000,
                         win=8192,
                         hop_mult=2,
                         freeze_amount=0.6,
                         phase_jitter=0.15,
                         freeze_stride=2,
                         seed=2):
    """
    Returns (processed_harmonic_bed, residual_untouched)
    where residual = bed_stereo - smoothed harmonic bed.
    """
    harmonic = moving_average_stereo(bed_stereo, smooth_win)
    residual = bed_stereo - harmonic

    hop = max(1, int(win // max(1, hop_mult)))
    rng = np.random.default_rng(seed)

    out = np.zeros_like(harmonic, dtype=np.float64)
    w = np.hanning(win).astype(np.float64)

    frame = 0
    for i in range(0, harmonic.shape[0] - win, hop):
        frame += 1
        if freeze_stride > 1 and (frame % freeze_stride) != 0:
            continue

        chunk = harmonic[i:i+win] * w[:, None]
        spec = np.fft.rfft(chunk, axis=0)

        mag = np.abs(spec)
        ph = np.angle(spec)
        ph = ph + rng.uniform(-phase_jitter, phase_jitter, size=ph.shape)

        frozen = mag * np.exp(1j * ph)
        recon = np.fft.irfft(frozen, axis=0).real

        out[i:i+win] += recon * freeze_amount

    processed = harmonic * 0.5 + out * 0.5
    return processed, residual


def add_shifting_formants(stereo, sr, amount=0.015):
    n = stereo.shape[0]
    t = np.linspace(0.0, n/sr, n, endpoint=False, dtype=np.float64)

    centers = (300.0, 800.0, 2000.0)
    sweep_rate = 0.0007
    sweep_depth = 0.4

    form = np.zeros_like(stereo, dtype=np.float64)
    for c in centers:
        sweep = c * (1.0 + sweep_depth*np.sin(2*np.pi*sweep_rate*t))
        osc = np.sin(2*np.pi*sweep*t) * amount
        form[:, 0] += osc
        form[:, 1] += osc

    return stereo + form


def apply_climax_release_arc(stereo):
    n = stereo.shape[0]
    env = np.ones(n, dtype=np.float64)

    cs = int(n * 0.55)
    ce = int(n * 0.78)

    cs = max(1, min(cs, n - 2))
    ce = max(cs + 1, min(ce, n - 1))

    env[:cs] = np.linspace(0.85, 1.0, cs, dtype=np.float64)
    env[cs:ce] = np.linspace(1.0, 1.40, ce - cs, dtype=np.float64)
    env[ce:] = np.linspace(1.40, 0.70, n - ce, dtype=np.float64)

    return stereo * env[:, None]


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sensual_alvanoto_transcendent_master.wav")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--length_sec", type=float, default=220.0, help="Default 3:40 (220s)")
    ap.add_argument("--end_clicks_only_sec", type=float, default=10.0)

    # Performance knobs for freeze
    ap.add_argument("--freeze_win", type=int, default=8192)
    ap.add_argument("--hop_mult", type=int, default=2)
    ap.add_argument("--freeze_stride", type=int, default=2)

    args = ap.parse_args()

    sr = int(args.sr)
    dur = float(args.length_sec)
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)

    # 1) Bed (pads + counterlines)
    pad = lush_pad(sr, t)
    counter = slow_counterlines(sr, t)
    counter = variable_soft_distort(counter, sr)  # distort melodies only
    bed = pad + counter

    # 2) Clicks + crackle
    clicks_st, crackle_st = noto_clicks_and_crackle(sr, t, args.end_clicks_only_sec)

    # 3) Xerrox slicing ONLY on bed+crackle (NOT on clicks)
    bed_plus_crackle = bed + crackle_st
    sliced = gridlocked_spectral_slice(bed_plus_crackle, sr, grid_rate=4.0, win=2048, hop=2048)

    # Re-add untouched clicks
    mix = sliced + clicks_st

    # 4) Ensure last N seconds are clicks only (no bed, no crackle)
    tail_n = int(args.end_clicks_only_sec * sr)
    mix[-tail_n:, :] = clicks_st[-tail_n:, :]

    # 5) Modern spectral mastering on NON-click content
    non_click = mix - clicks_st  # everything except clicks

    frozen_bed, residual = freeze_harmonic_only(
        non_click, sr,
        smooth_win=12000,
        win=int(args.freeze_win),
        hop_mult=int(args.hop_mult),
        freeze_amount=0.6,
        phase_jitter=0.15,
        freeze_stride=int(args.freeze_stride),
        seed=2
    )

    frozen_bed = add_shifting_formants(frozen_bed, sr, amount=0.015)
    frozen_bed = apply_climax_release_arc(frozen_bed)

    mastered = (frozen_bed + residual) + clicks_st

    # Re-assert clicks-only ending after mastering
    mastered[-tail_n:, :] = clicks_st[-tail_n:, :]

    mastered = normalize(mastered, 0.95)
    sf.write(args.out, mastered, sr, subtype="PCM_24")
    print("✅ Wrote:", args.out)


if __name__ == "__main__":
    main()

