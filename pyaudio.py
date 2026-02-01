#!/usr/bin/env python3
"""
Generate + master a track (Sensual Embrace lushness + Alva Noto microsound),
with modern spectral mastering:

- Pads + emotional harmony (lush)
- Slow melodic counterlines (yearning motifs)
- Alva Noto clicks + crackle
- Grid-locked Xerrox-style spectral slicing on bed ONLY (clicks untouched)
- Variable soft-distortion on melody stem
- Harmonic-only spectral freezing (not noise/clicks)
- Shifting formant resonators (Hecker/Fennesz-ish smear)
- Climax + release arc
- End at 3:40 with last ~10 seconds: clicks only; crackle drops out there

Output: 48kHz / 24-bit stereo WAV
"""

import argparse
import numpy as np
import soundfile as sf


# ----------------------------
# Utility
# ----------------------------
def normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    p = float(np.max(np.abs(x)))
    return x if p < 1e-12 else (x / p) * peak


def pan_mono(sig: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """pos: -1..+1 array"""
    left = sig * (0.5 * (1 - pos))
    right = sig * (0.5 * (1 + pos))
    return np.stack([left, right], axis=1)


def adsr_sine_env(n: int, power: float = 1.0) -> np.ndarray:
    env = np.sin(np.linspace(0, np.pi, n)) ** power
    return env.astype(np.float64)


def moving_average_stereo(x: np.ndarray, win: int) -> np.ndarray:
    """Heavy smoothing to approximate harmonic bed."""
    if win <= 1:
        return x.copy()
    k = np.ones(win, dtype=np.float64) / win
    y = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        y[:, ch] = np.convolve(x[:, ch], k, mode="same")
    return y


# ----------------------------
# Sound sources
# ----------------------------
def lush_pad(sr: int, t: np.ndarray) -> np.ndarray:
    """
    Lush pad bed with slow chord progression.
    Returns stereo.
    """
    # Simple emotionally "minor-ish" triads (Hz), sectioned across 3:40
    # You can swap these for richer 7ths/9ths if you want.
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
        env = adsr_sine_env(n, power=1.2)

        seg = np.zeros(n, dtype=np.float64)
        for f in freqs:
            # detuned pair for warmth
            seg += np.sin(2*np.pi*f*t[idx])
            seg += 0.65*np.sin(2*np.pi*(f*1.0045)*t[idx])
        seg *= (0.75 / max(1, len(freqs)))  # normalize per chord size
        pad[idx] += seg * env

    # shimmer air
    shimmer = np.sin(2*np.pi*880.0*t) * (0.02 + 0.02*np.sin(2*np.pi*0.01*t))
    pad += shimmer

    # gentle overall fade edges
    fade = int(sr * 5)
    env = np.ones_like(pad)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    pad *= env

    # stereo drift
    pos = np.sin(2*np.pi*0.001*t)
    return pan_mono(pad * 0.75, pos)


def slow_counterlines(sr: int, t: np.ndarray) -> np.ndarray:
    """
    Slow melodic counterlines (legato) — stereo.
    """
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
            env = adsr_sine_env(n, power=1.0)
            vibr = 0.003*np.sin(2*np.pi*0.33*t[idx])
            counter[idx] += np.sin(2*np.pi*(f*(1+vibr))*t[idx]) * env * 0.10
        pos += float(rng.uniform(10.0, 18.0))

    # stereo drift separate from pad
    pan_pos = np.sin(2*np.pi*0.0015*t)
    return pan_mono(counter, pan_pos)


def variable_soft_distort(stereo: np.ndarray, sr: int) -> np.ndarray:
    """
    Variable soft distortion (tanh waveshaper) with slow-moving drive.
    Applied to the given stereo stem.
    """
    n = stereo.shape[0]
    t = np.linspace(0, n/sr, n, endpoint=False, dtype=np.float64)
    drive = 0.4 + 0.6*np.sin(2*np.pi*0.004*t)  # slow drive sweep
    driven = np.copy(stereo)
    for ch in range(2):
        driven[:, ch] = np.tanh(stereo[:, ch] * (1.0 + drive*2.0))
    # blend to keep it gentle
    return stereo*0.65 + driven*0.35


def noto_clicks_and_crackle(sr: int, t: np.ndarray, end_clicks_only_sec: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce (clicks_stereo, crackle_stereo).
    We'll later ensure crackle drops out during the last 10s.
    """
    n = len(t)
    rng = np.random.default_rng(0)

    # Click impulses: sparse, crisp
    clicks = np.zeros(n, dtype=np.float64)
    click_count = 12000  # adjust intensity
    click_times = rng.choice(n - int(sr*0.002), size=click_count, replace=False)
    click_len = int(sr * 0.001)  # 1ms
    win = np.hanning(click_len).astype(np.float64)

    for ct in click_times:
        amp = float(rng.uniform(0.4, 1.0))
        clicks[ct:ct+click_len] += win * amp

    # Grid pulse bursts (halved-ish tempo feel)
    pulse = (np.sin(2*np.pi*3.0*t) > 0.995).astype(np.float64) * 0.8
    clicks += pulse

    # Stereo flicker
    pan_pos = np.sin(2*np.pi*0.07*t)
    clicks_st = pan_mono(clicks * 0.35, pan_pos)

    # Crackle/noise bed (separate from clicks)
    crackle = rng.standard_normal(n).astype(np.float64)
    # simple "vinyl-ish" gating
    crackle = np.tanh(crackle * 0.8) * (0.03 + 0.02*np.sin(2*np.pi*0.11*t))
    crackle_st = pan_mono(crackle, np.sin(2*np.pi*0.02*t))

    # Drop crackle in last N seconds
    tail_n = int(end_clicks_only_sec * sr)
    crackle_st[-tail_n:, :] *= 0.0

    return clicks_st, crackle_st


# ----------------------------
# Xerrox-style slicing (bed ONLY)
# ----------------------------
def gridlocked_spectral_slice(stereo_bed: np.ndarray, sr: int, grid_rate: float = 4.0,
                             win: int = 2048, hop: int | None = None) -> np.ndarray:
    """
    Grid-locked band-window slicing.
    Applies ONLY to provided bed. (Keep clicks separate & untouched.)
    """
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
            keep = np.arange(spec.shape[0]) > spec.shape[0] * 0.35  # upper emphasis
        else:
            keep = np.arange(spec.shape[0]) < spec.shape[0] * 0.22  # low-mid emphasis

        spec *= keep[:, None]
        recon = np.fft.irfft(spec, axis=0).real
        out[i:i+win] += recon

    return out


# ----------------------------
# Modern spectral mastering (bed ONLY)
# ----------------------------
def freeze_harmonic_only(bed_stereo: np.ndarray, sr: int,
                         smooth_win: int = 12000,
                         win: int = 8192,
                         hop_mult: int = 2,
                         freeze_amount: float = 0.6,
                         phase_jitter: float = 0.15,
                         freeze_stride: int = 2,
                         seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (processed_bed, residual_untouched)
    where residual = original - smoothed harmonic.
    """
    harmonic = moving_average_stereo(bed_stereo, smooth_win)
    residual = bed_stereo - harmonic

    hop = max(1, win // max(1, hop_mult))
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
    # recombine residual later outside if you want residual untouched
    return processed, residual


def add_shifting_formants(stereo: np.ndarray, sr: int, amount: float = 0.015) -> np.ndarray:
    n = stereo.shape[0]
    t = np.linspace(0, n/sr, n, endpoint=False, dtype=np.float64)

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


def apply_climax_release_arc(stereo: np.ndarray) -> np.ndarray:
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
# Main build
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sensual_alvanoto_transcendent_master.wav")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--length_sec", type=float, default=220.0, help="Total length (default 3:40 = 220s)")
    ap.add_argument("--end_clicks_only_sec", type=float, default=10.0)
    args = ap.parse_args()

    sr = args.sr
    dur = float(args.length_sec)
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False, dtype=np.float64)

    # 1) Build bed (pads + counterlines)
    pad = lush_pad(sr, t)
    counter = slow_counterlines(sr, t)
    counter = variable_soft_distort(counter, sr)  # variable soft-distort melodies only
    bed = pad + counter

    # 2) Add clicks + crackle (kept separate for processing rules)
    clicks_st, crackle_st = noto_clicks_and_crackle(sr, t, end_clicks_only_sec=args.end_clicks_only_sec)

    # 3) Xerrox slicing ONLY on bed + crackle (not on clicks)
    bed_plus_crackle = bed + crackle_st
    sliced = gridlocked_spectral_slice(bed_plus_crackle, sr, grid_rate=4.0, win=2048, hop=2048)

    # Re-add untouched clicks
    mix = sliced + clicks_st

    # 4) Ensure last ~10s is clicks only (no bed, no crackle)
    tail_n = int(args.end_clicks_only_sec * sr)
    mix[-tail_n:, :] = clicks_st[-tail_n:, :]

    # 5) Modern spectral mastering:
    #    harmonic spectral freezing ONLY on chords (bed), not on clicks/noise
    #    We'll apply it to the "non-click" portion.
    non_click = mix - clicks_st  # approximate: everything except clicks

    frozen_bed, residual = freeze_harmonic_only(
        non_click, sr,
        smooth_win=12000,
        win=8192,
        hop_mult=2,
        freeze_amount=0.6,
        phase_jitter=0.15,
        freeze_stride=2,
        seed=2
    )

    # Add formant resonators to the frozen harmonic bed
    frozen_bed = add_shifting_formants(frozen_bed, sr, amount=0.015)

    # Climax + release arc (apply to harmonic bed only)
    frozen_bed = apply_climax_release_arc(frozen_bed)

    # Recombine residual (keeps non-harmonic content more intact), then add untouched clicks
    mastered = (frozen_bed + residual) + clicks_st

    # Re-assert last 10s clicks-only after mastering
    mastered[-tail_n:, :] = clicks_st[-tail_n:, :]

    # Final normalize and export
    mastered = normalize(mastered, 0.95)
    sf.write(args.out, mastered, sr, subtype="PCM_24")
    print(f"✅ Wrote: {args.out}  ({dur:.2f}s, {sr}Hz, 24-bit)")


if __name__ == "__main__":
    main()
