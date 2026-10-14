"""
infer_vaaman.py

Real-time footstep detection for the Vicharak Vaaman (or any Linux/ARM SBC)
using ONNX Runtime only -- no PyTorch required on the board.

This mirrors infer_live.py's pipeline exactly, but:
  - loads model.onnx + model_config.json (produced by export_onnx.py)
    instead of a .pt checkpoint
  - runs inference with onnxruntime.InferenceSession instead of torch
  - keeps the same robust mic-handling (probe real samplerate/channels/dtype
    combinations) that fixed "external mic doesn't work" on the PC side

Setup on the Vaaman (once):
    python3 -m venv venv
    source venv/bin/activate
    pip install onnxruntime librosa numpy sounddevice soundfile

Files needed on the board (copied from your PC via scp):
    model.onnx
    model_config.json
    infer_vaaman.py           (this file)
    preprocessing/audio.py     (pure numpy/librosa, no torch -- safe to reuse)
    preprocessing/features.py  (same)

Usage:
    python3 infer_vaaman.py --list_devices
    python3 infer_vaaman.py --onnx model.onnx --config model_config.json --device_name "USB"
"""

import argparse
import json
import queue
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import librosa
import onnxruntime as ort

from preprocessing.audio import preprocess_buffer
from preprocessing.features import compute_log_mel_spectrogram, compute_low_freq_ratio

try:
    import sounddevice as sd
except OSError:
    print("ERROR: could not load PortAudio / sounddevice. Install with:\n"
          "  sudo apt install portaudio19-dev\n  pip install sounddevice")
    raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, default="model.onnx")
    p.add_argument("--config", type=str, default="model_config.json")
    p.add_argument("--infer_interval_sec", type=float, default=0.3)
    p.add_argument("--prob_threshold", type=float, default=0.80)
    p.add_argument("--min_low_freq_ratio", type=float, default=0.35,
                   help="inference-time safety gate: reject as footstep if "
                        "low-frequency energy ratio is below this, even if "
                        "the model is confident (rejects speech/claps)")
    p.add_argument("--smoothing_window", type=int, default=3)
    p.add_argument("--smoothing_min_hits", type=int, default=2)
    p.add_argument("--device_index", type=int, default=None)
    p.add_argument("--device_name", type=str, default=None)
    p.add_argument("--input_block_sec", type=float, default=0.05)
    p.add_argument("--list_devices", action="store_true")
    return p.parse_args()


class RingBuffer:
    def __init__(self, size_samples: int):
        self.size = size_samples
        self.buf = np.zeros(size_samples, dtype=np.float32)

    def push(self, chunk: np.ndarray):
        n = len(chunk)
        if n >= self.size:
            self.buf[:] = chunk[-self.size:]
            return
        self.buf = np.roll(self.buf, -n)
        self.buf[-n:] = chunk

    def snapshot(self) -> np.ndarray:
        return self.buf.copy()


def resolve_device_index(device_index, device_name):
    if device_name is None:
        return device_index
    matches = []
    for idx, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0 and device_name.lower() in info["name"].lower():
            matches.append((idx, info["name"]))
    if not matches:
        print(f"ERROR: no input device matched name '{device_name}'")
        print(sd.query_devices())
        sys.exit(1)
    if len(matches) > 1:
        print(f"  (multiple matches for '{device_name}', using first: {[m[1] for m in matches]})")
    print(f"Matched device_name '{device_name}' -> index {matches[0][0]} ({matches[0][1]})")
    return matches[0][0]


def find_working_stream_config(device_index, target_sr: int):
    """Probe real (samplerate, channels, dtype) combos the device actually
    supports via sd.check_input_settings(), instead of hardcoding
    channels=1/dtype=float32 -- this is what made external mics work on PC
    and is just as important on SBC USB audio interfaces/HATs."""
    try:
        info = sd.query_devices(device_index, kind="input")
        native_sr = int(info["default_samplerate"])
        max_channels = max(1, int(info["max_input_channels"]))
    except Exception:
        native_sr = target_sr
        max_channels = 1

    candidate_rates = list(dict.fromkeys([native_sr, target_sr, 44100, 48000, 16000]))
    candidate_channels = list(dict.fromkeys([1, max_channels]))
    candidate_dtypes = ["float32", "int16"]

    for sr_try in candidate_rates:
        for ch in candidate_channels:
            for dtype in candidate_dtypes:
                try:
                    sd.check_input_settings(device=device_index, samplerate=sr_try,
                                             channels=ch, dtype=dtype)
                    return sr_try, ch, dtype
                except Exception:
                    continue
    return None


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print(f"Loading ONNX model from {args.onnx} ...")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_names = [i.name for i in sess.get_inputs()]
    output_name = sess.get_outputs()[0].name

    with open(args.config) as f:
        cfg = json.load(f)

    sr = cfg["sample_rate"]
    duration_sec = cfg["duration_sec"]
    mel_params = cfg["mel_params"]
    class_names = cfg["class_names"]
    use_aux = cfg.get("use_aux_features", False)
    print(f"Loaded. sample_rate={sr}, duration_sec={duration_sec}, "
          f"mel_params={mel_params}, use_aux_features={use_aux}")
    print(f"ONNX inputs: {input_names}")

    device_index = resolve_device_index(args.device_index, args.device_name)

    found = find_working_stream_config(device_index, sr)
    if found is None:
        print("ERROR: could not find a working (samplerate, channels, dtype) "
              "combination for this device.")
        print(sd.query_devices())
        return
    capture_sr, capture_channels, capture_dtype = found
    print(f"Opening stream: samplerate={capture_sr}, channels={capture_channels}, "
          f"dtype={capture_dtype}")

    needs_resample = (capture_sr != sr)

    ring = RingBuffer(size_samples=int(duration_sec * sr))
    recent_predictions = deque(maxlen=args.smoothing_window)
    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[mic warning] {status}", file=sys.stderr)
        chunk = indata[:, 0].copy()
        if capture_dtype == "int16":
            chunk = chunk.astype(np.float32) / 32768.0
        else:
            chunk = chunk.astype(np.float32)
        if needs_resample:
            chunk = librosa.resample(chunk, orig_sr=capture_sr, target_sr=sr)
        audio_q.put(chunk)

    block_size = int(args.input_block_sec * capture_sr)

    stream = sd.InputStream(
        samplerate=capture_sr,
        channels=capture_channels,
        dtype=capture_dtype,
        blocksize=block_size,
        device=device_index,
        callback=audio_callback,
    )

    print("\nListening for footsteps... (Ctrl+C to stop)\n")
    last_infer_time = time.time()

    try:
        with stream:
            while True:
                try:
                    while True:
                        chunk = audio_q.get_nowait()
                        ring.push(chunk)
                except queue.Empty:
                    pass

                now = time.time()
                if now - last_infer_time >= args.infer_interval_sec:
                    last_infer_time = now

                    raw = ring.snapshot()
                    processed = preprocess_buffer(raw, sr=sr, duration_sec=duration_sec)
                    log_mel = compute_log_mel_spectrogram(
                        processed,
                        sample_rate=mel_params["sample_rate"],
                        n_fft=mel_params["n_fft"],
                        hop_length=mel_params["hop_length"],
                        win_length=mel_params["win_length"],
                        n_mels=mel_params["n_mels"],
                    )
                    low_freq_ratio = compute_low_freq_ratio(processed, sample_rate=sr)

                    x = log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,n_mels,n_frames)
                    feed = {input_name: x}
                    if use_aux and len(input_names) > 1:
                        aux_name = input_names[1]
                        feed[aux_name] = np.array([[low_freq_ratio]], dtype=np.float32)

                    logits = sess.run([output_name], feed)[0][0]
                    exp = np.exp(logits - np.max(logits))
                    probs = exp / exp.sum()
                    footstep_prob = float(probs[1])

                    passes_freq_gate = low_freq_ratio >= args.min_low_freq_ratio
                    single_hit = (footstep_prob >= args.prob_threshold) and passes_freq_gate
                    recent_predictions.append(single_hit)
                    hits = sum(recent_predictions)

                    is_footstep = (len(recent_predictions) == args.smoothing_window and
                                   hits >= args.smoothing_min_hits)

                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    label = "FOOTSTEP DETECTED" if is_footstep else "NO FOOTSTEP"
                    conf = footstep_prob if is_footstep else (1 - footstep_prob)
                    print(f"[{ts}] {label} | confidence={conf*100:.1f}% | "
                          f"low_freq_ratio={low_freq_ratio:.2f}")

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
