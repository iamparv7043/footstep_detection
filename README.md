# Real-Time Footstep Detection (Edge-Deployable)

A CNN-based binary audio classifier that detects human footsteps from
live microphone audio in real time, built with edge deployment in mind
(Raspberry Pi, Jetson, or other ONNX Runtime-compatible boards).

This is a heavier, more accurate successor to an earlier lightweight
prototype -- deeper CNN, an auxiliary low-frequency energy feature to
reject speech/claps/broadband false positives, robust external
microphone support, full training diagnostics, and ONNX export for
deployment off of PyTorch entirely.

## Features

- **Log-Mel spectrogram + CNN** classification pipeline (footstep vs. no_footstep)
- **4-block CNN (~430K params)** with BatchNorm, Dropout2d, Global Average Pooling
- **Low-frequency energy ratio** auxiliary feature -- footsteps are low-freq
  thumps; speech, claps, and most confusable sounds are broadband/high-freq.
  Used both as a model input and as an explicit safety gate at inference time
- **File-level train/val/test split** to prevent data leakage between
  overlapping audio segments from the same recording
- **Class-weighted loss + SpecAugment** for imbalance and regularization
- **Robust microphone handling** -- probes real (sample rate, channel count,
  dtype) combinations against the device instead of guessing once, so
  external USB mics/interfaces work, not just a laptop's built-in mic
- **Temporal smoothing** (majority vote over recent predictions) to prevent
  flickering between FOOTSTEP DETECTED / NO FOOTSTEP
- **Full training report** -- loss/accuracy curves, parameter breakdown,
  GPU info, inference timing, all saved automatically after training
- **ONNX export** with an accompanying deployment config (sample rate, mel
  params, expected input shape) so a target device never has to guess
  preprocessing details

## Project structure

```
footstep_detection/
├── dataset/
│   ├── footstep/           # your footstep WAV files go here
│   └── no_footstep/        # background/speech/claps/etc WAV files go here
├── preprocessing/
│   ├── audio.py             # load, mono, resample, normalize, fixed-length windowing
│   └── features.py          # Log-Mel spectrogram, low-freq ratio, SpecAugment
├── models/
│   └── cnn.py                # 4-block CNN architecture
├── dataset.py                # PyTorch Dataset, anti-leakage split, augmentation
├── train.py                  # training loop + full diagnostic report
├── evaluate.py                # test-set metrics + confusion matrix
├── infer_live.py               # live microphone inference
├── export_onnx.py              # export trained model to ONNX for edge deployment
├── checkpoints/                 # trained model, plots, and reports land here
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Linux only, for microphone support:
```bash
sudo apt install portaudio19-dev
```

## Dataset

Put WAV files (any sample rate, mono or stereo, any length) into:
```
dataset/footstep/*.wav
dataset/no_footstep/*.wav
```
`no_footstep` should include not just background noise but confusable
sounds -- speech, claps, coughing, laughing, etc. -- since these are the
most common source of false positives in real deployments.

## Train

```bash
python train.py --dataset_root dataset --epochs 30 --num_workers 6
```
Set `--num_workers` to roughly your CPU core count -- audio preprocessing
is CPU-bound, and parallelizing it across workers keeps the GPU fed
instead of idling between batches.

Key flags:
```
--batch_size     (default 32)
--lr             (default 1e-3)
--dropout        (default 0.4)
--patience       (default 8, early stopping)
--num_workers    (default 4)
```

After training, `checkpoints/` will contain:
- `best_model.pt` -- the trained model (best validation F1, self-contained
  with preprocessing params)
- `loss_curve.png`, `accuracy_curve.png` -- training diagnostics
- `training_report.txt` -- epochs trained, GPU used, parameter breakdown,
  loss function, dropout, batch size, inference timing, final metrics

## Evaluate

```bash
python evaluate.py --dataset_root dataset --checkpoint checkpoints/best_model.pt
```
Reports accuracy, precision, recall, F1, confusion matrix, and per-sample
inference time on the held-out test set.

## Live inference

```bash
python infer_live.py --checkpoint checkpoints/best_model.pt --list_devices
python infer_live.py --checkpoint checkpoints/best_model.pt --device_name "USB"
```
Output:
```
[12:41:03.120] NO FOOTSTEP | confidence=91.1% | low_freq_ratio=0.12
[12:41:03.520] FOOTSTEP DETECTED | confidence=94.2% | low_freq_ratio=0.61
```
Tunable flags: `--prob_threshold`, `--min_low_freq_ratio`,
`--smoothing_window`, `--smoothing_min_hits`, `--device_index` /
`--device_name`.

## Export for edge deployment

```bash
python export_onnx.py --checkpoint checkpoints/best_model.pt --output checkpoints/model.onnx
```
Produces `model.onnx` (loadable with `onnxruntime` alone, no PyTorch
required on the target device) and `model_config.json` (every
preprocessing parameter the target device needs). The script verifies
PyTorch and ONNX Runtime outputs match before reporting success.

ONNX Runtime runs on Raspberry Pi, Jetson, and most Linux/ARM SBCs
without modification. For microcontroller-class targets (ESP32 etc.),
further work is needed: the Log-Mel preprocessing itself would need a
fixed-point FFT rewrite, and the model would likely need INT8
quantization, since neither `librosa` nor a float32 CNN this size runs
on bare-metal microcontrollers.

## Design notes

- **Anti-leakage split**: splitting happens at the source-file level
  first, then all sliding-window segments from a file inherit that
  file's split -- no chunk from the same recording appears in two splits.
- **Identical preprocessing everywhere**: training, evaluation, and live
  inference all call the same `preprocessing/audio.py` and
  `preprocessing/features.py` functions, and live inference reads every
  parameter from the checkpoint itself, so they can't drift out of sync.
- **Low-frequency gate**: applied both as a learned input feature and as
  an explicit rule at inference time (`--min_low_freq_ratio`), so even if
  the CNN is fooled, a genuinely broadband sound (speech, clap) won't be
  reported as a footstep.
