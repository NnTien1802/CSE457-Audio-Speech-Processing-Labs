"""Lab 4F – run *installed* VieNeu v3 Turbo SDK/weights in strict offline mode.

No SDK installation, network request, automatic model download or fabricated WAV.
Do not treat a successful SDK import as confirmation weights are installed.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

# Set before importing SDK, huggingface_hub, transformers.
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_var] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

DIGITS = dict(zip("0123456789", ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]))
VOCAB = {"mời", "khách", "hàng", "số", "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "đến", "quầy"}


def normalize_queue_text(text):
    parts = re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)
    tokens = []
    for word in parts:
        if word.isdecimal():
            if not all(d in DIGITS for d in word):
                raise ValueError(f"Số không hỗ trợ: {word}")
            tokens.extend(DIGITS[d] for d in word)
        else:
            tokens.append(word)
    if not tokens or set(tokens) - VOCAB:
        raise ValueError(f"Token trống/ngoài inventory: {set(tokens) - VOCAB}")
    return tokens, " ".join(tokens).capitalize() + "."


def read_tests(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 3:
        raise ValueError("Lab 4F cần đúng ba câu kiểm thử")
    rows, seen = [], set()
    for item in data:
        test_id = str(item["id"])
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", test_id) or test_id in seen:
            raise ValueError(f"ID trùng/không hợp lệ: {test_id}")
        seen.add(test_id)
        tokens, spoken = normalize_queue_text(item["text"])
        rows.append(dict(id=test_id, original_text=item["text"], spoken_text=spoken, tokens=tokens))
    return rows


def run(args):
    rows = read_tests(Path(args.input))
    print("SDK local executable:", sys.executable, flush=True)
    print("Offline flags:", {k: os.environ[k] for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}, flush=True)
    for row in rows:
        print(f"{row['id']}: {row['spoken_text']}")
    if args.dry_run:
        print("DRY RUN OK – không tải model, không tạo WAV")
        return

    import numpy as np
    import soundfile as sf
    from vieneu import Vieneu

    try:
        sdk_version = importlib.metadata.version("vieneu")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "unknown"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Khởi tạo model CÓ SẴN: mode=v3turbo, backend={args.backend}", flush=True)
    start = time.perf_counter()
    try:
        if args.backend == "auto":
            tts = Vieneu(mode="v3turbo")
        else:
            tts = Vieneu(mode="v3turbo", backend=args.backend)
    except Exception as exc:
        raise RuntimeError("Không load được weights cục bộ. KHÔNG tự tải model. Hãy kiểm tra model cache/đường dẫn và backend CPU/GPU đã cài từ trước.") from exc
    load_seconds = time.perf_counter() - start
    metrics = []
    for i, row in enumerate(rows, 1):
        print(f"Generating {i}/3 {row['id']}...", flush=True)
        t0 = time.perf_counter()
        if args.voice:
            audio = tts.infer(row["spoken_text"], voice=args.voice)
        else:
            audio = tts.infer(row["spoken_text"])
        t1 = time.perf_counter()
        wav_path = output / f"neural_{row['id']}.wav"
        tts.save(audio, str(wav_path))
        t2 = time.perf_counter()
        x, sr = sf.read(wav_path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr <= 0 or len(x) < 1600 or not np.isfinite(x).all():
            raise RuntimeError(f"Invalid output audio: {wav_path}")
        rms = float(np.sqrt(np.mean(x.astype("float64") ** 2)))
        if rms < 1e-5:
            raise RuntimeError(f"Silence output: {wav_path}")
        dur = len(x) / sr
        metrics.append({
            **row, "sdk": "vieneu", "sdk_version": sdk_version,
            "model": "VieNeu-TTS-v3-Turbo", "backend": args.backend,
            "voice": args.voice or "SDK default", "model_load_s": round(load_seconds, 5),
            "sample_rate": sr, "audio_duration_s": round(dur, 5),
            "latency_s": round(t1 - t0, 5),
            "input_to_audio_save_s": round(t2 - t0, 5),
            "rtf": round((t1 - t0) / dur, 5),
            "rms": rms, "native_audio_path": str(wav_path),
            "offline": True, "is_mock": False,
            "python_executable": sys.executable,
            "os": platform.platform(),
        })
        print(f"Saved {wav_path} | {dur:.2f}s audio | infer {t1-t0:.2f}s | RTF {(t1-t0)/dur:.3f}", flush=True)
    eval_dir = output.parent / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / "f_neural_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("3 WAV generated locally, no download option exposed", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="lab04_f_tests.json")
    p.add_argument("--output", default="audio_output")
    p.add_argument("--backend", default="auto", choices=("auto", "onnx", "torch"), help="Select existing local backend only")
    p.add_argument("--voice", default="", help="Empty = built-in SDK default preset")
    p.add_argument("--dry-run", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
