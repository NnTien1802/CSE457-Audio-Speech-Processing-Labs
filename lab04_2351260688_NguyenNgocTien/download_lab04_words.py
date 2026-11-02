#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tải các file WAV từ đơn tiếng Việt từ lephuongbg/words (GitHub).
Nguồn: https://github.com/lephuongbg/words (repo đánh dấu GPL-3.0).
Không tạo dữ liệu giả hoặc coi các tệp chưa nghe kiểm chứng là đạt chất lượng.

Sử dụng:
    python -m pip install requests soundfile numpy scipy
    python download_lab04_words.py
    python download_lab04_words.py --candidates 3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

SOURCE_REPO = "lephuongbg/words"
API_BASE = f"https://api.github.com/repos/{SOURCE_REPO}"
SOURCE_HOME = f"https://github.com/{SOURCE_REPO}"
WORDS = {
    "mời": "moi", "khách": "khach", "hàng": "hang", "số": "so",
    "không": "khong", "một": "mot", "hai": "hai", "ba": "ba",
    "bốn": "bon", "năm": "nam", "sáu": "sau", "bảy": "bay",
    "tám": "tam", "chín": "chin", "đến": "den", "quầy": "quay"
}
KNOWN_SPEAKERS = ["lephuong", "longtran", "linhdo"]
SR = 16000
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Lab04-VietnameseTTS-Educational/1.0", "Accept": "application/vnd.github+json"})


def get(url: str) -> bytes:
    for attempt in range(4):
        try:
            res = SESSION.get(url, timeout=(15, 55))
            if res.status_code in (403, 429):
                detail = res.text[:160]
                raise RuntimeError(f"GitHub API giới hạn truy cập: HTTP {res.status_code}. {detail}")
            res.raise_for_status()
            return res.content
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Tải thất bại")


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip().lower()


def ascii_slug(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def candidate_score(path: str, word: str, slug: str):
    """Chỉ chọn file có từ đơn tương ứng tường minh trong tên hoặc thư mục.
    Không ánh xạ file đánh số nếu không biết bảng chỉ mục (tránh gán sai nhãn).
    """
    p = Path(path)
    if p.suffix.lower() != ".wav":
        return None
    pieces = [norm(x) for x in p.parts]
    stem = norm(p.stem)
    keys = {norm(word), slug, ascii_slug(word)}
    if stem in keys:
        quality = 0
    elif any(comp in keys for comp in pieces[:-1]):
        quality = 1
    else:
        # ghép tiền tố/ hậu tố speaker / take, ví dụ khach_lephuong.wav
        segment = re.split(r"[_\-\.\s]+", stem)
        if any(c in keys for c in segment):
            quality = 2
        else:
            return None
    speaker = next((s for s in KNOWN_SPEAKERS if s in pieces or s in re.split(r"[_\-\.\s]+", stem)), "unknown")
    # ưu tiên đúng dấu, một unit nguyên vẹn, có speaker rõ
    accuracy = 0 if word in pieces or stem == norm(word) else 1
    return (quality, accuracy, 0 if speaker != "unknown" else 1), speaker


def main() -> int:
    ap = argparse.ArgumentParser(description="Tải inventory WAV Lab04 từ dữ liệu từ đơn tiếng Việt GitHub")
    ap.add_argument("--candidates", type=int, default=2, choices=[2, 3])
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--inspect", action="store_true", help="Chỉ kiểm tra cây file nguồn, không tải âm thanh")
    args = ap.parse_args()
    raw_dir = args.root / "lab4_data" / "recordings_raw"
    data_dir = args.root / "lab4_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Đang đọc danh sách tệp từ {SOURCE_HOME}")
    try:
        repo = json.loads(get(API_BASE))
        branch = repo.get("default_branch", "master")
        tree_data = json.loads(get(f"{API_BASE}/git/trees/{quote(branch)}?recursive=1"))
    except Exception as exc:
        print("Không lấy được GitHub tree:", exc, file=sys.stderr)
        return 2
    all_paths = [x["path"] for x in tree_data.get("tree", []) if x.get("type") == "blob"]
    wavs = [p for p in all_paths if p.lower().endswith(".wav")]
    print(f"Nguồn có {len(wavs)} đường dẫn WAV mà API trả về; truncated={tree_data.get('truncated',False)}")
    print("Ví dụ đường dẫn:", *wavs[:8], sep="\n  ")
    if tree_data.get("truncated"):
        print("CẢNH BÁO: danh sách GitHub không đầy đủ; nên kiểm tra repo thủ công.")

    selected = {}
    candidates = defaultdict(list)
    for label, slug in WORDS.items():
        for path in wavs:
            result = candidate_score(path, label, slug)
            if result is not None:
                score, speaker = result
                candidates[label].append((score, path, speaker))
        candidates[label].sort(key=lambda x: (x[0], KNOWN_SPEAKERS.index(x[2]) if x[2] in KNOWN_SPEAKERS else 10, x[1]))
        chosen, used_speakers = [], set()
        for item in candidates[label]:
            if item[2] not in used_speakers:
                chosen.append(item)
                used_speakers.add(item[2])
                if len(chosen) == args.candidates:
                    break
        if len(chosen) < args.candidates:
            chosen.extend(x for x in candidates[label] if x not in chosen)  # chỉ các file KHÁC nhau
            chosen = chosen[:args.candidates]
        selected[label] = chosen
        print(f"{label:>7}: tìm {len(candidates[label]):2d} đường dẫn, chọn {len(chosen)} => {[x[2] for x in chosen]}")

    if args.inspect:
        print("\nĐây chỉ là kiểm tra danh sách nguồn. Chạy lại KHÔNG có --inspect để tải.")
        return 0
    if not any(selected.values()):
        print("\nKhông nhận diện được tên file bằng nhãn từ. Repo có thể dùng mã số thay tên từ.")
        print("Hãy kiểm tra ví dụ đường dẫn trên. Không tự đoán ánh xạ file số sang từ vì có thể sai nội dung.")
        return 3

    print("[2/3] Đang tải và chuẩn hóa về WAV 16 kHz/mono/PCM16...")
    records = []
    failures = []
    for label, slug in WORDS.items():
        for take, (_, path, speaker) in enumerate(selected[label], 1):
            url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{quote(branch)}/{quote(path, safe='/')}"
            filename = f"{slug}_{take:02d}.wav"
            dst = raw_dir / filename
            try:
                blob = get(url)
                if blob[:4] == b"version ":
                    raise ValueError("Tệp Git LFS pointer; cần tải từ LFS hoặc clone repository")
                audio, src_sr = sf.read(io.BytesIO(blob), always_2d=True, dtype="float32")
                if src_sr < 8000 or not np.isfinite(audio).all():
                    raise ValueError("WAV nguồn không hợp lệ")
                mono = np.mean(audio, axis=1)
                if src_sr != SR:
                    k = gcd(SR, int(src_sr))
                    mono = resample_poly(mono, SR // k, int(src_sr) // k)
                if len(mono) < int(0.1 * SR) or np.max(np.abs(mono)) < 0.001:
                    raise ValueError("Âm quá ngắn hoặc gần im lặng")
                peak = float(np.max(np.abs(mono)))
                mono = np.asarray(mono, np.float32)
                if peak > 1.0:
                    mono /= peak
                sf.write(dst, mono, SR, subtype="PCM_16")
                rec = {"label": label, "slug": slug, "speaker": speaker, "take": take,
                       "filename": filename, "source_path": path, "source_url": url,
                       "source_sha256": hashlib.sha256(blob).hexdigest(),
                       "source_sr": int(src_sr), "source_channels": int(audio.shape[1]),
                       "duration_s": round(len(mono) / SR, 4), "source_type": "recorded",
                       "source_repository": SOURCE_HOME, "license_repo": "GPL-3.0"}
                records.append(rec)
                print("  OK ", filename, "<", path)
            except Exception as exc:
                failures.append((label, path, str(exc)))
                print("  LỖI", filename, ":", exc)
    fields = ["label", "slug", "speaker", "take", "filename", "source_path", "source_url",
              "source_sha256", "source_sr", "source_channels", "duration_s", "source_type",
              "source_repository", "license_repo"]
    with (data_dir / "source_provenance.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with (data_dir / "download_report.txt").open("w", encoding="utf-8") as fp:
        fp.write(f"Source: {SOURCE_HOME}\nDownloaded: {len(records)}\nRequired: {len(WORDS)*args.candidates}\n")
        fp.write("Failures/missing:\n")
        for label, path, err in failures:
            fp.write(f"{label}: {path}: {err}\n")
        for label, picked in selected.items():
            if len(picked) < args.candidates:
                fp.write(f"MISSING: {label}: chỉ tìm {len(picked)}/{args.candidates} file\n")
    print(f"[3/3] Đã ghi {len(records)}/{16 * args.candidates} file; thông tin nguồn: lab4_data/source_provenance.csv")
    print("Kiểm tra đúng dấu tiếng Việt bằng cách NGHE từng WAV trước khi dùng.")
    print("Tiếp theo: chạy lại notebook Lab04_Buoc1_A_Dataset.ipynb từ đầu đến cuối.")
    return 0 if len(records) >= 16 * args.candidates else 1


if __name__ == "__main__":
    sys.exit(main())
