"""Generate the 5-command Vietnamese synthetic Lab 3 dataset using eSpeak.

Requires eSpeak with Vietnamese voices (vi, vi-hue, vi-sgn), numpy, scipy,
and soundfile. These are SYNTHESIZED utterances, not human recordings.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

TARGET_SR = 16000
RNG_SEED = 20261019
WORDS = {'mo':'mở', 'dong':'đóng', 'len':'lên', 'xuong':'xuống', 'dung':'dừng'}
# Each entry invokes eSpeak from the beginning, with a unique configuration.
# 6 train + 2 test per word. All instances share ONE synthesizer family.
CONFIGS = [
    {'voice':'vi',     'speed':112,'pitch':46,'amplitude':155,'split':'train'},
    {'voice':'vi',     'speed':137,'pitch':63,'amplitude':175,'split':'train'},
    {'voice':'vi-hue', 'speed':120,'pitch':53,'amplitude':165,'split':'train'},
    {'voice':'vi-hue', 'speed':151,'pitch':67,'amplitude':185,'split':'train'},
    {'voice':'vi-sgn', 'speed':117,'pitch':48,'amplitude':170,'split':'train'},
    {'voice':'vi-sgn', 'speed':143,'pitch':59,'amplitude':190,'split':'train'},
    {'voice':'vi',     'speed':125,'pitch':55,'amplitude':180,'split':'test'},
    {'voice':'vi-sgn', 'speed':132,'pitch':54,'amplitude':160,'split':'test'},
]


def synthesize(word: str, cfg: dict, output_file: Path, rng: np.random.Generator) -> dict:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'raw.wav'
        cmd = ['espeak', '-v', cfg['voice'], '-s', str(cfg['speed']), '-p', str(cfg['pitch']),
               '-a', str(cfg['amplitude']), '-w', str(p), word]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        y, sr = sf.read(p, always_2d=False, dtype='float32')
    if np.ndim(y) != 1 or y.size == 0:
        raise ValueError('eSpeak returned empty / non-mono audio')
    # Speech-sensitive trim, removing eSpeak's often lengthy built-in tail.
    # Keep the entirety of the nontrivial voiced sound plus a 35-ms guard.
    active = np.flatnonzero(np.abs(y) > max(0.003, np.max(np.abs(y)) * 0.012))
    if len(active) < sr * 0.06:
        raise ValueError(f'Suspect short/empty synthesis: {word!r} {cfg}')
    i0 = max(0, int(active[0]) - int(0.035*sr))
    i1 = min(len(y), int(active[-1]) + int(0.045*sr))
    y = y[i0:i1]
    if sr != TARGET_SR:
        from math import gcd
        d = gcd(sr, TARGET_SR)
        y = resample_poly(y, TARGET_SR//d, sr//d).astype('float32')
    # Normalize after synthesis to avoid clipping. A small per-utterance
    # intensity change is applied; not treated as a distinct human recording.
    peak = float(np.max(np.abs(y)))
    if not peak:
        raise ValueError(f'Silent synthesis for {word!r}')
    target_peak = float(rng.uniform(0.65, 0.84))
    y = np.clip((y / peak) * target_peak, -1, 1)
    fade_n = min(int(0.008*TARGET_SR), len(y)//4)
    y[:fade_n] *= np.linspace(0, 1, fade_n, dtype='float32')
    y[-fade_n:] *= np.linspace(1, 0, fade_n, dtype='float32')
    # Exactly silent segments, 0.2-0.5 s each, so endpoint detection is testable.
    lead_s = float(rng.uniform(0.22, 0.40))
    tail_s = float(rng.uniform(0.23, 0.42))
    all_y = np.concatenate([np.zeros(round(lead_s*TARGET_SR), dtype='float32'),
                            y.astype('float32'),
                            np.zeros(round(tail_s*TARGET_SR), dtype='float32')])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_file, all_y, TARGET_SR, subtype='PCM_16')
    sha = hashlib.sha256(output_file.read_bytes()).hexdigest()
    return dict(filename=output_file.name, duration_seconds=round(len(all_y)/TARGET_SR, 4),
                silence_start_seconds=round(lead_s, 3), silence_end_seconds=round(tail_s, 3),
                speech_seconds=round(len(y)/TARGET_SR, 3), sha256=sha)


def create(out_dir: Path):
    if shutil.which('espeak') is None:
        raise RuntimeError('Please install eSpeak with Vietnamese voice support first.')
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for label, word in WORDS.items():
        for idx, cfg in enumerate(CONFIGS, 1):
            split = cfg['split']
            path = out_dir / 'dataset' / split / label / f'{label}_{idx:02d}.wav'
            if path.exists():
                raise FileExistsError(f'File exists: {path}. Generate in an empty directory.')
            details = synthesize(word, cfg, path, rng)
            rows.append({
                'relative_path':path.relative_to(out_dir).as_posix(),
                'split':split, 'label':label, 'text':word,
                'origin':'eSpeak offline synthetic Vietnamese', **cfg, **details,
            })
    manifest = out_dir / 'metadata.csv'
    with manifest.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)
    readme = '''# Bộ dữ liệu Lab 3: 5 lệnh tiếng Việt (TTS tổng hợp)

## Cấu trúc
- 5 lớp: `mo` = mở; `dong` = đóng; `len` = lên; `xuong` = xuống; `dung` = dừng.
- 8 file/lớp: 6 train và 2 test. Tổng: 40 file.
- Định dạng: WAV mono, 16.000 Hz, PCM 16-bit, mỗi file một từ.
- Có 0,2-0,5 giây khoảng lặng ở đầu và cuối mỗi tệp.
- File `metadata.csv` chứa nhãn, câu nói, train/test, tham số giọng tổng hợp,
  thời lượng và SHA256. Không có file trùng byte ở train/test.

## Cảnh báo về chất lượng và đánh giá
Đây KHÔNG PHẢI 40 bản ghi âm độc lập của người thật. Tất cả đều được tổng
hợp ngoại tuyến bằng eSpeak (voice vi / vi-hue / vi-sgn), có thay đổi tốc độ,
cao độ và âm lượng. Tiếng máy tương đối cơ học; phát âm/dấu thanh có thể
khác giọng người thật. Các âm thanh có chung engine nên train/test không
độc lập theo người nói và không chứng minh khả năng tổng quát hóa ASR.

Phù hợp để chạy thử quy trình Lab 3, kiểm tra MFCC, Baum-Welch, Viterbi,
confusion matrix, DTW. KHÔNG nên báo cáo accuracy này như chất lượng trên
người nói thật. Với yêu cầu robustness 'người nói mới', hãy thu âm người thật
riêng để có thử nghiệm xác thực. Nên nghe thử 5-10 file trước khi dùng.

## Dùng trên VS Code
1. Giải nén toàn bộ. Sao chép thư mục `dataset/` vào cùng vị trí với `Lab03.ipynb`.
2. Kiểm tra trong notebook: `LABELS = ['mo', 'dong', 'len', 'xuong', 'dung']`
   và `DATASET_ROOT = Path('dataset')` (hoặc đường dẫn tương ứng).
3. Chạy lần lượt các cell. Từng nhãn có 6 tệp trong train và 2 tệp trong test.
4. Dữ liệu này chỉ có tiếng một từ, không cung cấp lexicon phone chuẩn hoặc
   bigram corpus của giảng viên; cần chuẩn bị riêng theo Lab 3.

## Nguồn và tái tạo
- Engine: eSpeak 1.48.15, giọng vi / vi-hue / vi-sgn.
- Không có âm thanh nào lấy từ người nói thật hoặc dataset công khai.
- Mã nguồn sinh file: `create_lab03_tts_dataset.py` (nếu cung cấp kèm).
- Dùng `python create_lab03_tts_dataset.py <thư-mục-trống>` trên máy có eSpeak,
  numpy, scipy và soundfile để tái tạo (theo cùng cấu hình).
'''
    (out_dir / 'README.txt').write_text(readme, encoding='utf-8')
    print('Created',len(rows),'files in',out_dir)
    for label in WORDS:
        d = [r for r in rows if r['label'] == label]
        print(f"{label:6s} {d[0]['text']:6s}: train={sum(x['split']=='train' for x in d)} "
              f"test={sum(x['split']=='test' for x in d)}, "
              f"duration={min(x['duration_seconds'] for x in d):.2f}-{max(x['duration_seconds'] for x in d):.2f}s")
    return rows

if __name__ == '__main__':
    import sys
    create(Path(sys.argv[1] if len(sys.argv)>1 else 'lab03_vietnamese_tts'))
