# LAB 04 – Text-to-Speech Synthesis

Course: CSE457 – Audio and Speech Processing

## Overview

This project implements a Vietnamese mini Text-to-Speech system
for queue announcement sentences.

It covers three TTS approaches:
- Concatenative TTS
- HMM-based parameter generation
- Pretrained Neural TTS (VieNeu-TTS)

## Experiments

| Part | Description |
|------|-------------|
| A | Text normalization and speech inventory |
| B | Baseline waveform concatenation |
| C | Unit selection and cross-fade smoothing |
| D | Prosody, F0, speed and pitch modification |
| E | HMM-based acoustic parameter generation |
| F | Offline pretrained Neural TTS inference |
| G | Evaluation and comparison |

## Dataset

16 Vietnamese word units, with 2 recordings per word.

Audio format: mono WAV, 16 kHz, PCM 16-bit.

See `README_Lab04_Dataset.md` for dataset preparation.

## Execution

Run notebooks in order from A to G using VS Code or Jupyter.

Install the dependencies for each notebook before execution.

Part F requires a separately installed VieNeu-TTS model and
its local pretrained weights. Model weights are not included.

## Results

- Concatenative TTS waveform generation
- Dynamic programming unit selection
- Cross-fade smoothing
- Prosody experiments
- HMM acoustic parameter trajectories
- Offline Neural TTS inference
- Comparative evaluation

## Report

See `Bao_cao_Lab04_Tong_hop_TTS.pdf`.

## Limitations

The HMM demo produces acoustic parameter trajectories but
does not include a waveform vocoder.

The listening-test MOS is not available unless actual listener
ratings are collected. Automated audio quality checks must
not be represented as human MOS.
