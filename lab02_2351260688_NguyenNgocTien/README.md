# Lab 02 - Speech Features and Recognition using MFCC + DTW

**Course:** CSE457 - Audio and Speech Processing  
**Student:** Nguyễn Ngọc Tiến  
**Student ID:** 2351260688  

## 1. Overview

This lab implements an isolated-word speech recognition system using
**Mel-Frequency Cepstral Coefficients (MFCC)** and
**Dynamic Time Warping (DTW)**.

The recognition vocabulary contains five English commands:

- `down`
- `go`
- `left`
- `right`
- `up`

The complete pipeline includes:

**Audio → Framing → Energy/ZCR → Endpoint Detection → MFCC → DTW → Nearest-Template Recognition**

DTW is implemented using dynamic programming and backtracking instead
of calling a DTW library directly.

---

## 2. Dataset

The dataset used in the experiment contains **25 WAV files**:

| Word | Number of files |
|---|---:|
| down | 5 |
| go | 5 |
| left | 5 |
| right | 5 |
| up | 5 |
| **Total** | **25** |

Audio configuration:

- Sampling rate: 16 kHz
- Mono audio
- Duration: approximately 1 second per original recording
- 5 command classes
- Mixed recording quality

For each class:

- Files `01-03`: templates/training samples
- Files `04-05`: test samples

Therefore:

- Number of templates: **15**
- Number of test samples: **10**

The selected split uses different speakers between the template and
test sets, making the evaluation closer to a **cross-speaker**
recognition setting.

---

## 3. Speech Processing Pipeline

### 3.1 Short-Time Analysis

The speech signal is divided into short overlapping frames using:

- Frame length: 25 ms
- Hop length: 10 ms
- Hamming window

The following short-time features are calculated:

- Short-Time Energy
- Root Mean Square (RMS)
- Zero-Crossing Rate (ZCR)

These features are used to analyze voiced, unvoiced and silence regions.

### 3.2 Endpoint Detection

Endpoint detection is performed using:

- Relative log-energy threshold
- ZCR refinement
- 50 ms margin around detected speech

The purpose is to remove leading and trailing silence before feature
extraction.

The experiment also compares recognition **with and without endpoint
detection**.

### 3.3 MFCC

The MFCC configuration is:

| Parameter | Value |
|---|---|
| Sampling rate | 16 kHz |
| Frame length | 25 ms |
| Hop length | 10 ms |
| Pre-emphasis | 0.97 |
| FFT size | 512 |
| Mel filters | 24 |
| MFCC coefficients | 13 |
| Normalization | CMN |

The MFCC pipeline is:

**Pre-emphasis → Framing → Hamming → FFT → Power Spectrum → Mel Filterbank → Log → DCT → CMN**

An additional experiment evaluates **MFCC + Delta features**.

---

## 4. Dynamic Time Warping

DTW is implemented manually using dynamic programming.

The local distance between two MFCC frames is Euclidean distance:

```text
d(i,j) = ||x_i - y_j||
```

The accumulated DTW cost is:

```text
D(i,j) = d(i,j) +
         min(
             D(i-1,j),
             D(i,j-1),
             D(i-1,j-1)
         )
```

The final cost is normalized by the optimal path length:

```text
DTW_norm = D(N-1,M-1) / |P|
```

The implementation also performs backtracking to recover the optimal
warping path.

A sanity check gives:

```text
DTW(X, X) = 0
```

---

## 5. Recognition

Recognition uses the **nearest-template** method.

For each test utterance:

1. Extract MFCC features.
2. Compute DTW distance to all 15 templates.
3. Find the minimum DTW distance for each class.
4. Predict the class with the smallest distance.

---

## 6. Experimental Results

### Baseline

The baseline system uses:

**Endpoint Detection + MFCC13 + DTW**

Result:

```text
Accuracy = 50.00% (5/10)
```

Per-class test results:

| Class | Correct |
|---|---:|
| down | 1/2 |
| go | 2/2 |
| left | 0/2 |
| right | 1/2 |
| up | 1/2 |

The `left` class is the most difficult class in the current test set.

### Experiment E1 - Endpoint Detection

| Configuration | Accuracy |
|---|---:|
| Without endpoint detection | **60%** |
| With endpoint detection | **50%** |

For this dataset, the current endpoint detector does not improve
recognition accuracy.

### Experiment E2 - Delta Features

| Features | Accuracy |
|---|---:|
| MFCC 13 | **50%** |
| MFCC 13 + Delta | **40%** |

Adding Delta features does not improve recognition on this small
dataset.

---

## 7. Project Structure

```text
lab02_2351260688_NguyenNgocTien/
│
├── Lab2_2351260688_NguyenNgocTien.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   ├── down/
│   ├── go/
│   ├── left/
│   ├── right/
│   └── up/
│
├── figures/
│   ├── A_waveforms_3_words.png
│   ├── B_energy_zcr_down_01.png
│   ├── B_energy_zcr_go_01.png
│   ├── B_energy_zcr_left_01.png
│   ├── C_endpoint_down_01.png
│   ├── D_mfcc_down_01.png
│   ├── D_mfcc_go_01.png
│   ├── E_dtw_same_word.png
│   ├── E_dtw_different_words.png
│   └── G_confusion_baseline.png
│
├── results/
│   ├── results.csv
│   └── experiments_E1_E2.csv
│
└── report/
    └── Lab2_Report.pdf
```

The `trimmed/` directory is generated automatically during execution
and is therefore excluded from Git.

---

## 8. How to Run

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

Then open:

```text
Lab2_2351260688_NguyenNgocTien.ipynb
```

and run all cells from top to bottom.

The notebook will perform:

```text
Load audio
    ↓
Waveform analysis
    ↓
Energy / RMS / ZCR
    ↓
Endpoint detection
    ↓
MFCC extraction
    ↓
DTW
    ↓
Nearest-template recognition
    ↓
Evaluation
```

---

## 9. Main Results

The experiments show that MFCC + DTW can recognize isolated commands
with a small template set, but performance is sensitive to speaker
variation, recording quality and endpoint detection.

The best configuration in the current experiments is:

```text
No Endpoint Detection + MFCC13 + DTW
Accuracy = 60%
```

The baseline configuration with endpoint detection achieves **50%**.

Because the test set contains only 10 samples, these results should be
interpreted as a small case study rather than a general estimate of
recognition performance.

---

## 10. Report

The complete experimental report is available at:

```text
report/Lab2_Report.pdf
```

It contains waveform analysis, Energy/ZCR, endpoint detection, MFCC
visualization, DTW paths, confusion matrix, controlled experiments and
discussion of the results.