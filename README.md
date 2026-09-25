# TinyML_KWS_on_ARDUINO

Embedded AI, TinyML Keyword Spotting (KWS) project for **Arduino Nano 33 BLE Sense Lite**

# 🎙️ TinyML KWS — Keyword Spotting on Arduino Nano 33 BLE Sense

> Deployment of a **Keyword Spotting** model at the edge on an Arduino microcontroller,
> capable of recognizing **"yes"** and **"no"** in real time, with no network connection and no server.

---

## 🎯 Objective

This project is a hands-on workshop for learning **TinyML** from end to end.
It implements a complete pipeline, from audio data collection to deployment on an Arduino Nano 33 BLE Sense Rev2:

* Training a lightweight CNN with TensorFlow
* int8 quantization and TFLite conversion
* Embedded inference with TFLite Micro
* Real-time visual feedback through the RGB LED

---

## 🛠️ Required Hardware

| Component      | Details                            |
| -------------- | ---------------------------------- |
| **Board**      | Arduino Nano 33 BLE Sense **Rev2** |
| **Microphone** | Integrated MP34DT06JTR (PDM)       |
| **LED**        | Built-in RGB (LEDR / LEDG / LEDB)  |

---

## 🧠 Pipeline Architecture

```text
Raw audio (PDM)
      ↓
  Hann windowing (periodic)
      ↓
  512-point FFT
      ↓
  Mel filterbank (40 filters, Slaney normalization)
      ↓
  Log + DCT-II → 13 MFCC coefficients
      ↓
  Z-score normalization
      ↓
  Lightweight CNN (DS-CNN)  ←── trained on Google Speech Commands
      ↓
  int8 quantization (TFLite Micro)
      ↓
  Decision: yes / no / background
      ↓
  RGB LED (green / red)
```

---

## 📁 Project Structure

```text
kws-tinyml/
│
├── .venvkws/
│   └── kws_training.py          # Complete Python pipeline (data → model → export)
│
├── training/
│   └── kws_model.h              # model that will be used by arduino code
│   └── kws_model.keras          # model keras (optionnal) 
│   └── kws_model.tflite         # model tflite (optionnal) 
│   └── confusion_matrix.png     # confusion matrix of my model training 
│   └── mfcc_visualization.png   # to have an idea of what a 1s audio looks like in 2D (with mfcc values)
│   └── training_history.png     # train + validation graph
│
├── arduino/
│   ├── kws_inference.ino        # Main Arduino sketch
│   ├── audio_preprocessing.h    # On-device MFCC (FFT, Mel filterbank, DCT)
│   ├── arduino_output.h         # RGB LED management
│   └── kws_model.h              # Exported TFLite model as a C array (generated)
│
└── logs.md                      # history of the problems encountered and how to solve them  
└── notions.txt                  # some notions important to understand how the training works 
└── procedure.txt                # big picture of the procedure to realize this project
└── README.md
```

---

## ⚙️ Python Pipeline — `kws_training.py`

### Dependencies

```bash
pip install tensorflow tensorflow-datasets librosa numpy matplotlib scikit-learn tqdm soundfile
```

### Execution

```bash
python kws_training.py
```

The script automatically runs through 7 steps:

| Step | Description                                                                |
| ---- | -------------------------------------------------------------------------- |
| 1    | Download the Google Speech Commands dataset via `tensorflow_datasets`      |
| 2    | Extract MFCC features (13 coefficients, `center=False`, Slaney filterbank) |
| 3    | Build the lightweight CNN (Conv2D + DepthwiseConv2D + GAP)                 |
| 4    | Train with EarlyStopping, ReduceLROnPlateau, and class weights             |
| 5    | Evaluate the model and generate a confusion matrix                         |
| 6    | Post-training int8 quantization (TFLite)                                   |
| 7    | Export the model as a C array → `kws_model.h`                              |

### Recognized Classes

| Index | Label        | LED      |
| ----- | ------------ | -------- |
| 0     | `yes`        | 🟢 Green |
| 1     | `no`         | 🔴 Red   |

---

## 🔧 Arduino Deployment

### Required Libraries

In the Arduino IDE (Tools → Manage Libraries):

* `Harvard_TinyMLx` — TFLite Micro + supported operators
* Board package: **Arduino Mbed OS Nano Boards**

### Configuration

```cpp
// In audio_preprocessing.h — must exactly match the Python pipeline
#define SAMPLE_RATE   16000
#define CLIP_SAMPLES  16000   // 1 second
#define N_MFCC        13
#define N_FRAMES      97      // center=False
#define N_FFT         512
#define HOP_LENGTH    160
#define N_MEL         40
```

### Flashing

```text
1. Copy kws_model.h into the sketch folder
2. Tools → Board → Arduino Mbed OS Nano Boards → Arduino Nano 33 BLE
3. Sketch → Upload
4. Open the Serial Monitor at 115200 baud
```

### Serial Monitor Output

```text
=== KWS TinyML Rev2 ===
✓ Tensors allocated — RAM used: 53 KB
  Input shape: [1, 13, 97, 1]
✓ Mel filterbank (Slaney norm) initialized
✓ Precomputed trigonometric tables (periodic Hann)
✓ PDM ready

Listening!
─────────────────────────────────
  Computing MFCC... 316 ms
[183ms] yes: 91.2%  no: 5.1%  background: 3.7%
>>> yes (91%)
```

---

## 🔑 Key Lessons Learned

### 1. Python ↔ Arduino Preprocessing Parity

This is the main challenge in TinyML audio applications. Any mismatch between the training pipeline and the embedded preprocessing can degrade production accuracy.

| Parameter      | Python (librosa)    | Arduino (implemented)           |
| -------------- | ------------------- | ------------------------------- |
| Hann window    | Periodic (`N_FFT`)  | `cos(2π·i / N_FFT)`             |
| Mel filterbank | `norm='slaney'`     | `2.0 / hz_width`                |
| DCT            | `type=2, norm=None` | `Σ mel[m] · cos(π·c·(m+0.5)/N)` |
| N_FRAMES       | `center=False` → 97 | `#define N_FRAMES 97`           |

### 2. PDM Quirk on the Nano 33 BLE Sense Rev2

The MP34DT06JTR chip (Rev2) requires a **callback** to unlock the internal DMA buffer. Polling alone does not work. In addition, the PDM driver becomes blocked after each complete clip and requires a `PDM.end()` / `PDM.begin()` cycle between captures.

```cpp
// Reliable pattern for Rev2
PDM.onReceive(onPDMdata);   // unlocks the DMA
PDM.end();                  // recycle the driver between each clip
delay(50);
PDM.begin(1, SAMPLE_RATE);
```

### 3. RAM Constraints

The Nano 33 BLE Sense has **256 KB of RAM**. The main pitfalls are:

* `mel_filterbank[N_MEL][N_FFT/2+1]` allocated on the stack → **guaranteed stack overflow** → use static global arrays
* `AllOpsResolver` → prefer `MicroMutableOpResolver<8>` to save approximately 25% of RAM
* Target: **< 70% dynamic RAM usage** for stability

### 4. The "background" Class Is Essential

Without a silence/background-noise class, the model is forced to choose between "yes" and "no" continuously, resulting in a high false-positive rate. The `background` class generated from `_background_noise_` in the Speech Commands dataset solves this issue.

---

## 📊 Performance

| Metric                | Value         |
| --------------------- | ------------- |
| int8 TFLite model     | ~15 KB        |
| Tensor arena RAM      | ~53 KB        |
| Inference latency     | ~183 ms       |
| MFCC computation time | ~316 ms       |
| Total latency         | ~1.5 s / clip |
| Dynamic RAM           | ~67%          |

---

## 🗺️ Potential Improvements

* [ ] Train on French "yes" / "no" data using custom recordings
* [ ] Replace the custom FFT with `arm_rfft_fast_f32()` (CMSIS-DSP) → 5× faster MFCC computation
* [ ] Implement a continuous sliding window for detection without a fixed 1-second delay
* [ ] Add more keywords (home automation commands, etc.)
* [ ] Experiment with a GRU or an MLP on flattened MFCC features

---

## 📚 Resources

* [Google Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
* [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
* [Harvard TinyMLx Library](https://github.com/tinyMLx/arduino-library)
* [Pete Warden — Speech Commands Paper](https://arxiv.org/abs/1804.03209)
* [DS-CNN Architecture (Keyword Spotting)](https://arxiv.org/abs/1711.07128)

---

## 📄 License

MIT — free to use, modify, and distribute.
