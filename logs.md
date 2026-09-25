## Error 1

Generating splits...: 67%|████████████████████████████      D:\DOCS\Perso\TinyML.venvkws\Lib\site-packages\pydub\utils.py:198: RuntimeWarning: Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work

warn("**Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work**", RuntimeWarning)

tensorflow_datasets unavailable: [WinError 2] The system cannot find the file specified

### solution

* manually download ffmpeg and add it to the path.
* if it still does not work, even if **FFmpeg is installed on Windows**, the **Python environment (venv in VSCode)** may completely ignore it.
* VSCode + venv = snapshot of the PATH at the time the terminal is opened.
* solution: restart VSCode

## Error 2

**Terminal:**

Low memory available, stability problems may occur. Sketch uses 363296 bytes (36%) of program storage space. Maximum is 983040 bytes. Global variables use 235168 bytes (89%) of dynamic memory, leaving 26976 bytes for local variables. Maximum is 262144 bytes.

### solution

Reduce RAM usage (main issue)

The `compute_mfcc()` function allocates `mel_filterbank[40][257]` on the **stack** → **~40 KB at once** → guaranteed stack overflow with 89% RAM usage.

-> use **static memory** and a less memory-intensive FFT

Remove this line from the main sketch:

```cpp
static float audio_float[CLIP_SAMPLES];
```

-> `compute_mfcc` works directly on `audio_buffer` (int16) without an intermediate buffer

Therefore, remember to remove this line from the preprocessing:

```cpp
normalize_audio(audio_buffer, audio_float, CLIP_SAMPLES);
```

and in the `compute_mfcc` call, also replace `audio_float` with `audio_buffer`

(remove the intermediate buffer)

The filterbank must also be initialized in `setup()`.

```cpp
// Just before PDM.begin(...)
init_mel_filterbank(); Serial.println("✓ Mel filterbank initialized");
```

## Error 3

**Serial monitor:**

⚠ Audio timeout — restarting

### possible causes

* Nano 33 BLE → no microphone
* Nano 33 BLE Sense → MP34DT05 microphone
* Nano 33 BLE Sense Rev2 → different microphone (MP34DT06JTR)
  I am in the 3rd case.

**Modify the PDM callback (`onPDMdata`) to:**

```cpp
void onPDMdata() {

int bytesAvailable = PDM.available();  

static int16_t tempBuffer[256];  

// read AND empty the buffer, this is mandatory

PDM.read(tempBuffer, bytesAvailable);  

int n = bytesAvailable / 2;  

if (samples_written + n <= CLIP_SAMPLES) {

memcpy(audio_buffer + samples_written, tempBuffer, bytesAvailable);

samples_written += n;  

if (samples_written >= CLIP_SAMPLES) {

Serial.println("PDM callback");

clip_ready = true;

}

}

}
```

Change

```cpp
void setup() {

Serial.begin(115200);

while (!Serial && millis() < 3000);  // maximum 3s wait
```

To

```cpp
void setup() {

Serial.begin(115200);

while (!Serial && millis() < 6000);  // maximum 6s wait

NOP
```

CHANGE

```cpp
    int bytesAvailable = PDM.available();

    static int16_t tempBuffer[256];

    // read AND empty the buffer, this is mandatory

    PDM.read(tempBuffer, bytesAvailable);
```

TO

```cpp
    static int16_t tempBuffer[256];

    int bytesAvailable = PDM.available();

    if (bytesAvailable > sizeof(tempBuffer)) {

        bytesAvailable = sizeof(tempBuffer);

    }

    // read AND empty the buffer, this is mandatory

    PDM.read(tempBuffer, bytesAvailable);
```

NOP

### Space Issue:

```text
262144 - 177440 - 54324 = 30380 bytes

total RAM - Global Variables - RAM arena

we need to locate (PDM buf, stack, MbedOS buf, serial buf, audio buf, mel filterbank) inside

BUF audio = 31.25 KB

BUF mfcc = 5 KB

ARENA = 53 KB

TOTAL = 89 KB

RESULT: the problem is most likely somewhere else
```

CHANGE

```cpp
static tflite::AllOpsResolver resolver;
```

TO

```cpp
static tflite::MicroMutableOpResolver<8> resolver;
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
resolver.AddMaxPool2D();
resolver.AddMean();
resolver.AddFullyConnected();
resolver.AddSoftmax();
resolver.AddQuantize();
resolver.AddDequantize();
```

On Rev2 you have:

* ARM Cortex-M4
* Mbed OS
* integrated PDM driver + circular buffer
* TensorFlow Micro (blocking for 100–500 ms sometimes)

👉 Therefore, during inference:

* CPU monopolizes the core
* PDM callback is delayed
* PDM buffer becomes full → data is dropped → callback is never triggered

*solution*: use polling instead of the PDM callback

NOP

This is a deterministic bug in the nRF52840 PDM driver. **After the first complete read, the driver goes into a paused state** and waits for a reset signal that we are not sending it.

### solution:

**Stop and restart the PDM after each clip.**

```cpp
bool capture_audio_blocking() {

    // ── Restart PDM cleanly for each clip ─────────

    // On nRF52840 (Rev2), the PDM driver becomes blocked after

    // a complete capture if it is not recycled.

    // PDM.end() + PDM.begin() properly resets the DMA.

    PDM.end();

    delay(50);  // allows the driver to terminate cleanly
```

```cpp
pdm_samples_written = 0;
pdm_recording       = true;

// Re-register the callback (PDM.end() disables it on Rev2)
PDM.onReceive(onPDMdata);
PDM.setGain(20);

if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("  [PDM] ERROR begin()!");
    pdm_recording = false;
    return false;
}

// Stabilization: wait 100 ms and discard the first
// noisy samples before starting the actual capture
delay(100);
pdm_samples_written = 0;   // reset after stabilization

unsigned long t_start = millis();

while (pdm_samples_written < CLIP_SAMPLES) {
    if (millis() - t_start > 2000) {
        pdm_recording = false;
        Serial.print("  [PDM] Timeout — captured: ");
        Serial.print(pdm_samples_written);
        Serial.println(" / 16000");
        return false;
    }
    yield();
}

pdm_recording = false;
return true;
```

```cpp
void setup() {

Serial.begin(115200);

while (!Serial && millis() < 3000);
```

```cpp
Serial.println("=== KWS TinyML Rev2 ===");
init_leds();

// Model + tensors (unchanged)
tfl_model = tflite::GetModel(kws_model);
// ... AllocateTensors etc ...

init_mel_filterbank();
init_trig_tables();

// ← No more PDM.begin() here, capture_audio_blocking() handles it
Serial.println("✓ Ready\n\nListening!");
Serial.println("─────────────────────────────────");
```

## Error 4

Bad results, when saying yes, it still answers no.

### possible causes

Change the `N_FRAMES` value in `audio_preprocessing.h`.

Actually, the model input size is `[1, N_MFCC, N_FRAMES, 1] [1 13 101 1]` instead of `[1 13 98 1]`.

Check the MFCC in Arduino and Python.

Both follow the following path:

```text
audio > Hann window > FFT > power spectrum > mel filter > log() > DCT > MFCC
```

The fricative **"s"** in **"yes"** is in the high frequencies (3–8 kHz). Without Slaney normalization, these filters have a very low amplitude → the model barely "sees" the **"s"** → it confuses **"yes"** with **"no"**.

#### Librosa (Python):

##### frame number

Add 256 padding

--> 1 sec = 16000 samples + padding 256 + 256 (beginning and ending)

--> 98 frames

##### Hann window

`0.5−0.5cos(2πn/n-1​)`

##### FFT

numpy FFT

(same)

##### power spectrum

librosa power spectrum

(same)

##### mel filter

/!\

```python
librosa.filters.mel(sr=16000, n_fft=512, n_mels=40, htk=False, norm='slaney')
```

`norm='slaney'`

Every filter is divided by `(hz_right - hz_left)`.

##### log

`log()`

(same)

##### DCT

`scipy.fftpack.dct(type=2, norm='ortho')`

`norm='ortho'` can change the coefficients

##### Final normalization

/!

Nothing

##### Number of mel filters

/!

`librosa.feature.mfcc(...)` has `n_mels=128` by default

#### Arduino:

##### frame number

no padding

--> offset = frame * HOP_LENGTH;

##### Hann window

```cpp
fft_real[i] = sample * hann_table[i];
```

with

```cpp
hann_table[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (N_FFT - 1)));
```

##### FFT

`fft_inplace(...)`

Classic FFT Cooley-Tukey

(same)

##### power spectrum

```cpp
power = (re²+im²)/N_FFT;
```

(same)

##### mel filter

```cpp
mel_bin_left[m]   = (int)((N_FFT+1) * mel_points[m]   / SAMPLE_RATE);
mel_bin_center[m] = (int)((N_FFT+1) * mel_points[m+1] / SAMPLE_RATE);
mel_bin_right[m]  = (int)((N_FFT+1) * mel_points[m+2] / SAMPLE_RATE);
```

**htk=False**

The Slaney formula is used here. (same)

Mel O'Shaughnessy scale.

/!

**norm='slaney'**

##### log

```cpp
logf(energy + 1e-10f)
```

(same)

##### DCT

```cpp
sum += mel[m] * dct_cos_table[c][m];
```

##### Final normalization

/!

Z-score

```cpp
mfcc = (mfcc-mean)/std
```

##### Number of mel filters

/!

```cpp
#define N_MEL 40
```

| Element               | Librosa                                    | C code                 |
| --------------------- | ------------------------------------------ | ---------------------- |
| FFT                   | Yes                                        | Yes                    |
| Hann                  | Yes                                        | Yes                    |
| Mel                   | Slaney by default                          | Manual triangles       |
| Number of Mel filters | **128 by default**                         | **40**                 |
| DCT                   | Type II with `ortho` normalization         | Manual DCT             |
| Padding (`center`)    | Yes by default                             | No                     |
| Number of frames      | ≈98 (with `center=False` for 1 s, 512/160) | 101 (manually defined) |
| Z-score               | No                                         | Yes                    |

### solution

##### MFCC normalization

Modify the Mel filter in `compute_mfcc()` and `init_mel_filterbank()` so that it applies Slaney normalization.

In `init_mel_filterbank()`:

```cpp
// Also store the width of each filter for normalization

static float mel_filter_norm[N_MEL];  // ← new

// Evenly spaced Mel points (N_MEL + 2 to have the boundaries)

float hz_points[N_MEL + 2];  // ← new: keep the Hz values for normalization

for (int m = 0; m < N_MEL; m++) {

    // FFT bins (unchanged)

    mel_bin_left[m]   = (int)((N_FFT+1) * mel_points[m] / SAMPLE_RATE);

    mel_bin_center[m] = (int)((N_FFT+1) * mel_points[m+1] / SAMPLE_RATE);

    mel_bin_right[m]  = (int)((N_FFT+1) * mel_points[m+2] / SAMPLE_RATE);

    // ── Slaney normalization ───────────────────────────────────

    // librosa divides by (hz_right - hz_left) in real Hz.

    // This ensures that the area of each triangular filter = 1.0,

    // regardless of its frequency width.

    float hz_width = hz_points[m+2] - hz_points[m];  // total width in Hz

    if (hz_width > 0.0f) {

        mel_filter_norm[m] = 2.0f / hz_width;  // "2/width" factor = Slaney normalization

    } else {

        mel_filter_norm[m] = 1.0f;

    }

}
```

```cpp
Serial.println("✓ Mel filterbank (Slaney norm) initialized");
```

and in `compute_mfcc()` Mel filter section:

```cpp
// Mel filters

for (int m = 0; m < N_MEL; m++) {

    float energy = 0.0f;

    int l = mel_bin_left[m];

    int c = mel_bin_center[m];

    int r = mel_bin_right[m];

    // Triangle rising edge (left edge → center)

    if (c > l) {

        for (int k = l; k <= c; k++) {

            float weight = (float)(k - l) / (c - l);

            energy += weight * power_buf[k];

        }

    }

    // Triangle falling edge (center → right edge)

    if (r > c) {

        for (int k = c; k <= r; k++) {

            float weight = (float)(r - k) / (r - c);

            energy += weight * power_buf[k];

        }

    }

    // ── Apply Slaney normalization ────────────────────────

    // Same as librosa norm='slaney': multiply by 2/width_Hz

    energy *= mel_filter_norm[m];

    mel_buf[m] = logf(energy + 1e-10f);

}
```

Now we test:

### PYTHON:

```python
# Load a test file
audio, sr = librosa.load("./speech_commands_subset/yes/0000.wav", sr=16000)
audio = librosa.util.fix_length(audio, size=16000)

mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

print(f"mfcc[0, 0] = {mfcc[0, 0]:.3f}")  # should match mfcc_buffer[0]
print(f"mfcc[1, 0] = {mfcc[1, 0]:.3f}")  # should match mfcc_buffer[N_FRAMES]
print(f"mfcc[2, 0] = {mfcc[2, 0]:.3f}")  # should match mfcc_buffer[2*N_FRAMES]
```

**result:**

```text
mfcc[0, 0] = -3.822
mfcc[1, 0] = 0.689
mfcc[2, 0] = 0.289
```

### ARDUINO:

```arduino
// MFCC
Serial.print(" Calculating MFCC... ");
long t0 = millis();
compute_mfcc(audio_buffer, mfcc_buffer);

// Debug: display the first 5 raw Mel filterbank values
// (before log and DCT) to compare with Python
// Remove once verified

Serial.print(" mel[0..4] = ");
// These values correspond to mel_buf[] BEFORE the log
// To see them, compute_mfcc() must be temporarily instrumented
// Easier: display the final mfcc_buffer

Serial.print(" mfcc[0]= "); Serial.print(mfcc_buffer[0], 3);
Serial.print(" mfcc[1]= "); Serial.print(mfcc_buffer[N_FRAMES], 3);
Serial.print(" mfcc[2]= "); Serial.println(mfcc_buffer[2*N_FRAMES], 3);
```

**result:**

```text
Calculating MFCC...

mel[0..4] =

mfcc[0]= -3.491
mfcc[1]= 0.390
mfcc[2]= 0.369
```

##### n_mel

### PYTHON:

```python
# ── MFCC calculation ───────────────────────────────────
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=40
)
```

Now Python matches Arduino.

##### DCT

###### before

```cpp
// DCT with precomputed table (no more cosf() here)

for (int c = 0; c < N_MFCC; c++) {

    float sum = 0.0f;

    for (int m = 0; m < N_MEL; m++)

        sum += mel_buf[m] * dct_cos_table[c][m];

    out_mfcc[c * N_FRAMES + frame] = sum;

}
```

###### after

```cpp
for (int c = 0; c < N_MFCC; c++)
{

    float alpha;

    if (c == 0)

        alpha = sqrtf(1.0f / N_MEL);

    else

        alpha = sqrtf(2.0f / N_MEL);

    for (int m = 0; m < N_MEL; m++)

    {

        dct_cos_table[c][m] =

        alpha

        *

        cosf(

            M_PI*

            (m + 0.5f) *

            c /

            N_MEL

        );

    }

}
```

##### MFCC padding

```python
# ── MFCC calculation ───────────────────────────────────
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=40,
    center=False
)  # shape: (N_MFCC, N_FRAMES) = (13, 97)
```

We add `center=False` to remove the padding.

before

```text
mfcc.shape = (13, 101)
```

after

```text
mfcc.shape = (13, 97)
```

## Error 5

`mfcc[0]`: Python -3.822 vs Arduino -3.491 → difference 0.331 (9%)

`mfcc[1]`: Python 0.689 vs Arduino 0.390 → difference 0.299 (43%) ← problematic

`mfcc[2]`: Python 0.289 vs Arduino 0.369 → difference 0.080 (28%)

Arduino code:

```cpp
// arduino_output.h

// RGB LED management and serial feedback

// The Nano 33 BLE Sense has 3 individual LEDs: LEDR, LEDG, LEDB

#ifndef ARDUINO_OUTPUT_H
#define ARDUINO_OUTPUT_H

void init_leds() {

pinMode(LEDR, OUTPUT);

pinMode(LEDG, OUTPUT);

pinMode(LEDB, OUTPUT);

pinMode(LED_BUILTIN, OUTPUT);

// RGB LEDs are active LOW on the Nano 33 BLE Sense

digitalWrite(LEDR, HIGH);

digitalWrite(LEDG, HIGH);

digitalWrite(LEDB, HIGH);

}

// Red = "no"

void led_no() {

digitalWrite(LEDR, LOW);

digitalWrite(LEDG, HIGH);

digitalWrite(LEDB, HIGH);

delay(400);

digitalWrite(LEDR, HIGH);

}

// Green = "yes"

void led_yes() {

digitalWrite(LEDR, HIGH);

digitalWrite(LEDG, LOW);

digitalWrite(LEDB, HIGH);

delay(400);

}

// Blinking blue = listening

void led_listening() {

digitalWrite(LEDB, LOW);

delay(50);

digitalWrite(LEDB, HIGH);

}

void print_scores(float* scores, int n_labels,

const char** labels, long inference_ms) {

Serial.print("[");

Serial.print(inference_ms);

Serial.print("ms] ");

for (int i = 0; i < n_labels; i++) {

Serial.print(labels[i]);

Serial.print(": ");

Serial.print(scores[i] * 100, 1);

Serial.print("% ");

}

Serial.println();

}

#endif // ARDUINO_OUTPUT_H
```

On Python:

I modified:

* the padding
* the number of frames
* the Mel filter
* the Z-score

so that the Arduino and Python code are consistent.

After deployment on Arduino I get:

```text
14:16:15.958 -> [silence / noise]

14:16:17.128 -> Calculating MFCC... mel[0..4] = mfcc[0]= -3.344 mfcc[1]= 0.508 mfcc[2]= 0.385

14:16:17.476 -> 316 ms

14:16:17.662 -> [183ms] yes: 22.3% no: 76.2% background: 1.6%

14:16:17.662 -> >>> no (76%)

14:16:19.238 -> Calculating MFCC... mel[0..4] = mfcc[0]= -3.436 mfcc[1]= 0.399 mfcc[2]= 0.326

14:16:19.591 -> 316 ms

14:16:19.776 -> [183ms] yes: 23.8% no: 71.9% background: 3.9%

14:16:19.776 -> [silence / noise]

14:16:20.963 -> Calculating MFCC... mel[0..4] = mfcc[0]= -3.467 mfcc[1]= 0.373 mfcc[2]= 0.342

14:16:21.305 -> 316 ms

14:16:21.490 -> [184ms] yes: 33.2% no: 28.9% background: 37.9%

14:16:21.490 -> [silence / noise]

nothing changes when I say yes or no...
```

Where could the error be?

### PROBLEM

Difference between Python MFCC and Arduino MFCC

### SOLUTION

Replace the Python MFCC function with one that is closer to the Arduino implementation, which differs from the default Librosa implementation:

```python
def extract_mfcc(file_path: str) -> np.ndarray:
    """
    Load a .wav file, normalize its duration to 1s,
    then calculate and normalize the MFCCs.

    Returns an array of shape (N_MFCC, N_FRAMES, 1)
    ready for a CNN (height × width × channels).
    """

    # 1. Load with automatic resampling if necessary

    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # ── Duration normalization ──────────────────────────

    # All clips must be exactly 1s long.

    # A clip that is too short is padded with zeros (silence).

    # A clip that is too long is truncated.

    target_length = int(SAMPLE_RATE * CLIP_DURATION)

    if len(audio) < target_length:

        audio = np.pad(audio, (0, target_length - len(audio)))

    else:

        audio = audio[:target_length]

    # ── MFCC calculation ───────────────────────────────────

    # 1. STFT spectrogram (Centered = False)

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window='hann',
        center=False
    )  # shape: (N_MFCC, N_FRAMES) = (13, 97)

    # 2. Power spectrogram (Power = 2, divided by N_FFT)

    power_spectrogram = (np.abs(stft) ** 2) / N_FFT

    # 3. Mel filterbank with Slaney normalization

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=N_FFT,
        n_mels=40,
        norm='slaney',
        htk=False
    )

    mel_spectrogram = np.dot(mel_basis, power_spectrogram)

    # 4. Natural log conversion (keeping epsilon 1e-10)

    log_mel = np.log(mel_spectrogram + 1e-10)

    # 5. Raw DCT-II (without orthogonal normalization)

    import scipy.fftpack

    mfcc = scipy.fftpack.dct(log_mel, type=2, axis=0, norm=None)[:N_MFCC]

    # 6. Global Z-Score

    mean = np.mean(mfcc)

    std = np.std(mfcc) + 1e-8

    mfcc_normalized = (mfcc - mean) / std

    return mfcc_normalized
```

Conclusion: **8% error in the MFCC calculations between Python and Arduino for the same audio signal.**

### PROBLEM

Pasted image 20260925145357.png

### SOLUTION
