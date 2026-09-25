## What Is the Tensor Arena Actually Used For?

The `tensor_arena` is **not**:

* the audio buffer
* the stack
* global variables
* PDM driver buffers
* serial buffers
* the MFCC buffer

It is used exclusively by TensorFlow Lite Micro to store:

* intermediate tensors between layers
* activations
* temporary convolution buffers
* internal TFLM runtime structures

```text
Input ↓ Conv2D output ↓ MaxPool output ↓ Depthwise output ↓ ... ↓ Softmax output
```

All of these intermediate results live in the Tensor Arena.

## What Is the Mel Filter Actually Used For?

The Mel filter transforms a power spectrum (FFT output) into a representation that mimics human hearing.

Human hearing works in a logarithmic way. We can easily perceive the difference between 100 MHz and 200 MHz, but the difference between 8100 MHz and 8200 MHz is much harder to perceive.

To reproduce this effect, we apply a Mel filter (**triangular filter**) to the output of our FFT.

### Why Is It Useful?

"yes" and "no" are mainly distinguished by their formants, the vocal resonances between 200 Hz and 3000 Hz. Without a Mel filter, the CNN would process the 257 FFT bins uniformly, including around ~200 bins in the high-frequency range that contain almost no speech information. The Mel filter compresses all of this into 40 perceptually relevant values.

### Slaney Normalization

Normalizes each triangular filter **according to its width in Hz**. Without this normalization, wider filters (high frequencies) would have much larger values than narrower filters (low frequencies), completely biasing the MFCCs.

## What Is the DCT?

The DCT (Discrete Cosine Transform) is a mathematical transformation that expresses a signal as a sum of cosines at different frequencies. It is the final step in the MFCC calculation: it takes the 40 log-Mel energies and compresses them into 13 coefficients.

The FFT decomposes a time-domain signal into frequencies.

The DCT decomposes the Mel energies into "spatial frequencies".
