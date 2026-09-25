// audio_preprocessing.h — Rev2 FINAL v3
#ifndef AUDIO_PREPROCESSING_H
#define AUDIO_PREPROCESSING_H

#include <Arduino.h>
#include <PDM.h>
#include <math.h>

#define SAMPLE_RATE   16000
#define CLIP_SAMPLES  16000
#define N_MFCC        13
#define N_FRAMES      97
#define N_FFT         512
#define HOP_LENGTH    160
#define N_MEL         40

// ── Buffers statiques ─────────────────────────────────────────
static int16_t audio_buffer[CLIP_SAMPLES];
static float   mfcc_buffer[N_MFCC * N_FRAMES];

static int mel_bin_left[N_MEL];
static int mel_bin_center[N_MEL];
static int mel_bin_right[N_MEL];

// Buffers de travail MFCC
static float fft_real[N_FFT];
static float fft_imag[N_FFT];
static float power_buf[N_FFT / 2 + 1];
static float mel_buf[N_MEL];

// ── État capture ──────────────────────────────────────────────
volatile int  pdm_samples_written = 0;
volatile bool pdm_recording       = false;

// ── Callback PDM ──────────────────────────────────────────────
// Règle clé : on ne touche JAMAIS au PDM en dehors du callback.
// Quand pdm_recording=false, on lit quand même pour vider
// le buffer DMA — sinon il déborde et le driver se bloque.
// Mais on utilise un buffer statique local, jamais de boucle while.
void onPDMdata() {
    static int16_t discard_buf[512];  // statique = pas sur la stack

    if (!pdm_recording) {
        // Vide proprement sans boucle while (une seule lecture)
        int n = PDM.available();
        if (n > 0) {
            int bytes = min(n, (int)sizeof(discard_buf));
            PDM.read(discard_buf, bytes);
        }
        return;
    }

    // Enregistrement actif
    int n         = PDM.available();
    int remaining = (CLIP_SAMPLES - pdm_samples_written) * 2;
    int to_read   = min(n, remaining);

    if (to_read > 0) {
        PDM.read(
            (char*)(audio_buffer + pdm_samples_written),
            to_read
        );
        pdm_samples_written += to_read / 2;
    }
}

// ── Capture audio ─────────────────────────────────────────────
bool capture_audio_blocking() {
    // ── Redémarre le PDM proprement à chaque clip ─────────
    // Sur nRF52840 (Rev2), le driver PDM se bloque après
    // une capture complète si on ne le recycle pas.
    // PDM.end() + PDM.begin() remet le DMA à zéro proprement.
    PDM.end();
    delay(50);  // laisse le driver se terminer proprement

    pdm_samples_written = 0;
    pdm_recording       = true;

    // Réenregistre le callback (PDM.end() le désactive sur Rev2)
    PDM.onReceive(onPDMdata);
    PDM.setGain(20);

    if (!PDM.begin(1, SAMPLE_RATE)) {
        Serial.println("  [PDM] ERREUR begin() !");
        pdm_recording = false;
        return false;
    }

    // Stabilisation : attend 100ms et vide les premiers samples
    // parasites avant de commencer la vraie capture
    delay(100);
    pdm_samples_written = 0;   // remet à zéro après stabilisation

    unsigned long t_start = millis();

    while (pdm_samples_written < CLIP_SAMPLES) {
        if (millis() - t_start > 2000) {
            pdm_recording = false;
            Serial.print("  [PDM] Timeout — capturé : ");
            Serial.print(pdm_samples_written);
            Serial.println(" / 16000");
            return false;
        }
        yield();
    }

    pdm_recording = false;
    return true;
}

// ── Init filterbank mel ───────────────────────────────────────

// audio_preprocessing.h — filterbank IDENTIQUE à librosa htk=False norm='slaney'

// Stocke aussi la largeur de chaque filtre pour la normalisation
static float mel_filter_norm[N_MEL];  // ← nouveau

void init_mel_filterbank() {
    float mel_low  = 2595.0f * log10f(1.0f + 0.0f / 700.0f);
    float mel_high = 2595.0f * log10f(1.0f + (SAMPLE_RATE / 2.0f) / 700.0f);

    // Points mel régulièrement espacés (N_MEL + 2 pour avoir les bords)
    float mel_points[N_MEL + 2];
    float hz_points[N_MEL + 2];   // ← nouveau : on garde les Hz pour la norme

    for (int i = 0; i < N_MEL + 2; i++) {
        float mel = mel_low + i * (mel_high - mel_low) / (N_MEL + 1);
        mel_points[i] = mel;
        // Conversion mel → Hz (inverse de la formule Slaney)
        hz_points[i]  = 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }
    for (int m = 0; m < N_MEL; m++) {
        // Bins FFT (inchangé)
        mel_bin_left[m]   = (int)((N_FFT+1) * mel_points[m]   / SAMPLE_RATE);
        mel_bin_center[m] = (int)((N_FFT+1) * mel_points[m+1] / SAMPLE_RATE);
        mel_bin_right[m]  = (int)((N_FFT+1) * mel_points[m+2] / SAMPLE_RATE);
        // ── Normalisation Slaney ───────────────────────────────────
        // librosa divise par (hz_right - hz_left) en Hz réels.
        // Cela garantit que l'aire de chaque filtre triangulaire = 1.0,
        // quel que soit sa largeur en fréquence.
        float hz_width = hz_points[m+2] - hz_points[m];   // largeur totale en Hz
        if (hz_width > 0.0f) {
            mel_filter_norm[m] = 2.0f / hz_width;  // facteur "2/width" = norme Slaney
        } else {
            mel_filter_norm[m] = 1.0f;
        }
    }
    Serial.println("✓ Filterbank mel (Slaney norm) initialisé");
}

// ── Tables cosinus précalculées pour la DCT ───────────────────
// Précalculer les cosinus évite ~N_MFCC × N_MEL × N_FRAMES
// appels à cosf() pendant l'inférence → passe de ~2000ms à ~200ms
static float dct_cos_table[N_MFCC][N_MEL];
static float hann_table[N_FFT];

void init_trig_tables() {
    // Table DCT
    for (int c = 0; c < N_MFCC; c++)
    {
        float scale;

        if (c == 0)
            scale = sqrtf(1.0f / N_MEL);
        else
            scale = sqrtf(2.0f / N_MEL);

        for (int m = 0; m < N_MEL; m++)
        {
            dct_cos_table[c][m] =
                scale *
                cosf(M_PI * c * (m + 0.5f) / N_MEL);
        }
    }

    // Table fenêtre de Hann
    for (int i = 0; i < N_FFT; i++)
        hann_table[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (N_FFT - 1)));

    Serial.println("✓ Tables trig précalculées");
}

// ── FFT Cooley-Tukey ─────────────────────────────────────────
void fft_inplace(float* re, float* im, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        float wR = cosf(ang), wI = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cR = 1.0f, cI = 0.0f;
            for (int k = 0; k < len/2; k++) {
                float uR = re[i+k],           uI = im[i+k];
                float vR = re[i+k+len/2]*cR - im[i+k+len/2]*cI;
                float vI = re[i+k+len/2]*cI + im[i+k+len/2]*cR;
                re[i+k]       = uR+vR; im[i+k]       = uI+vI;
                re[i+k+len/2] = uR-vR; im[i+k+len/2] = uI-vI;
                float tmp = cR*wR - cI*wI;
                cI = cR*wI + cI*wR; cR = tmp;
            }
        }
    }
}

// ── Calcul MFCC (version optimisée avec tables) ───────────────
void compute_mfcc(const int16_t* audio, float* out_mfcc) {
    for (int frame = 0; frame < N_FRAMES; frame++) {
        int offset = frame * HOP_LENGTH;

        // Fenêtrage avec table précalculée
        for (int i = 0; i < N_FFT; i++) {
            int idx = offset + i;
            float s = (idx < CLIP_SAMPLES) ? audio[idx] / 32768.0f : 0.0f;
            fft_real[i] = s * hann_table[i];
            fft_imag[i] = 0.0f;
        }

        fft_inplace(fft_real, fft_imag, N_FFT);

        for (int i = 0; i <= N_FFT/2; i++) {
            power_buf[i] = (fft_real[i]*fft_real[i] +
                            fft_imag[i]*fft_imag[i]) / N_FFT;
        }

        // Filtres mel
        for (int m = 0; m < N_MEL; m++) {
            float energy = 0.0f;
            int l = mel_bin_left[m];
            int c = mel_bin_center[m];
            int r = mel_bin_right[m];

            // Montée du triangle (bord gauche → centre)
            if (c > l) {
                for (int k = l; k <= c; k++) {
                    float weight = (float)(k - l) / (c - l);
                    energy += weight * power_buf[k];
                }
            }
            // Descente du triangle (centre → bord droit)
            if (r > c) {
                for (int k = c; k <= r; k++) {
                    float weight = (float)(r - k) / (r - c);
                    energy += weight * power_buf[k];
                }
            }
            // ── Application de la norme Slaney ────────────────────────
            // Identique à librosa norm='slaney' : multiplie par 2/width_Hz
            energy *= mel_filter_norm[m];

            mel_buf[m] = logf(energy + 1e-10f);
        }

        // DCT avec table précalculée (plus de cosf() ici)
        for (int c = 0; c < N_MFCC; c++) {
            float sum = 0.0f;
            for (int m = 0; m < N_MEL; m++)
                sum += mel_buf[m] * dct_cos_table[c][m];
            out_mfcc[c * N_FRAMES + frame] = sum;
        }
    }

    // Z-score
    int total = N_MFCC * N_FRAMES;
    float mean = 0.0f;
    for (int i = 0; i < total; i++) mean += out_mfcc[i];
    mean /= total;
    float var = 0.0f;
    for (int i = 0; i < total; i++) {
        float d = out_mfcc[i] - mean;
        var += d * d;
    }
    float std = sqrtf(var / total) + 1e-8f;
    for (int i = 0; i < total; i++)
        out_mfcc[i] = (out_mfcc[i] - mean) / std;
}

#endif