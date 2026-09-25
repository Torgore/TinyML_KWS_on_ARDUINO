// kws_inference.ino

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <PDM.h>

#include "kws_model.h"
#include "audio_preprocessing.h"
#include "arduino_output.h"
#include "yes0001.h"

// degub
bool debug_done = false;

const int   N_LABELS = 3;
// const char* LABELS[] = {"yes", "no", "background"};
const float CONFIDENCE_THRESHOLD = 0.60f;

constexpr int TENSOR_ARENA_SIZE = 80 * 1024;
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static const tflite::Model*      tfl_model       = nullptr;
static tflite::MicroInterpreter* tfl_interpreter  = nullptr;
static TfLiteTensor*             tfl_input        = nullptr;
static TfLiteTensor*             tfl_output       = nullptr;

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    Serial.println("=== KWS TinyML Rev2 ===");
    init_leds();

    // Modèle
    tfl_model = tflite::GetModel(kws_model);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERREUR version modèle !");
        while (1);
    }

    // Ops nécessaires seulement
    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter interpreter(
        tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE
    );
    tfl_interpreter = &interpreter;

    if (tfl_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.print("ERREUR AllocateTensors ! RAM dispo : ");
        Serial.println(90088);  // depuis ta compilation
        while (1);
    }

    tfl_input  = tfl_interpreter->input(0);
    tfl_output = tfl_interpreter->output(0);

    Serial.print("✓ Tenseurs alloués — RAM utilisée : ");
    Serial.print(tfl_interpreter->arena_used_bytes() / 1024);
    Serial.println(" KB");
    Serial.print("  Input shape : [");
    for (int i = 0; i < tfl_input->dims->size; i++) {
        Serial.print(tfl_input->dims->data[i]);
        if (i < tfl_input->dims->size-1) Serial.print(", ");
    }
    Serial.println("]");

    init_mel_filterbank();
    init_trig_tables();

    // ← Plus de PDM.begin() ici, capture_audio_blocking() le fait
    Serial.println("✓ Prêt\n\nEn écoute !");
    Serial.println("─────────────────────────────────");
}


void loop() {

    led_listening();

    // Capture 1 seconde d'audio par polling
    bool ok = capture_audio_blocking();


    if (!ok) {
        // Pas de timeout → on re-essaie après une courte pause
        // Souvent causé par une saturation du buffer PDM
        delay(100);
        // Vide le buffer PDM accumulé
        while (PDM.available()) {
            static int16_t tmp[256];
            PDM.read(tmp, min(PDM.available(), (int)sizeof(tmp)));
        }
        return;
    }


    // MFCC
    Serial.print("  Calcul MFCC... ");
    long t0 = millis();
    compute_mfcc(audio_buffer, mfcc_buffer);

    Serial.print(millis() - t0);
    Serial.println(" ms");

    // Remplissage tenseur int8
    float   in_scale = tfl_input->params.scale;
    int32_t in_zp    = tfl_input->params.zero_point;
    int8_t* in_data  = tfl_input->data.int8;

    for (int i = 0; i < N_MFCC * N_FRAMES; i++) {
        float q = mfcc_buffer[i] / in_scale + in_zp;
        q = roundf(q);
        q = fmaxf(-128.0f, fminf(127.0f, q));
        in_data[i] = (int8_t)q;
    }

    // Inférence
    long t1 = millis();
    if (tfl_interpreter->Invoke() != kTfLiteOk) {
        Serial.println("ERREUR Invoke() !");
        return;
    }
    long infer_ms = millis() - t1;

    // Scores
    float   out_scale = tfl_output->params.scale;
    int32_t out_zp    = tfl_output->params.zero_point;
    int8_t* out_data  = tfl_output->data.int8;

    float scores[N_LABELS];
    int   best = 0;
    for (int i = 0; i < N_LABELS; i++) {
        scores[i] = (out_data[i] - out_zp) * out_scale;
        if (scores[i] > scores[best]) best = i;
    }

    // Affichage
    Serial.print("[");
    Serial.print(infer_ms);
    Serial.print("ms] ");
    for (int i = 0; i < N_LABELS; i++) {
        Serial.print(LABELS[i]);
        Serial.print(": ");
        Serial.print(scores[i] * 100, 1);
        Serial.print("%  ");
    }
    Serial.println();

    if (scores[best] >= CONFIDENCE_THRESHOLD) {
        Serial.print(">>> ");
        Serial.print(LABELS[best]);
        Serial.print(" (");
        Serial.print(scores[best] * 100, 0);
        Serial.println("%)");
        if (best == 0) led_oui();
        else           led_non();
    } else {
        Serial.println("    [silence / bruit]");
    }
}
