// arduino_output.h
// Gestion des LEDs RGB et du retour série
// Le Nano 33 BLE Sense a 3 LEDs individuelles : LEDR, LEDG, LEDB

#ifndef ARDUINO_OUTPUT_H
#define ARDUINO_OUTPUT_H

void init_leds() {
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    pinMode(LED_BUILTIN, OUTPUT);
    // LEDs RGB actives à LOW sur le Nano 33 BLE Sense
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
}

// Rouge = "non"
void led_non() {
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    delay(400);
    digitalWrite(LEDR, HIGH);
}

// Vert = "oui"
void led_oui() {
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, HIGH);
    delay(400);
    digitalWrite(LEDG, HIGH);
}

// Bleu clignotant = en écoute
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
        Serial.print("%  ");
    }
    Serial.println();
}

#endif // ARDUINO_OUTPUT_H