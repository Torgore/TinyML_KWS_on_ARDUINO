# ============================================================
# WORKSHOP TINYML — KWS (Key Word Spotting) sur Arduino
# Nano 33 BLE Sense (Arduino TinyML Kit)
#
# Objectif : entraîner un modèle capable de reconnaître
# "oui" et "non" à l'edge, sans connexion réseau,
# directement sur le microcontrôleur.
#
# Auteur   : Cours TinyML
# Prérequis: pip install tensorflow librosa numpy matplotlib
#             scikit-learn tqdm
# ============================================================


# ─────────────────────────────────────────────────────────────
# ÉTAPE 0 — IMPORTS & CONFIGURATION GLOBALE
#
# Pourquoi cette étape ?
#   On centralise toutes les constantes ici. En TinyML, les
#   hyperparamètres audio (sample rate, durée, n_mfcc) doivent
#   être IDENTIQUES à l'entraînement et sur l'Arduino. Un seul
#   endroit pour les modifier évite les bugs silencieux.
# ─────────────────────────────────────────────────────────────
import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import urllib.request
import tarfile
import struct
import tensorflow_datasets as tfds



# ── Paramètres audio ──────────────────────────────────────────
SAMPLE_RATE   = 16000   # Hz — standard pour la parole (LSM9DS1 sur Arduino)
CLIP_DURATION = 1.0     # secondes — Google Speech Commands = clips de 1s
N_MFCC        = 13      # coefficients MFCC (bon compromis info/taille)
N_FFT         = 512     # fenêtre FFT (~32ms à 16kHz)
HOP_LENGTH    = 160     # décalage entre fenêtres (~10ms)
N_FRAMES      = 98      # frames temporelles pour 1s (≈ 1000ms / 10ms)

# ── Classes cibles ────────────────────────────────────────────
# "yes" / "no" en anglais (Speech Commands dataset)
# Adapte en "oui"/"non" si tu enregistres tes propres données.
LABELS     = ["yes", "no", "background"]
NUM_LABELS = len(LABELS)  # 3

# ── Chemins ───────────────────────────────────────────────────
DATA_DIR      = "./speech_commands_subset"
MODEL_PATH    = "./kws_model.keras"
TFLITE_PATH   = "./kws_model.tflite"
C_ARRAY_PATH  = "./kws_model.h"

# ── Reproductibilité ──────────────────────────────────────────
# Fixer le seed permet d'obtenir exactement les mêmes résultats
# à chaque run, crucial pour le débogage sur ressources limitées.
tf.random.set_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────
# ÉTAPE 1 — TÉLÉCHARGEMENT & PRÉPARATION DES DONNÉES
#
# Pourquoi cette étape ?
#   On utilise Google Speech Commands v2, un dataset public
#   de mots isolés enregistrés à 16kHz / 1s.
#   On ne garde que "yes" et "no" pour rester dans la RAM
#   de l'Arduino (256 KB sur Nano 33 BLE Sense).
#
#   Si tu veux entraîner sur du français ("oui"/"non"),
#   remplace cette section par tes propres enregistrements
#   (voir BONUS en bas du fichier).
# ─────────────────────────────────────────────────────────────

def download_speech_commands():

    """
    Stratégie en cascade :
    1. tensorflow_datasets (si installé)
    2. storage.googleapis.com (miroir direct)
    3. Instructions Kaggle/HuggingFace si tout échoue
    """
    if os.path.exists(DATA_DIR):
        print(f"[1/7] Dataset déjà présent dans {DATA_DIR}/")
        return

    # ── Tentative 1 : tensorflow_datasets ────────────────
    try:
        import tensorflow_datasets as tfds
        print("[1/7] Téléchargement via tensorflow_datasets...")

        ds_train, info = tfds.load(
            "speech_commands",
            split="train",
            with_info=True
        )
        label_names = info.features["label"].names
        yes_idx = label_names.index("yes")
        no_idx  = label_names.index("no")

        # Sauvegarde les .wav dans la structure attendue
        import soundfile as sf  # pip install soundfile

        for label_name, label_idx in [("yes", yes_idx), ("no", no_idx)]:
            out_dir = os.path.join(DATA_DIR, label_name)
            os.makedirs(out_dir, exist_ok=True)
            count = 0
            for sample in ds_train:
                if sample["label"].numpy() == label_idx:
                    audio = sample["audio"].numpy()
                    sf.write(
                        os.path.join(out_dir, f"{count:04d}.wav"),
                        audio, 16000, subtype="PCM_16"
                    )
                    count += 1
            print(f"  {label_name}: {count} fichiers sauvegardés")
        return

    except Exception as e:
        print(f"  tensorflow_datasets indisponible : {e}")


def load_audio_paths():
    """
    Scanne les dossiers et retourne (chemin_fichier, label_index).
    Retourne aussi la répartition pour vérification.
    """
    paths, labels_idx = [], []

    for idx, label in enumerate(LABELS):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"Dossier introuvable : {folder}\n"
                f"Lance d'abord download_speech_commands()"
            )
        files = [
            f for f in os.listdir(folder)
            if f.endswith(".wav")
        ]
        print(f"  {label:10s}: {len(files):5d} fichiers")
        for f in files:
            paths.append(os.path.join(folder, f))
            labels_idx.append(idx)

    return np.array(paths), np.array(labels_idx)


# ─────────────────────────────────────────────────────────────
# ÉTAPE 2 — EXTRACTION DES FEATURES (MFCC)
#
# Pourquoi les MFCC ?
#   L'Arduino ne peut pas traiter un signal audio brut :
#   il est trop volumineux (16000 samples/s) et contient
#   des redondances. Les MFCC (Mel-Frequency Cepstral
#   Coefficients) compressent l'information perceptuelle
#   de la voix en ~13 coefficients par frame de 10ms.
#
#   Résultat : une image 2D de shape (13, 98) au lieu de
#   (16000,) → 200x moins de données, et bien mieux adapté
#   aux CNNs.
#
#   L'échelle MEL imite l'oreille humaine qui perçoit les
#   fréquences de façon logarithmique.
# ─────────────────────────────────────────────────────────────

def extract_mfcc(file_path: str) -> np.ndarray:
    """
    Charge un fichier .wav, normalise sa durée à 1s,
    puis calcule et normalise les MFCC.

    Retourne un array de shape (N_MFCC, N_FRAMES, 1)
    prêt pour un CNN (hauteur × largeur × canaux).
    """
    # Chargement avec resampling automatique si nécessaire
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # ── Normalisation de la durée ──────────────────────────
    # Tous les clips doivent faire exactement 1s.
    # Un clip trop court est complété par des zéros (silence).
    # Un clip trop long est tronqué.
    target_length = int(SAMPLE_RATE * CLIP_DURATION)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # ── Calcul des MFCC ───────────────────────────────────
    # 1. Spectrogramme STFT (Centré = False)
    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window='hann',
        center=False
    )  # shape: (N_MFCC, N_FRAMES) = (13, 97)

# 2. Spectrogramme de puissance (Power = 2, divisé par N_FFT)
    power_spectrogram = (np.abs(stft) ** 2) / N_FFT

# 3. Banc de filtres Mel avec norme Slaney
    mel_basis = librosa.filters.mel(
        sr=sr, 
        n_fft=N_FFT, 
        n_mels=40, 
        norm='slaney', 
        htk=False
    )
    mel_spectrogram = np.dot(mel_basis, power_spectrogram)

# 4. Conversion Log naturel (maintien de l'epsilon 1e-10)
    log_mel = np.log(mel_spectrogram + 1e-10)
    
# 5. DCT-II brute (sans normalisation orthogonale)
    import scipy.fftpack
    mfcc = scipy.fftpack.dct(log_mel, type=2, axis=0, norm=None)[:N_MFCC]

# 6. Z-Score globale
    mean = np.mean(mfcc)
    std = np.std(mfcc) + 1e-8
    mfcc_normalized = (mfcc - mean) / std

    # Ajout de la dimension "canal" pour le CNN : (13, 98, 1)
    return mfcc_normalized[..., np.newaxis]


def build_dataset(paths: np.ndarray, labels: np.ndarray):
    """
    Extrait les features pour tout le dataset.
    Affiche une barre de progression.
    """
    print("[2/7] Extraction des MFCC...")
    X, y = [], []

    for path, label in tqdm(
        zip(paths, labels),
        total=len(paths),
        desc="  MFCC"
    ):
        try:
            feat = extract_mfcc(path)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"  Erreur sur {path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"  Dataset shape   : {X.shape}")   # (N, 13, 98, 1)
    print(f"  Distribution    : yes={np.sum(y==0)}, no={np.sum(y==1)}, background={np.sum(y==2)}")
    return X, y


def visualize_mfcc(X: np.ndarray, y: np.ndarray):
    """
    Affiche un exemple de MFCC pour chaque classe du dataset.
    Utile pour vérifier visuellement que les features sont
    bien différentes entre les classes.
    """

    n_labels = len(LABELS)

    fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 4))

    # Cas particulier si une seule classe
    if n_labels == 1:
        axes = [axes]

    for ax, label_idx, label_name in zip(axes, range(n_labels), LABELS):

        indices = np.where(y == label_idx)[0]

        if len(indices) == 0:
            ax.set_title(f"{label_name}\n(Aucun exemple)")
            ax.axis("off")
            continue

        idx = indices[0]

        im = ax.imshow(
            X[idx, :, :, 0],
            aspect="auto",
            origin="lower",
            cmap="viridis"
        )

        ax.set_title(f"MFCC — '{label_name}'")
        ax.set_xlabel(f"Frames temporelles ({X.shape[2]})")
        ax.set_ylabel(f"Coefficients MFCC ({X.shape[1]})")

        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("mfcc_visualization.png", dpi=150)
    plt.show()

    print("Visualisation sauvegardée : mfcc_visualization.png")


# ─────────────────────────────────────────────────────────────
# ÉTAPE 3 — ARCHITECTURE DU MODÈLE (CNN LÉGER)
#
# Pourquoi un CNN ?
#   Les MFCC sont des "images" 2D : temps × fréquence.
#   Un CNN détecte des patterns locaux (comme les transitions
#   formantiques du "ou" ou la fricative du "n") peu importe
#   leur position exacte dans le temps.
#
# Pourquoi aussi léger ?
#   L'Arduino Nano 33 BLE Sense a :
#   - 256 KB de RAM (flash)
#   - 1 MB de mémoire programme
#   Un ResNet ou MobileNet ne rentre pas.
#   Notre CNN ~ 20-30 KB après quantification.
#
# Architecture inspirée de DS-CNN (Depthwise Separable CNN)
# utilisée dans les papers KWS de Google.
# ─────────────────────────────────────────────────────────────

def build_model(input_shape: tuple) -> tf.keras.Model:
    """
    Construit un CNN léger pour KWS.
    input_shape = (N_MFCC, N_FRAMES, 1) = (13, 98, 1)
    """
    inputs = tf.keras.Input(shape=input_shape, name="mfcc_input")

    # ── Bloc 1 : Conv classique ────────────────────────────
    # 32 filtres 3×3 pour apprendre les patterns bas-niveau
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding="same", name="conv1"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    # → shape: (6, 49, 32)

    # ── Bloc 2 : Depthwise Separable Conv ─────────────────
    # Sépare la convolution spatiale et la projection
    # de profondeur : 8-9x moins de multiplications.
    # C'est la clé de l'efficacité sur microcontrôleur !
    x = tf.keras.layers.DepthwiseConv2D(
        (3, 3), padding="same", name="dw_conv1"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        64, (1, 1), padding="same", name="pw_conv1"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    # → shape: (3, 24, 64)

    # ── Bloc 3 : Global Average Pooling ───────────────────
    # Alternative au Flatten : réduit chaque feature map
    # à sa moyenne → supprime les infos spatiales résiduelles.
    # Résultat plus compact et moins sujet à l'overfitting.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # → shape: (64,)

    # ── Tête de classification ─────────────────────────────
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        NUM_LABELS, activation="softmax", name="output"
    )(x)
    
    model = tf.keras.Model(inputs, outputs, name="KWS_CNN")
    return model


# ─────────────────────────────────────────────────────────────
# ÉTAPE 4 — ENTRAÎNEMENT
#
# Pourquoi ces choix ?
#   - Adam : adaptatif, converge vite sur de petits datasets.
#   - EarlyStopping : arrête l'entraînement dès que la loss
#     de validation ne s'améliore plus → évite l'overfitting.
#   - ReduceLROnPlateau : réduit le learning rate si le modèle
#     stagne → affine les poids en fin d'entraînement.
#   - Data augmentation (TimeShift) : en parole, le mot peut
#     être dit plus tôt ou plus tard dans la seconde.
#     Shifter aléatoirement le signal rend le modèle robuste.
# ─────────────────────────────────────────────────────────────

class TimeShiftAugmentation(tf.keras.layers.Layer):
    """
    Layer de data augmentation : décale aléatoirement
    les MFCC dans le temps (axe des frames).
    Simule des variations de timing à la parole.
    Actif uniquement à l'entraînement (training=True).
    """
    def __init__(self, max_shift=10, **kwargs):
        super().__init__(**kwargs)
        self.max_shift = max_shift

    def call(self, x, training=False):
        if not training:
            return x
        shift = tf.random.uniform(
            (), -self.max_shift, self.max_shift, dtype=tf.int32
        )
        return tf.roll(x, shift=shift, axis=2)


def train_model(
    X_train, y_train,
    X_val, y_val,
    model: tf.keras.Model,
    epochs=100,
    batch_size=32
):
    """
    Compile, entraîne et sauvegarde le modèle.
    Retourne l'historique d'entraînement.
    """
    print("[4/7] Entraînement du modèle...")
    model.summary()

    # Comptage des paramètres — critique pour TinyML !
    total_params = model.count_params()
    print(f"\n  Total paramètres : {total_params:,}")
    print(f"  Taille estimée (float32) : {total_params * 4 / 1024:.1f} KB")
    print(f"  Taille estimée (int8)    : {total_params / 1024:.1f} KB")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,          # attend 10 epochs sans amélioration
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,           # divise lr par 2
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n  Modèle sauvegardé : {MODEL_PATH}")
    return history


def plot_training_history(history):
    """
    Trace les courbes d'accuracy et de loss.
    Un bon modèle : val_accuracy proche de train_accuracy
    (pas d'overfitting), toutes deux en croissance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# ÉTAPE 5 — ÉVALUATION
#
# Pourquoi évaluer avant de déployer ?
#   Sur Arduino, déboguer un modèle est difficile (pas de
#   logs, pas d'IDE interactif). Il vaut mieux vérifier ici :
#   - La matrice de confusion révèle les confusions oui/non.
#   - On calcule aussi l'accuracy sur X_test (données jamais
#     vues à l'entraînement).
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le set de test.
    Affiche la matrice de confusion.
    """
    print("[5/7] Évaluation sur le set de test...")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test accuracy : {acc * 100:.2f}%")
    print(f"  Test loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matrice de confusion (test acc={acc*100:.1f}%)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    return acc


# ─────────────────────────────────────────────────────────────
# ÉTAPE 6 — QUANTIFICATION & CONVERSION TFLITE
#
# Pourquoi quantifier ?
#   Le modèle TF natif utilise des float32 (4 octets/paramètre).
#   La quantification int8 remplace chaque poids par un entier
#   sur 8 bits (1 octet) → modèle 4× plus petit.
#   Sur Arduino (256 KB RAM), c'est souvent la différence entre
#   "ça rentre" et "ça ne rentre pas".
#
#   De plus, les opérations int8 sont beaucoup plus rapides
#   sur les MCU sans FPU (Floating Point Unit).
#
#   Le "representative dataset" sert à calibrer les plages
#   de valeurs pour les activations (pas seulement les poids).
# ─────────────────────────────────────────────────────────────

def quantize_and_convert(
    model: tf.keras.Model,
    X_representative: np.ndarray
) -> bytes:
    """
    Quantifie le modèle en int8 et le convertit en TFLite.
    Retourne les bytes du modèle TFLite.
    """
    print("[6/7] Quantification int8 + conversion TFLite...")

    # ── Dataset de calibration ────────────────────────────
    # TFLite a besoin de données réelles pour calculer les
    # valeurs min/max de chaque activation du réseau.
    # On utilise ~100 exemples, ce n'est pas un entraînement.
    def representative_dataset_gen():
        for i in range(min(100, len(X_representative))):
            sample = X_representative[i:i+1]  # shape: (1, 13, 98, 1)
            yield [sample.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Activation de la quantification post-entraînement complète
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    # Force les I/O en int8 aussi (obligatoire pour MCU sans FPU)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  Modèle TFLite sauvegardé : {TFLITE_PATH}")
    print(f"  Taille du modèle int8    : {size_kb:.1f} KB")

    if size_kb > 200:
        print("  ⚠️  Attention : >200 KB, risque de ne pas rentrer sur Arduino !")
    else:
        print("  ✅ Taille compatible avec l'Arduino Nano 33 BLE Sense")

    return tflite_model


def verify_tflite_model(
    tflite_model: bytes,
    X_sample: np.ndarray,
    y_sample: np.ndarray
):
    """
    Vérifie que le modèle TFLite quantifié donne
    des prédictions cohérentes avec le modèle original.
    La quantification peut dégrader légèrement l'accuracy.
    On accepte en général une perte < 2%.
    """
    print("  Vérification du modèle TFLite sur quelques exemples...")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Récupère les paramètres de quantification pour les I/O
    input_scale, input_zero_point = (
        input_details[0]["quantization"]
    )

    correct = 0
    n_samples = min(50, len(X_sample))

    for i in range(n_samples):
        sample = X_sample[i:i+1]  # (1, 13, 98, 1)

        # Conversion float32 → int8 selon les paramètres de quant.
        sample_int8 = (sample / input_scale + input_zero_point
                       ).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], sample_int8)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        pred   = np.argmax(output, axis=1)[0]

        if pred == y_sample[i]:
            correct += 1

    acc = correct / n_samples
    print(f"  Accuracy TFLite (int8) : {acc * 100:.1f}% sur {n_samples} exemples")
    return acc


# ─────────────────────────────────────────────────────────────
# ÉTAPE 7 — GÉNÉRATION DU TABLEAU C POUR ARDUINO
#
# Pourquoi un tableau C ?
#   L'Arduino ne sait pas lire un fichier .tflite depuis
#   un système de fichiers. On convertit le modèle en tableau
#   d'octets C qui sera directement compilé dans le firmware.
#   C'est équivalent à la commande bash :
#     xxd -i kws_model.tflite > kws_model.h
# ─────────────────────────────────────────────────────────────

def convert_tflite_to_c_array(
    tflite_model: bytes,
    var_name: str = "kws_model"
):
    """
    Convertit le modèle TFLite en tableau C.
    Génère un fichier .h prêt à être copié dans l'IDE Arduino.
    """
    print("[7/7] Génération du tableau C pour Arduino...")

    hex_array = ", ".join(f"0x{b:02x}" for b in tflite_model)
    model_len = len(tflite_model)

    c_header = f"""// ============================================================
// Modèle KWS généré automatiquement
// Ne pas modifier manuellement.
//
// Classes : {LABELS}
// Taille  : {model_len / 1024:.1f} KB ({model_len} octets)
// Format  : TFLite int8 quantifié
// ============================================================

#ifndef KWS_MODEL_H
#define KWS_MODEL_H

#include <stdint.h>

// Alignement 8 octets requis par TFLite Micro
alignas(8) const uint8_t {var_name}[] = {{
  {hex_array}
}};

const unsigned int {var_name}_len = {model_len};

// Paramètres audio — DOIVENT correspondre au preprocessing Python
#define SAMPLE_RATE   {SAMPLE_RATE}
#define CLIP_DURATION {CLIP_DURATION}f
#define N_MFCC        {N_MFCC}
#define N_FRAMES      {N_FRAMES}
#define NUM_LABELS    {NUM_LABELS}

// Labels dans le même ordre que l'entraînement
const char* LABELS[] = {{{', '.join(f'"{l}"' for l in LABELS)}}};

#endif // KWS_MODEL_H
"""
    with open(C_ARRAY_PATH, "w") as f:
        f.write(c_header)

    print(f"  Tableau C sauvegardé : {C_ARRAY_PATH}")
    print(f"  Copie ce fichier dans ton projet Arduino IDE.")
    return c_header


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WORKSHOP TINYML — KWS Arduino")
    print("=" * 60)

    # ── 1. Données ───────────────────────────────────────────
    download_speech_commands()
    paths, labels_idx = load_audio_paths()

    # ── 2. Features ──────────────────────────────────────────
    X, y = build_dataset(paths, labels_idx)
    visualize_mfcc(X, y)

    # ── Split train / val / test ─────────────────────────────
    # 70% train | 15% val | 15% test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )
    print(f"\n  Train : {len(X_train):5d} | Val : {len(X_val):5d} | Test : {len(X_test):5d}")

    # ── 3. Modèle ────────────────────────────────────────────
    input_shape = X_train.shape[1:]  # (13, 98, 1)
    model = build_model(input_shape)

    # ── 4. Entraînement ──────────────────────────────────────
    history = train_model(
        X_train, y_train, X_val, y_val, model,
        epochs=100, batch_size=32
    )
    plot_training_history(history)

    # ── 5. Évaluation ────────────────────────────────────────
    acc = evaluate_model(model, X_test, y_test)

    if acc < 0.85:
        print("\n⚠️  Accuracy < 85% — consulte la section CONSEILS en bas du fichier.")

    # ── 6. Quantification ────────────────────────────────────
    tflite_model = quantize_and_convert(model, X_train)
    verify_tflite_model(tflite_model, X_test, y_test)

    # ── 7. Export C ──────────────────────────────────────────
    convert_tflite_to_c_array(tflite_model)

    print("\n" + "=" * 60)
    print("  Pipeline terminé ! Prochaines étapes :")
    print("  1. Ouvre Arduino IDE")
    print("  2. Installe 'Arduino_TensorFlowLite' via le Library Manager")
    print("  3. Copie kws_model.h dans ton sketch")
    print("  4. Utilise le code Arduino fourni ci-dessous")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# BONUS — AIDE POUR ENREGISTRER TES PROPRES DONNÉES (FRANÇAIS)
#
# Si tu veux "oui"/"non" en français plutôt qu'anglais,
# tu dois enregistrer tes propres audio.
# Ce helper génère des commandes sox pour te guider.
# ─────────────────────────────────────────────────────────────

def record_custom_data_instructions():
    """
    Affiche les instructions pour enregistrer tes propres
    mots-clés en français avec sox ou un script Python.
    """
    instructions = """
╔══════════════════════════════════════════════════════════╗
║  ENREGISTREMENT DE DONNÉES PERSONNALISÉES                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Structure de dossiers attendue :                        ║
║                                                          ║
║  ./custom_data/                                          ║
║    oui/                                                  ║
║      001.wav  002.wav  ...  (min. 100 fichiers)          ║
║    non/                                                  ║
║      001.wav  002.wav  ...  (min. 100 fichiers)          ║
║                                                          ║
║  Spécifications audio :                                  ║
║  - Format  : WAV 16-bit PCM mono                         ║
║  - Sample rate : 16000 Hz                                ║
║  - Durée  : exactement 1 seconde                         ║
║                                                          ║
║  Avec sox (terminal) :                                   ║
║  rec -r 16000 -c 1 -b 16 oui/001.wav trim 0 1           ║
║                                                          ║
║  Conseils de qualité :                                   ║
║  - Varie les locuteurs (hommes, femmes, enfants)         ║
║  - Varie les accents                                     ║
║  - Enregistre avec du bruit de fond léger                ║
║  - Minimum 150 exemples par classe                       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """
    print(instructions)


if __name__ == "__main__":
    main()

    # Décommente pour voir les instructions d'enregistrement custom :
    # record_custom_data_instructions()
