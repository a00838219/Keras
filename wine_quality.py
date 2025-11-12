"""Entrenamiento y predicción para el modelo de calidad de vino."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_FILE = BASE_DIR / "wine_quality_model.keras"
SCALER_FILE = BASE_DIR / "wine_scaler.pkl"
METADATA_FILE = BASE_DIR / "wine_metadata.json"
HISTORY_FILE = BASE_DIR / "training_history.json"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.45


def load_dataset() -> pd.DataFrame:
    """Combina vinos tintos y blancos, agregando el tipo como característica."""
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    red["wine_type"] = 0
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    white["wine_type"] = 1
    df = pd.concat([red, white], ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def create_target(series: pd.Series) -> pd.Series:
    """Target binario: 1 si el vino se considera premium (>=7)."""
    return (series >= 7).astype(int)


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """Escala los datos y genera particiones estratificadas."""
    df = df.copy()
    df["target"] = create_target(df["quality"])
    X = df.drop(columns=["quality", "target"])
    y = df["target"].astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Pesos manuales para reforzar la clase positiva poco representada.
    weight_positive = max(2.5, float(len(y_train) - y_train.sum()) / float(y_train.sum()))
    weights = {0: 1.0, 1: weight_positive}

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        scaler,
        weights,
        X.columns.tolist(),
    )


def build_model(input_dim: int) -> keras.Model:
    """Modelo híbrido compacto para clasificación binaria."""
    inputs = keras.Input(shape=(input_dim,), name="features")
    norm = keras.layers.BatchNormalization()(inputs)

    dense_branch = keras.layers.Dense(160, activation="relu")(norm)
    dense_branch = keras.layers.Dropout(0.35)(dense_branch)

    seq = keras.layers.Reshape((input_dim, 1))(norm)
    seq = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(seq)
    seq = keras.layers.BatchNormalization()(seq)
    seq = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(seq)
    seq = keras.layers.BatchNormalization()(seq)
    seq = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False))(seq)
    seq = keras.layers.Dropout(0.35)(seq)

    merged = keras.layers.concatenate([dense_branch, seq])
    merged = keras.layers.Dense(128, activation="relu")(merged)
    merged = keras.layers.Dropout(0.3)(merged)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="quality")(merged)

    model = keras.Model(inputs, outputs, name="wine_quality_hybrid")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=7e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train() -> None:
    df = load_dataset()
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        class_weights,
        feature_order,
    ) = prepare_data(df)

    model = build_model(X_train.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(patience=18, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.35, patience=8, monitor="val_loss"),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=128,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    probs = model.predict(X_test, verbose=0)
    binary_preds = (probs.flatten() >= DEFAULT_THRESHOLD).astype(int)
    accuracy = accuracy_score(y_test, binary_preds)
    macro_f1 = f1_score(y_test, binary_preds, average="macro")
    report = classification_report(
        y_test,
        binary_preds,
        target_names=["Estándar", "Premium"],
        output_dict=True,
        zero_division=0,
    )

    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    metadata = {
        "feature_order": feature_order,
        "buckets": {
            "0": "Estándar (<=6)",
            "1": "Premium (>=7)",
        },
        "model_file": MODEL_FILE.name,
        "scaler_file": SCALER_FILE.name,
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "samples": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
            "total": int(len(df)),
        },
        "threshold": DEFAULT_THRESHOLD,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_payload = {k: [float(v) for v in values] for k, values in history.history.items()}
    HISTORY_FILE.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    print("== Entrenamiento finalizado ==")
    print(f"Accuracy (test): {accuracy:.3f} | Macro F1: {macro_f1:.3f}")
    print(f"Artefactos guardados en: {MODEL_FILE}, {SCALER_FILE}, {METADATA_FILE}, {HISTORY_FILE}")


def predict(input_path: Path, threshold: float = DEFAULT_THRESHOLD) -> None:
    if not METADATA_FILE.exists() or not MODEL_FILE.exists() or not SCALER_FILE.exists():
        raise FileNotFoundError("No hay artefactos entrenados. Ejecuta `python wine_quality.py train` primero.")

    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    model = keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_order = metadata["feature_order"]

    df = pd.read_csv(input_path)
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {input_path}: {missing}")

    features = df[feature_order].copy()
    if "wine_type" in features.columns:
        features["wine_type"] = features["wine_type"].apply(lambda v: 0 if str(v).strip().lower() in {"0", "red", "tinto"} else 1)
    processed = scaler.transform(features)

    probs = model.predict(processed, verbose=0).flatten()
    label_map = metadata["buckets"]

    for idx, prob in enumerate(probs):
        label = label_map["1"] if prob >= threshold else label_map["0"]
        print(
            json.dumps(
                {
                    "row": int(idx),
                    "label": label,
                    "confidence": float(prob if prob >= threshold else 1 - prob),
                    "score_premium": float(prob),
                },
                ensure_ascii=False,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de calidad de vino (train/predict).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrena el modelo y guarda artefactos.")
    train_parser.set_defaults(func=lambda args: train())

    predict_parser = subparsers.add_parser("predict", help="Genera predicciones desde un CSV.")
    predict_parser.add_argument("--input", required=True, help="Ruta al CSV con características.")
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Umbral para clasificar como premium (probabilidad mínima).",
    )
    predict_parser.set_defaults(func=lambda args: predict(Path(args.input), args.threshold))

    return parser.parse_args()


def main() -> None:
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
