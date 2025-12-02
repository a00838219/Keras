"""Entrenamiento y predicción para modelos tabulares (vino y forest cover)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RANDOM_STATE = 42
DEFAULT_DATASET = "covtype"
DEFAULT_MAX_SAMPLES = 120_000


@dataclass
class DatasetConfig:
    name: str
    task: Literal["binary", "multiclass"]
    target: str
    class_names: List[str]
    positive_label: Optional[int] = None


def artifact_paths(dataset: str) -> Dict[str, Path]:
    """Genera rutas de artefactos específicas por dataset para evitar conflictos."""
    prefix = dataset.lower()
    return {
        "model": BASE_DIR / f"{prefix}_model.keras",
        "scaler": BASE_DIR / f"{prefix}_scaler.pkl",
        "metadata": BASE_DIR / f"{prefix}_metadata.json",
        "history": BASE_DIR / f"{prefix}_history.json",
    }


def load_covtype_dataset(max_samples: Optional[int] = DEFAULT_MAX_SAMPLES) -> Tuple[pd.DataFrame, DatasetConfig]:
    """Carga Forest Cover Type (UCI). Reduce muestras si max_samples está definido."""
    try:
        from sklearn.datasets import fetch_covtype
    except ImportError as exc:
        raise ImportError("scikit-learn es requerido para cargar el dataset covtype.") from exc

    ds = fetch_covtype(data_home=DATA_DIR, as_frame=True)
    df = ds.frame
    df = df.rename(columns={"Cover_Type": "target"})
    # Target llega entre 1 y 7; lo llevamos a 0-6.
    df["target"] = df["target"].astype(int) - 1

    if max_samples is not None and len(df) > max_samples:
        df, _ = train_test_split(
            df, train_size=max_samples, stratify=df["target"], random_state=RANDOM_STATE
        )

    config = DatasetConfig(
        name="covtype",
        task="multiclass",
        target="target",
        class_names=[
            "Spruce/Fir",
            "Lodgepole Pine",
            "Ponderosa Pine",
            "Cottonwood/Willow",
            "Aspen",
            "Douglas-fir",
            "Krummholz",
        ],
    )
    return df, config


def load_dataset(dataset: str, max_samples: Optional[int]) -> Tuple[pd.DataFrame, DatasetConfig]:
    dataset = dataset.lower()
    if dataset == "covtype":
        return load_covtype_dataset(max_samples)
    raise ValueError(f"Dataset no soportado: {dataset}. Usa 'covtype'.")


def prepare_data(df: pd.DataFrame, config: DatasetConfig) -> Tuple[np.ndarray, ...]:
    """Escala los datos y genera particiones estratificadas."""
    X = df.drop(columns=[config.target])
    y = df[config.target].astype(int)
    stratify = y if config.task in {"binary", "multiclass"} else None

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, stratify=stratify, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classes = np.unique(y_train)
    weights_arr = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(cls): float(w) for cls, w in zip(classes, weights_arr)}

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        scaler,
        class_weights,
        X.columns.tolist(),
        X_test.reset_index(drop=True),
    )


def residual_block(x: keras.layers.Layer, units: int, dropout: float) -> keras.layers.Layer:
    """Bloque residual sencillo para tabulares."""
    shortcut = keras.layers.Dense(units, kernel_initializer="he_normal", activation=None)(x)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    out = keras.layers.Dense(units, activation="relu", kernel_initializer="he_normal")(x)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Dropout(dropout)(out)
    out = keras.layers.Dense(units, activation=None, kernel_initializer="he_normal")(out)
    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Add()([shortcut, out])
    out = keras.layers.Activation("relu")(out)
    return out


def build_model(input_dim: int, num_classes: int, task: Literal["binary", "multiclass"]) -> keras.Model:
    """Modelo híbrido: MLP residual + rama convolucional 1D para cumplir uso de unidades especializadas."""
    inputs = keras.Input(shape=(input_dim,), name="features")
    x_norm = keras.layers.BatchNormalization()(inputs)

    # Rama MLP residual
    mlp = residual_block(x_norm, 256, 0.35)
    mlp = residual_block(mlp, 192, 0.3)
    mlp = residual_block(mlp, 128, 0.25)
    mlp = keras.layers.Dense(96, activation="relu", kernel_initializer="he_normal")(mlp)
    mlp = keras.layers.Dropout(0.2)(mlp)

    # Rama convolucional 1D (requerimiento de unidades especializadas)
    conv = keras.layers.Reshape((input_dim, 1))(x_norm)
    conv = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.GlobalMaxPooling1D()(conv)
    conv = keras.layers.Dropout(0.2)(conv)

    x = keras.layers.concatenate([mlp, conv])
    x = keras.layers.Dense(96, activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.Dropout(0.2)(x)

    if task == "binary":
        outputs = keras.layers.Dense(1, activation="sigmoid", name="score")(x)
        loss = "binary_crossentropy"
        metrics = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ]
    else:
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="class")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ]

    model = keras.Model(inputs, outputs, name=f"{task}_tabular_resnet")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=metrics)
    return model


def train(dataset: str, max_samples: Optional[int], epochs: int) -> None:
    df, config = load_dataset(dataset, max_samples)
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
        X_test_raw,
    ) = prepare_data(df, config)

    model = build_model(X_train.shape[1], len(config.class_names), config.task)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.4, patience=6, monitor="val_loss"),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=256 if dataset == "covtype" else 128,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    probs = model.predict(X_test, verbose=0)
    preds = probs.argmax(axis=1)

    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(
        y_test,
        preds,
        target_names=config.class_names,
        output_dict=True,
        zero_division=0,
    )

    paths = artifact_paths(dataset)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model.save(paths["model"])
    joblib.dump(scaler, paths["scaler"])

    metadata = {
        "dataset": config.name,
        "task": config.task,
        "feature_order": feature_order,
        "class_names": config.class_names,
        "positive_label": config.positive_label,
        "artifacts": {k: v.name for k, v in paths.items()},
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "samples": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
            "total": int(len(df)),
        },
        "threshold": None,
        "classification_report": report,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_payload = {k: [float(v) for v in values] for k, values in history.history.items()}
    paths["history"].write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    sample_file = BASE_DIR / f"sample_input_{dataset}.csv"
    X_test_raw.iloc[:5].to_csv(sample_file, index=False)

    print("== Entrenamiento finalizado ==")
    print(f"Dataset: {config.name} | Accuracy (test): {accuracy:.3f} | Macro F1: {macro_f1:.3f}")
    print(
        f"Artefactos guardados en: {paths['model']}, {paths['scaler']}, {paths['metadata']}, {paths['history']}, {sample_file}"
    )


def predict(input_path: Path, dataset: str) -> None:
    paths = artifact_paths(dataset)
    if not paths["metadata"].exists() or not paths["model"].exists() or not paths["scaler"].exists():
        raise FileNotFoundError(
            f"No hay artefactos entrenados para '{dataset}'. Ejecuta `python wine_quality.py train --dataset {dataset}` primero."
        )

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    model = keras.models.load_model(paths["model"])
    scaler = joblib.load(paths["scaler"])
    feature_order = metadata["feature_order"]

    df = pd.read_csv(input_path)
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {input_path}: {missing}")

    features = df[feature_order].copy()
    processed = scaler.transform(features)
    probs = model.predict(processed, verbose=0)

    class_names = metadata["class_names"]
    preds = probs.argmax(axis=1)
    for idx, (pred, prob_row) in enumerate(zip(preds, probs)):
        label = class_names[int(pred)]
        confidence = float(prob_row[pred])
        print(
            json.dumps(
                {
                    "row": int(idx),
                    "label": label,
                    "confidence": confidence,
                    "scores": {name: float(prob_row[i]) for i, name in enumerate(class_names)},
                },
                ensure_ascii=False,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento/predicción para datasets tabulares.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrena el modelo y guarda artefactos.")
    train_parser.add_argument(
        "--dataset",
        choices=["covtype"],
        default=DEFAULT_DATASET,
        help="Dataset a utilizar (solo covtype).",
    )
    train_parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Máximo de filas a usar para acelerar entrenamiento en covtype. Usa 0 para desactivar.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=140,
        help="Número máximo de epochs (default 140; usa uno menor si quieres entrenamientos más rápidos).",
    )
    train_parser.set_defaults(
        func=lambda args: train(args.dataset, None if args.max_samples == 0 else args.max_samples, args.epochs)
    )

    predict_parser = subparsers.add_parser("predict", help="Genera predicciones desde un CSV.")
    predict_parser.add_argument("--input", required=True, help="Ruta al CSV con características.")
    predict_parser.add_argument(
        "--dataset",
        choices=["covtype"],
        default=DEFAULT_DATASET,
        help="Dataset cuyo modelo se usará para predecir (solo covtype).",
    )
    predict_parser.set_defaults(func=lambda args: predict(Path(args.input), args.dataset))

    return parser.parse_args()


def main() -> None:
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
