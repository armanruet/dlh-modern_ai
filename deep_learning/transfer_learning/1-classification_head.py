#!/usr/bin/env python3
""" module for a function to build a frozen MobileNetV2 feature extractor """

from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """Bolt a trainable classification head onto a pooled-feature model."""
    inputs = base_model.input          # reuse the existing (224,224,3) input
    x = base_model.output              # pooled features: (None, 1280)

    x = keras.layers.Dense(128, activation="relu")(
        x)            # violet zone: bottleneck
    outputs = keras.layers.Dense(
        num_classes, activation="softmax")(x)  # probabilities

    model = keras.Model(inputs, outputs, name="transfer_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",   # integer labels 0..K-1
        metrics=["accuracy"],
    )
    return model
