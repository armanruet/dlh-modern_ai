#!/usr/bin/env python3
""" module for a function to build a frozen MobileNetV2 feature extractor """

from tensorflow import keras


def build_feature_extractor():
    """MobileNetV2 backbone, fully frozen, pooled feature vector."""
    # 1. Load ImageNet-pretrained MobileNetV2 WITHOUT its 1000-class classifier
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",          # the borrowed eyes (ImageNet weights)
        input_shape=(224, 224, 3),   # the geometry these weights expect
        include_top=False,           # drop the ImageNet classifier head
    )

    # 2. Freeze EVERY weight in the backbone (cyan zone)
    base_model.trainable = False

    # 3. Wire it up: Input -> frozen backbone -> global average pooling
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)   # keep BatchNorm in inference mode
    outputs = keras.layers.GlobalAveragePooling2D()(x)

    return keras.Model(inputs, outputs)
