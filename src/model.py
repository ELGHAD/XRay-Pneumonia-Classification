"""ResNet50 transfer-learning model."""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from src import config
from src.data import build_augmentation


@tf.keras.utils.register_keras_serializable(package="pneumonia")
class ResNetPreprocess(tf.keras.layers.Layer):
    """Applies ResNet50's ImageNet preprocessing (RGB->BGR, mean subtraction).

    Kept as a layer so the saved model is self-contained: inference code passes
    raw [0, 255] pixels and cannot get the preprocessing wrong.
    """

    def call(self, inputs):
        return preprocess_input(inputs)


def build_model():
    """ImageNet ResNet50 backbone with a binary classification head."""
    inputs = tf.keras.Input(shape=(*config.IMAGE_SIZE, 3), name="image")
    x = build_augmentation()(inputs)
    x = ResNetPreprocess(name="resnet_preprocess")(x)

    backbone = ResNet50(weights="imagenet", include_top=False, name="resnet50")
    backbone.trainable = False
    # training=False keeps BatchNorm statistics frozen, also during fine-tuning.
    x = backbone(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_prob")(x)

    return tf.keras.Model(inputs, outputs, name="resnet50_pneumonia")


def get_backbone(model):
    return model.get_layer("resnet50")


def unfreeze_top_layers(model, from_layer=config.FINE_TUNE_FROM_LAYER):
    """Unfreeze backbone layers from `from_layer` onward, keeping BatchNorm frozen."""
    backbone = get_backbone(model)
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        if i < from_layer or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
