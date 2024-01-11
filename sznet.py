import tensorflow as tf
from keras import Model
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, Input, MaxPool2D)


def temporal_conv(inputs: tf.Tensor, n_filters: int, kernel_regularizer, dropout: float = None):
    x = Conv2D(n_filters, kernel_size=(9, 1), strides=(1, 1), padding="valid", kernel_regularizer=kernel_regularizer())(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    return x

def temporal_max_pool(inputs: tf.Tensor):
    x = MaxPool2D((2, 1), strides=(2, 1), padding="valid")(inputs)
    return x

def temporal_feature_extraction_block(inputs: tf.Tensor, kernel_regularizer, depth=3, initial_filters=16, dropout=0.25):
    x = inputs
    for i in range(depth):
        filters = initial_filters * (2 ** i)
        x = temporal_conv(x, filters, kernel_regularizer, dropout)
        x = temporal_conv(x, filters, kernel_regularizer, dropout=None)
        x = temporal_max_pool(x)
    return x

def spatial_feature_extraction_block(inputs: tf.Tensor, n_filters: int, kernel_regularizer):
    x = Conv2D(n_filters, (1, 3), strides=(1, 1), padding="valid", kernel_regularizer=kernel_regularizer())(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def spatio_temporal_feature_extraction_block(inputs: tf.Tensor, n_filters: int, kernel_regularizer, dropout: float = None, batch_norm=False):
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="valid", kernel_regularizer=kernel_regularizer())(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    return x

def flatten_block(inputs: tf.Tensor):
    return Flatten()(inputs)

def dense_block(inputs: tf.Tensor, units: int, dropout: float, kernel_regularizer):    
    x = Dense(units, activation="relu", kernel_regularizer=kernel_regularizer())(inputs)
    x = Dropout(dropout)(x)
    x = Dense(units // 2, activation="relu", kernel_regularizer=kernel_regularizer())(x)
    x = Dropout(dropout)(x)
    x = Dense(units // 4, activation="relu", kernel_regularizer=kernel_regularizer())(x)

    return x

def build_sz_net(input_shape: tuple[int, ...], dropout=0.25, dense_dropout=0.25, initial_filters=16,
                 initial_deep_units=512, l2_regularization_coef=1e-4):
    input = Input(input_shape)

    def kernel_regularizer():
        if l2_regularization_coef is None:
            return None
        return tf.keras.regularizers.L2(l2_regularization_coef)

    x = temporal_feature_extraction_block(input, kernel_regularizer, depth=3, initial_filters=initial_filters, dropout=dropout)
    x = spatial_feature_extraction_block(x, initial_filters * 2, kernel_regularizer)
    x = spatio_temporal_feature_extraction_block(x, initial_filters * 2, kernel_regularizer, dropout=dropout)
    x = flatten_block(x)
    x = dense_block(x, initial_deep_units, dense_dropout, kernel_regularizer)
    
    output = Dense(1, activation="sigmoid")(x)

    model =  Model(input, output, name="sznet")
    return model
