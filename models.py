import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from helpers import *
import tensorflow as tf
import keras_tuner as kt
import tensorflow as tf
import keras_tuner as kt
import keras.backend as K




def basic_FCF():
    """
    Creates a basic fully connected feedforward neural network model.
    The hyperparameter (learning rate, number of layers, and units in each layer) were found using the keras tuner.

    Returns:
    model (tf.keras.Sequential): The created model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(160, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(288, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(96, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(2, activation='softmax', kernel_initializer=tf.keras.initializers.HeNormal())
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.F1Score(), 'accuracy'])
    
    return model

def basic_CNN():
    """
    Creates a basic Convolutional Neural Network (CNN) model (the convolution part is the one given).
    The hyperparameter of the FCF were found using the keras tuner.

    Returns:
    model (keras.Sequential): The created CNN model.
    """
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(),
                            bias_initializer=tf.keras.initializers.HeNormal())
    )

    model.add(
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
    )

    model.add(
        tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(),
                            bias_initializer=tf.keras.initializers.HeNormal())
    )

    model.add(
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
    )

    model.add(tf.keras.layers.Flatten())

    # Or units=416
    model.add(
            tf.keras.layers.Dense(units=160, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')
    )
    # Or units=32
    model.add(
            tf.keras.layers.Dense(units=288, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')
    )
    # Or units=32
    model.add(
            tf.keras.layers.Dense(units=96, kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')
    )

    model.add(tf.keras.layers.Dense(2, activation='softmax', kernel_initializer=tf.keras.initializers.HeNormal()))

    return model


def TverskyLoss(targets, inputs, smooth=1):
    """
    Calculates the Tversky loss between the target and input tensors.

    Args:
        targets (tensor): The target tensor.
        inputs (tensor): The input tensor.
        smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1.

    Returns:
        tensor: The Tversky loss.

    References:
        - Implementation: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
        - Paper: https://arxiv.org/abs/1706.05721
    """
    alpha = 0.5
    beta = 0.5

    # Flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - Tversky


def conv_block(x, filters, regularization=0, activation='relu', dropout=0):
    """
    A function that applies a convolutional block to the input tensor.

    Parameters:
    - x: Input tensor.
    - filters: Number of filters in the convolutional layers.
    - regularization: L2 regularization parameter (default=0).
    - activation: Activation function to use (default='relu').
    - dropout: Dropout rate (default=0).

    Returns:
    - Output tensor after applying the convolutional block.
    """
    initializer = tf.keras.initializers.HeNormal(seed=45)

    if regularization!=0:
        if activation == 'relu' :
            x = tf.keras.layers.Conv2D(filters, 3, padding = "same", kernel_regularizer=tf.keras.regularizers.L2(l2=regularization), activation = activation, kernel_initializer = initializer)(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding = "same", kernel_regularizer=tf.keras.regularizers.L2(l2=regularization), activation = activation, kernel_initializer = initializer)(x)
            x = tf.keras.layers.BatchNormalization()(x)

        else :
            x = tf.keras.layers.Conv2D(filters, 3, padding = "same", kernel_regularizer=tf.keras.regularizers.L2(l2=regularization), activation = tf.keras.layers.LeakyRelU(alpha=0.2), kernel_initializer = initializer)(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding = "same", kernel_regularizer=tf.keras.regularizers.L2(l2=regularization), activation = tf.keras.layers.LeakyRelU(alpha=0.2), kernel_initializer = initializer)(x)
            x = tf.keras.layers.BatchNormalization()(x)

    else :
        if activation == 'relu' :
            x = tf.keras.layers.Conv2D(filters, 3, padding = "same", activation = activation, kernel_initializer = initializer)(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding = "same", activation = activation, kernel_initializer = initializer)(x)
            x = tf.keras.layers.BatchNormalization()(x)

        else :
            x = tf.keras.layers.Conv2D(filters, 3, padding = "same", activation = tf.keras.layers.LeakyRelU(alpha=0.2), kernel_initializer = initializer)(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding = "same", activation = tf.keras.layers.LeakyRelU(alpha=0.2), kernel_initializer = initializer)(x)
            x = tf.keras.layers.BatchNormalization()(x)

    if dropout != 0 :
            x = tf.keras.layers.Dropout(dropout)(x)

    return x


def encoder_block(x, filters, regularization, activation, dropout) :
    """
    This function represents an encoder block in a convolutional neural network, more precisely for an UNET model.

    Parameters:
    - x: Input tensor.
    - filters: Number of filters in the convolutional layers.
    - regularization: Regularization parameter for the convolutional layers.
    - activation: Activation function to be used in the convolutional layers.
    - dropout: Dropout rate for the convolutional layers.

    Returns:
    - x: Output tensor from the convolutional layers, for the skip connections.
    - p: Output tensor from the encoder block (after the max pooling layer).
    """
    x = conv_block(x, filters, regularization, activation, dropout)
    p = tf.keras.layers.MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(x, skip_features, filters, regularization, activation, dropout) :
    """
    Decoder block for a U-Net model.

    Args:
        x (tf.Tensor): Input tensor.
        skip_features (tf.Tensor): Skip connection tensor from the encoder.
        filters (int): Number of filters for the convolutional layers.
        regularization (float): Regularization parameter for the convolutional layers.
        activation (str): Activation function to be used.
        dropout (float): Dropout rate for the dropout layer.

    Returns:
        tf.Tensor: Output tensor after applying the decoder block operations.
    """
    x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters, regularization, activation, dropout)
    return x


def build_unet(input_shape=(None, None, 3), start_filter=32, num_stages= 4, dropout=0, activation='relu', regularization=0):
    """
    Builds a U-Net model for image segmentation.

    Args:
        input_shape (tuple): The shape of the input images. Default is (None, None, 3).
        start_filter (int): The number of filters in the first convolutional layer. Default is 32.
        num_stages (int): The number of stages in the U-Net architecture. Default is 4.
        dropout (float): The dropout rate. Default is 0.
        activation (str): The activation function to use. Default is 'relu'.
        regularization (float): The regularization strength. Default is 0.

    Returns:
        tf.keras.Model: The U-Net model.
    """
    inputs = tf.keras.layers.Input(input_shape)

    skips = []

    filter = start_filter

    x = inputs
    
    # Encoder part of the U-Net
    for stage in range(0, num_stages):
        skip, x = encoder_block(x, filter, regularization, activation, dropout)
        skips.append(skip)
        filter = filter * 2

    # Bottleneck part of the U-Net
    x = conv_block(x, filter, regularization, activation, dropout)

    skips.reverse()

    # Decoder part of the U-Net
    for skip in skips :
        filter = filter / 2
        x = decoder_block(x, skip, filter, regularization, activation, dropout)


    output = tf.keras.layers.Conv2D(2, 1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), activation='softmax')(x)

    return  tf.keras.Model(inputs=inputs, outputs=output)






