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
#https://www.tensorflow.org/tutorials/generative/pix2pix

def down(filters, size, batch_norm=True, regularization=0, activation='relu', dropout=0):
  initializer = tf.keras.initializers.HeNormal()

  downsample = tf.keras.Sequential()

  if regularization!=0 :
    downsample.add(
        tf.keras.layers.Conv2D(filters, size, padding='same',
                              kernel_initializer=initializer, kernel_regularizer=keras.regularizers.L2(l2=regularization), activation=activation)
    )

  else :
    downsample.add(
        tf.keras.layers.Conv2D(filters, size, padding='same',
                              kernel_initializer=initializer, activation=activation)
    )

  if batch_norm :
    downsample.add(keras.layers.BatchNormalization())

  if dropout!=0 :
    downsample.add(keras.layers.Dropout(dropout))

  return downsample


def up(filters, size, batch_norm=True, regularization=0, activation='relu', dropout=0):
  initializer = tf.keras.initializers.HeNormal()

  upsample = tf.keras.Sequential()

  if regularization!=0 :
    upsample.add(
        tf.keras.layers.Conv2DTranspose(filters, size, 2, padding='same',
                              kernel_initializer=initializer, kernel_regularizer=keras.regularizers.L2(l2=regularization), activation=activation)
    )


  else :
    upsample.add(
        tf.keras.layers.Conv2DTranspose(filters, size, 2, padding='same',
                              kernel_initializer=initializer, activation=activation)
    )

  if batch_norm :
    upsample.add(keras.layers.BatchNormalization())

  if dropout!=0 :
    upsample.add(keras.layers.Dropout(dropout))

  return upsample

def unet(dropout=0, regularization=0, activation='relu', input=(400, 400, 3)):
  inputs = tf.keras.layers.Input(input)

  encoder = [
      down(32, 3, batch_norm=True, regularization=regularization, activation=activation, dropout=dropout),
      down(64, 3, batch_norm=True, regularization=regularization, activation=activation, dropout=dropout),
      down(128, 3, batch_norm=True, regularization=regularization, activation=activation, dropout=dropout),
      down(256, 3, batch_norm=True, regularization=regularization, activation=activation, dropout=dropout),
  ]

  decoder = [
      up(256, 3, batch_norm=True, regularization=regularization, activation=activation, dropout=dropout),
      up(128, 3, batch_norm=True, regularization=regularization, activation=activation, dropout=dropout),
      up(64, 3, batch_norm=True, regularization=regularization, activation=activation, dropout=dropout),
      up(32, 3, batch_norm=True, regularization=regularization, activation=activation, dropout=dropout)
  ]

  last = tf.keras.layers.Conv2D(2, 1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), activation='softmax')

  x = inputs
  skips = []

  for l in encoder:
      x = l(x)
      skips.append(x)
      x = tf.keras.layers.MaxPool2D(2)(x)


  #skips[3] = x
  x = tf.keras.layers.Conv2D(512, 2, padding='same',
                              kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=keras.regularizers.L2(l2=regularization), activation=activation)(x)
  #skips = reversed(skips[:-1])
  skips.reverse()
  #for l, i in zip(decoder, range(1, len(skips)+1)):
  #  x = l(x)
   # print(x)
    #print(skips[len(skips) - i])
    #x = tf.keras.layers.Concatenate(-1)([x, skips[len(skips) - i]])

  for l, skip in zip(decoder, skips):
    x = l(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = keras.layers.Conv2D(3, 1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')(x)
  x = last(x)


  return tf.keras.Model(inputs=inputs, outputs=x)

