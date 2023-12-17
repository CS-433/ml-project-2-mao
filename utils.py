import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.metrics import f1_score
from helpers import *

############################################ METRICS #########################################################

def evaluating_model_f1(tensor_images, tensor_gt, trained_model):
    """
    This function is used to evaluate a model with the F1-score.
    Arguments:
    tensor_images: a 4D tensor with images of shape (num_image, image_width, image_height, 3)
    tensor_gt: a 4D tensor with masks of shape (num_mask, mask_width, image_height, 2)
    """
    preds=trained_model.predict(tensor_images)

    preds=np.argmax(preds,3)
    one_hot_masks_1=np.argmax(tensor_gt,3)

    preds_flat=preds.flatten()
    one_hot_masks_flat=one_hot_masks_1.flatten()

    return f1_score(one_hot_masks_flat,preds_flat)

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0
        * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        / predictions.shape[0]
    )

############################################ DATA & VISUALIZATION #########################################################


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """
    # set seed
    np.random.seed(seed)

    # split the data based on the given ratio:
    N_train = int(np.floor(ratio * len(y)))
    indices = np.random.permutation(np.arange(len(y)))

    x_tr = x[indices[:N_train]]
    x_te = x[indices[N_train:]]
    y_tr = y[indices[:N_train]]
    y_te = y[indices[N_train:]]

    return x_tr, x_te, y_tr, y_te

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j : j + w, i : i + h] = l
            idx = idx + 1
    return array_labels

def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def get_prediction_conv(img, model):
  img = np.asarray([img])
  img = tf.stack(img)
  img = tf.cast(img, tf.float32)
  pred = model.predict(img)
  pred = np.argmax(pred, axis=2)
  img_prediction = label_to_img(
            img.shape[0],
            img.shape[1],
            IMG_PATCH_SIZE,
            IMG_PATCH_SIZE,
            pred,
        )


  return img_prediction


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay_conv(filename, image_idx, model):
  imageid = "satImage_%.3d" % image_idx
  image_filename = filename + "images/" + imageid + ".png"
  img = mpimg.imread(image_filename)
  gt_filename = filename + "groundtruth/" + imageid + ".png"
  gt = mpimg.imread(gt_filename)

  img_prediction = get_prediction_conv(img, model)
  oimg = make_img_overlay(img, img_prediction)
  pimg = make_img_overlay(img, gt)

  return oimg, pimg

def show_predictions_conv(model, num_images):
    fig, axs = plt.subplots(1, 2*num_images, figsize=(15*2, 5))
    for i in range(1,num_images+1):
        img_idx = i * 15
        filename = "/content/drive/Shareddrives/ML_PROJET2/Project_2/data/training/"
        pred_img, gt_img = get_prediction_with_overlay_conv(filename, img_idx, model)

        axs[2*i - 2].imshow(pred_img)
        axs[2*i - 2].set_title(f"Image {i} Prediction")
        axs[2*i - 1].imshow(gt_img)
        axs[2*i - 1].set_title(f"Image {i} Groundtruth")

    plt.show()

def get_prediction_unet(img, model):
  img = np.asarray([img])
  pred = model.predict(img)
  pred = np.argmax(pred, 3)
  return pred[0]


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay_unet(filename, image_idx, model):
  imageid = "satImage_%.3d" % image_idx
  image_filename = filename + "images/" + imageid + ".png"
  img = mpimg.imread(image_filename)
  gt_filename = filename + "groundtruth/" + imageid + ".png"
  gt = mpimg.imread(gt_filename)

  img_prediction = get_prediction_unet(img, model)
  oimg = make_img_overlay(img, img_prediction)
  pimg = make_img_overlay(img, gt)

  return oimg, pimg

def my_show_predictions1(model, num_images):
    fig, axs = plt.subplots(1, 2*num_images, figsize=(15*num_images, 5))
    for i in range(1,num_images+1):
        img_idx = i*20
        filename = "/content/drive/Shareddrives/ML_PROJET2/Project_2/data/training/"
        pred_img, gt_img = get_prediction_with_overlay_unet(filename, img_idx, model)

        axs[2*i - 2].imshow(pred_img)
        axs[2*i - 2].set_title(f"Image {i} Prediction")
        axs[2*i - 1].imshow(gt_img)
        axs[2*i - 1].set_title(f"Image {i} Groundtruth")

    plt.show()


def build_model(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten())

  num_layers = hp.Int('num_layers', min_value = 1, max_value = 3, step=1)

  for i in range(1, num_layers+1) :
    model.add(
        tf.keras.layers.Dense(units=hp.Int(f'units layer {i}', min_value=32, max_value=512, step=64), kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu')
    )

  learning = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
  model.add(tf.keras.layers.Dense(2, activation='softmax', kernel_initializer=tf.keras.initializers.HeNormal()))

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= learning),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.F1Score(), 'accuracy'])
  return model