# Helper functions
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import tensorflow as tf


NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 100
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

############################################ LOADING THE DATA #########################################################

def load_image(infilename):
    """
    Load an image from the specified file path.

    Parameters:
        infilename (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image data.
    """
    data = mpimg.imread(infilename)
    return data

def extract_data_for_unet(filename, num_images):
    """
    Extracts data for U-Net model from a given directory.
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0, 1].

    Args:
        filename (str): The base filename of the images.
        num_images (int): The number of images to extract.

    Returns:
        np.ndarray: An array of images.
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    print("Loaded " + str(len(imgs)) + " images.")
    return np.asarray(imgs)

def extract_labels_for_unet(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index].

    Args:
        filename (str): The path to the directory containing the image files.
        num_images (int): The number of images to extract labels from.

    Returns:
        numpy.ndarray: The labels in a 1-hot matrix of shape [num_images, height, width, num_classes].

    Raises:
        FileNotFoundError: If any of the image files specified by the filename and imageid do not exist.
    """
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            raise FileNotFoundError("File " + image_filename + " does not exist")
    print("Loaded " + str(len(gt_imgs)) + " groundtruth images.")

    num_classes = 2

    width = gt_imgs[0].shape[0]
    height = gt_imgs[0].shape[1]

    gt_imgs_np = np.asarray(gt_imgs)
    print("Shape of GT images : ",gt_imgs_np.shape)

    labels = np.zeros((len(gt_imgs), height, width, num_classes), dtype=np.float32)

    for i, img in enumerate(gt_imgs):
        for h in range(height):
            for w in range(width):
                labels[i, h, w] = value_to_class_for_unet(img[h, w])

    return labels


def value_to_class_for_unet(v):
    """
    Converts a value to a one-hot label for the UNet model.

    Args:
        v (float): The input value.

    Returns:
        list: A one-hot label representing the class. If the value is greater than the foreground threshold, 
              the label is [0, 1] (foreground). Otherwise, the label is [1, 0] (background).
    """
    foreground_threshold = 0.25
    if v > foreground_threshold:  # Foreground
        return [0, 1]
    else:  # Background
        return [1, 0]
    
def value_to_class(v):
    """
    Converts a pixel value to a class label.

    Parameters:
    v (numpy.ndarray): The pixel value.

    Returns:
    list: The class label as a one-hot encoded list.
    """
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]
    
def extract_data(filename, num_images):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0, 1].

    Args:
        filename (str): The base filename of the images.
        num_images (int): The number of images to extract.

    Returns:
        numpy.ndarray: A 4D tensor containing the extracted image patches.
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    print("Loaded " + str(len(imgs)) + " images.")

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    print("Extracted " + str(len(data)) + " patches of size " + str(IMG_PATCH_SIZE) + ".")

    return np.asarray(data)


def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")
    print("Loaded " + str(len(gt_imgs)) + " groundtruth images.")

    num_images = len(gt_imgs)
    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = np.asarray(
        [value_to_class(np.mean(data[i])) for i in range(len(data))]
    )
    print("Extracted " + str(len(labels)) + " labels.")
    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)

def get_images(filename,num_images):
    """
    Function to extract the training images (or mask) from the appropriate folder into a list of tensors. 
    Arguments: 
    filename: "training_path_from_which_to_extract"
    num_images: the number of images (or masks) we want to extract from the file (max 100)
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")
    return imgs


############################################ PROCESS THE DATA FOR MODELS #########################################################
def img_crop(im, w, h):
    """
    Crop an image into patches of specified width and height.

    Parameters:
    im (numpy.ndarray): The input image.
    w (int): The width of each patch.
    h (int): The height of each patch.

    Returns:
    list: A list of image patches.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

def img_float_to_uint8(img):
    """
    Converts a floating-point image to an 8-bit unsigned integer image.

    Parameters:
    img (numpy.ndarray): The input image as a numpy array.

    Returns:
    numpy.ndarray: The converted image as a numpy array of type uint8.
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def patches(imgs, gt_imgs, patch_size=16) :
    """
    Extracts patches from input images and ground truth images.

    Args:
        imgs (list): List of input images.
        gt_imgs (list): List of ground truth images.
        patch_size (int, optional): Size of the patches. Defaults to 16.

    Returns:
        tuple: A tuple containing two arrays - img_patches and gt_patches.
               img_patches: Array of image patches.
               gt_patches: Array of ground truth patches.
    """
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(len(imgs))]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(len(gt_imgs))]

    # Linearize list of patches
    img_patches = np.asarray(
        [
            img_patches[i][j]
            for i in range(len(img_patches))
            for j in range(len(img_patches[i]))
        ]
    )
    gt_patches = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    return img_patches, gt_patches

# assign a label to a patch
def patch_to_label(patch):
    """
    Converts a patch to a label based on the mean value of the patch.

    Parameters:
    patch (numpy.ndarray): The patch to be converted to a label.

    Returns:
    int: The label of the patch. Returns 1 if the mean value of the patch is greater than foreground_threshold, otherwise returns 0.
    """
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

############################################ SUBMISSION ##############################################################

def get_images_test(filename, num_images):
    """
    Function to extract the test images from the appropriate folder into a list of tensors. 
    Arguments: 
    filename: "test_path_from_which_to_extract"
    num_images: the number of images we want to extract from the file (max 50)
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "test_%.1d" % i + "/test_%.1d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")
    imgs=tf.stack(imgs)
    return imgs

def mask_to_submission_strings(im,im_number):
    """
    Reads a single image and outputs the strings that should go into the submission file
    Used in the masks_to_submission function
    """
    patch_size = 16
    for j in range(0, im.shape
                   [1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(im_number, j, i, label))


def masks_to_submission(submission_filename, imgs):
    """
    Converts predicted masks into a submission file
    Submission filename: .csv type
    imgs: list of all predicted masks
    Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for img_num in range(len(imgs)):
          f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(imgs[img_num],img_num+1))