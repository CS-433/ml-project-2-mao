import cv2
import os
import matplotlib.image as mpimg
import numpy as np

def load_images_from_folder(folder, num_images=100, is_gray=False):
    """
    Load images from a folder.

    Args:
        folder (str): Path to the folder containing the images.
        num_images (int, optional): Number of images to load. Defaults to 100.
        is_gray (bool, optional): Whether to load the images in grayscale. Defaults to False.

    Returns:
        list: List of loaded images.
    """
    images = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        imageid = imageid + ".png"
        img_path = os.path.join(folder, imageid)
        if is_gray:
            img = mpimg.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = mpimg.imread(img_path)

        if img is not None:
            images.append(img)
    print(len(images))
    return images

def create_folder_if_not_exists(folder):
    """
    Create a folder if it doesn't already exist.

    Args:
        folder (str): The path of the folder to be created.

    Returns:
        None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def translate_image(image, x, y):
    """
    Translates the given image by the specified amount in the x and y directions.

    Parameters:
    image (numpy.ndarray): The input image.
    x (int): The amount of translation in the x direction.
    y (int): The amount of translation in the y direction.

    Returns:
    numpy.ndarray: The translated image.
    """
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

def augment_images_no_save(satellite_images, groundtruth_images):
    """
    Augments a list of satellite images and groundtruth images by performing rotations, flips, translations, and zooms.
    
    Args:
        satellite_images (list): List of satellite images.
        groundtruth_images (list): List of groundtruth images.
        
    Returns:
        numpy.ndarray: Augmented satellite images as a numpy array.
        numpy.ndarray: Augmented groundtruth images as a numpy array.
        
    Raises:
        ValueError: If the number of satellite images and groundtruth images is not the same.
    """
    if len(satellite_images) != len(groundtruth_images):
        raise ValueError("The number of satellite images and groundtruth images must be the same")
    imgs = []
    gt_imgs = []

    angles = [45,90,135, 180, 225,270, 315]  # List of angles for rotations
    translations = [(120, 120), (-120, -120)]  # Translations (right-down and left-up)
    zoom_factors = [1.25, 1.35] # List of zooms

    for i in range(1, len(satellite_images) + 1):
        satellite_img = satellite_images[i - 1]
        groundtruth_img = groundtruth_images[i - 1]
        
        # Rotate and save
        for angle in angles:
            M = cv2.getRotationMatrix2D((satellite_img.shape[1] / 2, satellite_img.shape[0] / 2), angle, 1)
            M_gt = cv2.getRotationMatrix2D((groundtruth_img.shape[1] / 2, groundtruth_img.shape[0] / 2), angle, 1)
            rotated_satellite = cv2.warpAffine(satellite_img, M, (satellite_img.shape[1], satellite_img.shape[0]))
            rotated_groundtruth = cv2.warpAffine(groundtruth_img, M_gt, (groundtruth_img.shape[1], groundtruth_img.shape[0]),flags=cv2.INTER_NEAREST)

            imgs.append(rotated_satellite)
            gt_imgs.append(rotated_groundtruth)

        # Flip and save
        flipped_satellite = cv2.flip(satellite_img, 1)  # Horizontal flip
        flipped_groundtruth = cv2.flip(groundtruth_img, 1)

        imgs.append(flipped_satellite)
        gt_imgs.append(flipped_groundtruth)
        
        # Translate and save
        for x, y in translations:
            translated_satellite = translate_image(satellite_img, x, y)
            #translated_groundtruth = translate_image(groundtruth_img, x, y)
            
            translation_matrix_gt = np.float32([[1, 0, x], [0, 1, y]])
            translated_groundtruth = cv2.warpAffine(groundtruth_img, translation_matrix_gt, (groundtruth_img.shape[1], groundtruth_img.shape[0]), flags=cv2.INTER_NEAREST)  

            imgs.append(translated_satellite)
            gt_imgs.append(translated_groundtruth)

        for factor in zoom_factors:
            # Zoom
            zoomed_satellite = zoom_image(satellite_img, factor)
            zoomed_groundtruth = zoom_image(groundtruth_img, factor)

            imgs.append(zoomed_satellite)
            gt_imgs.append(zoomed_groundtruth)
            
    return np.asarray(imgs), np.asarray(gt_imgs)

def zoom_image(image, factor):
    """
    Zooms in or out on an image.

    Args:
        image (np.array): The image to be zoomed.
        factor (float): The zoom factor. Values < 1 will zoom out, values > 1 will zoom in.

    Returns:
        np.array: The zoomed image.
    """
    if factor <= 1:
        return image  # No zoom if factor is 1 or less

    height, width = image.shape[:2]
    new_height, new_width = int(height / factor), int(width / factor)
    startx = width // 2 - (new_width // 2)
    starty = height // 2 - (new_height // 2)    
    cropped_image = image[starty:starty + new_height, startx:startx + new_width]
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
    return zoomed_image

def one_hot(gts): 
  foreground_threshold = 0.25

  # Initialize labels array
  labels = np.zeros((gts.shape[0], gts.shape[1], gts.shape[2], 2), dtype=np.float32)

  # Apply threshold to create a boolean mask for foreground
  foreground_mask = gts > foreground_threshold

  # Use the mask to assign [0, 1] to foreground and [1, 0] to background
  labels[..., 0] = 1 - foreground_mask  # Background
  labels[..., 1] = foreground_mask      # Foreground

  return labels

def augment_images(satellite_images, groundtruth_images, output_folder):
    """
    Augments satellite images and corresponding groundtruth images by performing rotations, flips, and translations.
    
    Args:
        satellite_images (list): List of satellite images.
        groundtruth_images (list): List of corresponding groundtruth images.
        output_folder (str): Path to the output folder where augmented images will be saved.
        
    Raises:
        ValueError: If the number of satellite images and groundtruth images is not the same.
    """
    if len(satellite_images) != len(groundtruth_images):
        raise ValueError("The number of satellite images and groundtruth images must be the same")

    angles = [45,90,135, 180, 225,270, 315]  # List of angles for rotations
    translations = [(120, 120), (-120, -120)]  # Translations (right-down and left-up)
    for i in range(1, len(satellite_images) + 1):
        satellite_img = satellite_images[i - 1]
        groundtruth_img = groundtruth_images[i - 1]
        
        # Rotate and save
        for angle in angles:
            M = cv2.getRotationMatrix2D((satellite_img.shape[1] / 2, satellite_img.shape[0] / 2), angle, 1)
            M_gt = cv2.getRotationMatrix2D((groundtruth_img.shape[1] / 2, groundtruth_img.shape[0] / 2), angle, 1)
            rotated_satellite = cv2.warpAffine(satellite_img, M, (satellite_img.shape[1], satellite_img.shape[0]))
            rotated_groundtruth = cv2.warpAffine(groundtruth_img, M_gt, (groundtruth_img.shape[1], groundtruth_img.shape[0]),flags=cv2.INTER_NEAREST)

            rotation_folder = os.path.join(output_folder, f"rotation_{angle}")
            create_folder_if_not_exists(rotation_folder)
            satellite_folder = os.path.join(rotation_folder, "images")
            groundtruth_folder = os.path.join(rotation_folder, "groundtruth")
            create_folder_if_not_exists(satellite_folder)
            create_folder_if_not_exists(groundtruth_folder)

            cv2.imwrite(os.path.join(satellite_folder, f"rotated_satellite_{i:03d}.png"), rotated_satellite)
            cv2.imwrite(os.path.join(groundtruth_folder, f"rotated_groundtruth_{i:03d}.png"), rotated_groundtruth)

        # Flip and save
        flipped_satellite = cv2.flip(satellite_img, 1)  # Horizontal flip
        flipped_groundtruth = cv2.flip(groundtruth_img, 1)

        flip_folder = os.path.join(output_folder, "flipped")
        create_folder_if_not_exists(flip_folder)

        satellite_folder = os.path.join(flip_folder, "images")
        groundtruth_folder = os.path.join(flip_folder, "groundtruth")
        create_folder_if_not_exists(satellite_folder)
        create_folder_if_not_exists(groundtruth_folder)

        cv2.imwrite(os.path.join(satellite_folder, f"flipped_satellite_{i:03d}.png"), flipped_satellite)
        cv2.imwrite(os.path.join(groundtruth_folder, f"flipped_groundtruth_{i:03d}.png"), flipped_groundtruth)
               
        # Translate and save
        for x, y in translations:
            translated_satellite = translate_image(satellite_img, x, y)
            #translated_groundtruth = translate_image(groundtruth_img, x, y)
            
            translation_matrix_gt = np.float32([[1, 0, x], [0, 1, y]])
            translated_groundtruth = cv2.warpAffine(groundtruth_img, translation_matrix_gt, (groundtruth_img.shape[1], groundtruth_img.shape[0]), flags=cv2.INTER_NEAREST)  


            translation_folder = os.path.join(output_folder, f"translation_{x}_{y}")
            create_folder_if_not_exists(translation_folder)

            satellite_folder = os.path.join(translation_folder,"images")
            groundtruth_folder = os.path.join(translation_folder, "groundtruth")
            create_folder_if_not_exists(satellite_folder)
            create_folder_if_not_exists(groundtruth_folder)

            cv2.imwrite(os.path.join(satellite_folder, f"translated_{x}_{y}_satellite_{i:03d}.png"), translated_satellite)
                                         
            cv2.imwrite(os.path.join(groundtruth_folder, f"translated_{x}_{y}_groundtruth_{i:03d}.png"), translated_groundtruth)





