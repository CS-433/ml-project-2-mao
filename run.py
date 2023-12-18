"""
This script loads a pre-trained U-Net model and uses it to make predictions on test images.
The predicted masks are then converted into a submission file.

Functions:
- get_images_test(root_dir, num_images): Loads test images from the specified directory.
- build_unet(dropout, activation, regularization): Builds a U-Net model with the specified parameters.
- TverskyLoss(y_true, y_pred): Computes the Tversky loss between the true and predicted masks.
- masks_to_submission(filename, pred): Converts predicted masks into a submission file.

Variables:
- root_dir: The root directory of the test images.
- image_dir: The directory containing the test images.
- gt_dir: The directory containing the ground truth masks.
- test_imgs: The loaded test images.
- checkpoint_path: The path to the pre-trained model weights.
- model_A64RD: The U-Net model.
- predictions: The predicted masks.
- pred: The predicted class labels for each pixel.

Usage:
1. Set the root_dir to the appropriate path.
2. Specify the number of test images to load in the get_images_test function.
3. Set the checkpoint_path variable to the path of the pre-trained model weights.
4. Customize the U-Net model architecture in the build_unet function.
5. Run the script to generate the submission file.
"""

from helpers import *
from models import *


root_dir = "data/test_set_images/"
test_imgs = get_images_test(root_dir, 50)

checkpoint_path = 'Models/from_A32RDv2'
model = build_unet(dropout = 0.1, activation='relu', regularization=1e-6)
model.compile(optimizer='adam',
                loss=TverskyLoss,
                metrics=['accuracy'])

model.load_weights(checkpoint_path)

predictions = model.predict(test_imgs)

pred = np.argmax(predictions, 3)

masks_to_submission("submission.csv", pred)