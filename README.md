<div id="top"></div>

## About The Project
This project was implemented by Adam Ezzaim, Omar Boudarka, and Mya Jamal Lahjouji for the <a href="https://www.epfl.ch/labs/mlo/machine-learning-cs-433/">CS433 Machine Learning</a>.

The project consists in implementing image segmentation on aerial images from Google Maps. This allows to categorize these images as either ”road” or ”background”.

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Clone the repo

1. Clone the repo
   ```sh
   git clone https://github.com/CS-433/ml-project-2-mao.git
    ```
2. Install the project dependencies, run the following command:
     ```
     pip install -r requirements.txt
     ```

### Running

Run the script ```run.py```, it will create a file ```submission.csv``` to be submitted in AICrowd. 

<!-- PROJECT STRUCTURE -->
## Project Structure

The project is structured as follows:

```
├── data
│   ├── training
│      ├── images
│      ├── groundtruth
│   ├── output_data_augmentation
│      ├── flipped
│      ├── rotation_45
│      ...
│   ├── test_set_images
│      ├── test_1
│      ├── test_2
│      ...
|
├── Models
|
|__ run.py
|
|__ helpers.py
|__ utils.py
|__ models.py
|__ data_augmentation.py
|__ requirement.txt
|
|__ data_exploration.ipynb
|__ basic_models.ipynb
|__ unets_no_data_augmentation.ipynb
|__ unets_with_data_augmentation.ipynb
```

### **File description** :
**run.py**: Script that loads our best model and generate a submission file for AICrowd. <br>
**helpers.py**: Contains functions used for loading and processing the data. <br>
**utils.py**: Contains utility functions such as metrics for evaluating models, functions for processing the data and functions for visualisation. <br>
**models.py**: Contains the implementation of our models described in the report.pdf. <br>
**data_augmentation.py**: Contains the functions used for augmenting the training images. <br>
**data_exploration.ipynb**: Notebook where we conducted our data exploration. <br>
**basic_models.ipynb**: Notebook where we trained our first models. <br>
**unets_no_data_augmentation.ipynb**: Notebook where we trained our U-Nets on the original training data. <br>
**unets_with_data_augmentation.ipynb**: Notebook where we trained our U-Net on the augmented training data.
<p align="right">(<a href="#top">back to top</a>)</p>



