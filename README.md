# SRIP_Selection_Task



# Animal Classification Repository

## Overview

This repository contains code and documentation for a comprehensive analysis of animal image classification using various machine learning models. The primary goal is to explore different classification scenarios, starting with one-vs-rest, progressing to binary classification, and finally addressing a 5-class classification problem. The models will be evaluated using classification matrices to assess their performance.


## Contents

1. **Dataset Preparation:**
   - The dataset comprises 90 different animal images.
   - Initial organization for one-vs-rest classification.
   - Transformation for binary classification.
   - Restructuring for 5-class classification.
   - Utilization of 3-fold cross-validation to evaluate model performance.

2. **Model Development:**
   - Building a custom Convolutional Neural Network (CNN) model.
   - Avoiding the use of existing architectures like ResNet or DenseNet to create a unique model.

3. **Training and Evaluation:**
   - Training the model on prepared datasets for both one-vs-rest and 5-class classification.
   - Generating classification matrices for a comprehensive evaluation of model performance.

4. **Convolutional Layer Visualization:**
   - Plotting the output of all convolutional layers.
   - Discussion on insights gained from automatically created features.
  
5. **Documentation (.txt files):**
   - These files are for the user's reference.



## Usage

1. **Dataset Preparation:**
   - Execute `prepare_dataset.py` to organize the dataset for 3-fold cross-validation.

2. **Model Development & Training and Evaluation:**
   - Run `One_vs_Rest.py` to create a CNN model, train the model on prepared datasets using one-vs-rest classification and generate confusion matrices.
   - Run `5_class.py` to create a CNN model, train the model on prepared datasets using 5-class classification and generate confusion matrices.
   - Use the `.py` under the corresponding folder for the required CNN:
        - The ones under `one_vs_rest_ResNet50` and `5-class_classification_ResNet50` folder is for **ResNet50** (An existing CNN with 50 layers)
        - The ones under `Custom_CNN` folder uses a **custom CNN** of 9 layers (more details in `Documentation_CustomCNN.txt`)

3. **Convolutional Layer Visualization:**
   - Add `visualize_conv_layers.py` to the corresponding `.py` to visualize the output of convolutional layers.



## Requirements 
(version specified in `requirements.txt`)

- Python 
- TensorFlow 
- Matplotlib
- Scikit-learn


## Experiment Details

This was carried out in Kaggle with the GPU P100 accelerator (for faster computation).
Check out the Kaggle Notebook here: [Experiment](https://www.kaggle.com/devoditac/srip-iit-gandhinagar)


## Acknowledgments

- The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?resource=download).
- Inspiration for the custom CNN model design was drawn from [this article (Towards Data Science)](https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5).


Feel free to explore, experiment, and contribute to this repository!


**Note**: This repo is currently under construction.
