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



## Usage

1. **Dataset Preparation:**
   - Execute `prepare_dataset.py` to organize the dataset for 3-fold cross-validation.

2. **Model Development & Training and Evaluation:**
   - Run `One_vs_Rest.py` to create a CNN model, train the model on prepared datasets using one-vs-rest classification and generate confusion matrices.
   - Run `5_class.py` to create a CNN model, train the model on prepared datasets using 5-class classification and generate confusion matrices.
   - Use the `.py` under the corresponding folder for the required CNN:
        - The ones under `one_vs_rest_ResNet50` and `5-class_classification_ResNet50` folder is for **ResNet50** (An existing CNN with 50 layers)

3. **Convolutional Layer Visualization:**
   - Execute `visualize_conv_layers.py` to visualize the output of all convolutional layers.



## Requirements

- Python 3.x
- TensorFlow (version specified in `requirements.txt`)
- Matplotlib
- Scikit-learn
- GPU: P100



## Acknowledgments

- The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?resource=download).
- Inspiration for the custom CNN model design was drawn from [Link](https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5).


Feel free to explore, experiment, and contribute to this repository!


**Note**: This repo is currently under construction.
