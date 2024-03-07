import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Set the path to your dataset folder
dataset_folder = "/kaggle/working/" #change if needed

# Set the path to the input folder
input_folder = "/kaggle/input/animal-image-dataset-90-different-animals"

# Load the list of animal names from the text file
with open(os.path.join(input_folder, "name of the animals.txt"), "r") as file:
    animal_names = [line.strip() for line in file.readlines()]

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Initialize K-Fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Function to create ResNet50 model for binary classification
def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# Loop through the folds
for fold, (train_index, test_index) in enumerate(kf.split(np.arange(5400))):
    print(f"\nTraining Fold {fold + 1}")

    # Create the ResNet50 model
    model = create_resnet50_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Create data generators for training, validation, and test
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_folder, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=animal_names
    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_folder, 'validation'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=animal_names
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_folder, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=animal_names,
        shuffle=False
    )

    # Train the model
    model.fit(
        train_generator,
        epochs=5,  # Adjust the number of epochs as needed
        validation_data=validation_generator,
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Accuracy for Fold {fold + 1}: {test_accuracy}")

    # Save model weights
    model.save_weights(f"resnet50_fold_{fold + 1}.weights.h5")

    # Generate classification matrix for visualization
    predictions = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred = (predictions > 0.5).astype(int).reshape(-1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)

    print(f"\nConfusion Matrix for Fold {fold + 1}:\n", confusion_matrix.numpy())


