import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Set the path to your dataset folder
dataset_folder = "/kaggle/working/"  # change if needed

# Set the path to the input folder
input_folder = "/kaggle/input/animal-image-dataset-90-different-animals"  # change if needed

# Load the list of animal names from the text file
with open(os.path.join(input_folder, "name of the animals.txt"), "r") as file:
    animal_names = [line.strip() for line in file.readlines()]

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Initialize K-Fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Function to create a basic CNN model for binary classification
def create_basic_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Loop through the folds
for fold, (train_index, test_index) in enumerate(kf.split(np.arange(5400))):
    print(f"\nTraining Fold {fold + 1}")

    # Create the basic CNN model
    model = create_basic_cnn_model()
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
    history = model.fit(
        train_generator,
        epochs=5,  # Adjust the number of epochs as needed
        validation_data=validation_generator,
    )

    # Plot the training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    # Plot the output of all convolutional layers
    sample_images, _ = next(train_generator)  # Get a batch of sample images
    layer_outputs = [layer.output for layer in model.layers[:6] if 'conv2d' in layer.name]  # Choose the first 6 convolutional layers
    activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

    activations = activation_model.predict(sample_images)

    # Plot the output of each convolutional layer
    for i, activation in enumerate(activations):
        plt.figure(figsize=(8, 8))
        plt.title(f'Activation of Conv2D Layer {i + 1}')
        plt.imshow(activation[0, :, :, 0], cmap='viridis')  # Display the first channel of the activation
        plt.show()

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Accuracy for Fold {fold + 1}: {test_accuracy}")

    # Save model weights
    model.save_weights(f"basic_cnn_fold_{fold + 1}.weights.h5")

    # Generate classification matrix for visualization
    predictions = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred = (predictions > 0.5).astype(int).reshape(-1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)

    print(f"\nConfusion Matrix for Fold {fold + 1}:\n", confusion_matrix.numpy())
