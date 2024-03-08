import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold

# Set the path to your dataset folder
dataset_folder = "/kaggle/working/"  # change if needed

# Set the path to the input folder
input_folder = "/kaggle/input/animal-image-dataset-90-different-animals" # change if needed

# Load the list of animal names from the text file
with open(os.path.join(input_folder, "name of the animals.txt"), "r") as file:
    animal_names = [line.strip() for line in file.readlines()]

# Define image size and batch size
img_size = (224, 224)
batch_size = 32
num_classes = len(animal_names)

# Initialize K-Fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Function to create a basic CNN model for 5-class classification
def create_basic_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Change the number of units to num_classes
    return model

# Loop through the folds
for fold, (train_index, test_index) in enumerate(kf.split(np.arange(5400))):
    print(f"\nTraining Fold {fold + 1}")

    # Create the basic CNN model
    model = create_basic_cnn_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create data generators for training, validation, and test
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_folder, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=animal_names
    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_folder, 'validation'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=animal_names
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_folder, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=animal_names,
        shuffle=False
    )

    # Train the model
    history=model.fit(
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
    
    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Accuracy for Fold {fold + 1}: {test_accuracy}")

    # Save model weights
    model.save_weights(f"basic_cnn_fold_{fold + 1}.weights.h5")

    # Generate classification matrix for visualization
    predictions = model.predict(test_generator)
    y_true = test_generator.classes  # Use test_generator.classes instead of test_generator.labels
    y_pred = np.argmax(predictions, axis=1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)
    
    print(f"\nConfusion Matrix for Fold {fold + 1}:\n", confusion_matrix.numpy())
