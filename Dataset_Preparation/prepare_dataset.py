import os
import shutil
import random

# Set the paths
input_folder = "/kaggle/input/animal-image-dataset-90-different-animals" #change as needed
output_folder = "/kaggle/working/" #change as needed

# Create train, test, and validation folders
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
val_folder = os.path.join(output_folder, "validation")

for folder in [train_folder, test_folder, val_folder]:
    os.makedirs(folder, exist_ok=True)

# Read the list of animals from the text file
animals_list_path = os.path.join(input_folder, "name of the animals.txt")
with open(animals_list_path, 'r') as file:
    animals_list = file.read().splitlines()

# Set the ratio for train, test, and validation
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# Loop through each animal folder and organize images
for animal in animals_list:
    animal_folder = os.path.join(input_folder, "animals", "animals", animal)
    output_train_folder = os.path.join(train_folder, animal)
    output_test_folder = os.path.join(test_folder, animal)
    output_val_folder = os.path.join(val_folder, animal)

    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)
    os.makedirs(output_val_folder, exist_ok=True)

    # Get the list of images for the current animal
    images = os.listdir(animal_folder)
    random.shuffle(images)

    # Split the images into train, test, and validation sets
    train_size = int(len(images) * train_ratio)
    test_size = int(len(images) * test_ratio)

    train_set = images[:train_size]
    test_set = images[train_size:train_size + test_size]
    val_set = images[train_size + test_size:]

    # Copy images to the corresponding output folders
    for image in train_set:
        shutil.copy(os.path.join(animal_folder, image), os.path.join(output_train_folder, image))

    for image in test_set:
        shutil.copy(os.path.join(animal_folder, image), os.path.join(output_test_folder, image))

    for image in val_set:
        shutil.copy(os.path.join(animal_folder, image), os.path.join(output_val_folder, image))

print("Dataset organized successfully.")
