import os
import random
import shutil

# Folder locations

RAW_DIR = "../dataset/raw"

TRAIN_DIR = "../dataset/train"

VALID_DIR = "../dataset/valid"

TEST_DIR = "../dataset/test"

# Split percentages


TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# Create destination folders

for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:

    os.makedirs(folder, exist_ok=True)

# Loop through every fish class

for fish_class in os.listdir(RAW_DIR):

    source_folder = os.path.join(RAW_DIR, fish_class)

    if not os.path.isdir(source_folder):
        continue

    # Create class folders

    os.makedirs(os.path.join(TRAIN_DIR, fish_class), exist_ok=True)

    os.makedirs(os.path.join(VALID_DIR, fish_class), exist_ok=True)

    os.makedirs(os.path.join(TEST_DIR, fish_class), exist_ok=True)

    # Read all images

    images = os.listdir(source_folder)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)

    valid_end = int(total * (TRAIN_RATIO + VALID_RATIO))

    train_images = images[:train_end]

    valid_images = images[train_end:valid_end]

    test_images = images[valid_end:]

    # Copy images
    
    for image in train_images:

        shutil.copy(

            os.path.join(source_folder, image),

            os.path.join(TRAIN_DIR, fish_class, image)

        )

    for image in valid_images:

        shutil.copy(

            os.path.join(source_folder, image),

            os.path.join(VALID_DIR, fish_class, image)

        )

    for image in test_images:

        shutil.copy(

            os.path.join(source_folder, image),

            os.path.join(TEST_DIR, fish_class, image)

        )

    print(f"{fish_class} complete")

print()

print("Dataset successfully prepared!")