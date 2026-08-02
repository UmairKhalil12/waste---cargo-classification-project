import random
import shutil
from pathlib import Path

# Original dataset

SOURCE_DIR = Path("data/cargo classification dataset/cargo_classification")

# New split dataset

OUTPUT_DIR = Path("data/cargo classification dataset/cargo_split")

# Split ratios

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Make sure ratios add up to 1

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

# Make the split reproducible

random.seed(42)

# Get all class folders

class_names = [folder.name for folder in SOURCE_DIR.iterdir() if folder.is_dir()]

print(f"Classes found: {class_names}")

for class_name in class_names:
    class_source_dir = SOURCE_DIR / class_name
    # Get all image files
    images = [
        image for image in class_source_dir.iterdir()
        if image.is_file()
    ]

    # Shuffle images randomly
    random.shuffle(images)

    # Calculate split points
    train_end = int(len(images) * TRAIN_RATIO)
    val_end = train_end + int(len(images) * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create destination folders
    train_dir = OUTPUT_DIR / "train" / class_name
    val_dir = OUTPUT_DIR / "val" / class_name
    test_dir = OUTPUT_DIR / "test" / class_name

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    for image in train_images:
        shutil.copy2(image, train_dir / image.name)

    for image in val_images:
        shutil.copy2(image, val_dir / image.name)

    for image in test_images:
        shutil.copy2(image, test_dir / image.name)

    print(
        f"{class_name}: "
        f"{len(train_images)} train | "
        f"{len(val_images)} val | "
        f"{len(test_images)} test"
    )

print("\nDataset split completed!")
