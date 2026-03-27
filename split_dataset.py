import os
import shutil
import random
from glob import glob
from tqdm import tqdm

source_dir = 'masterCategory'  # where your current class folders are
output_base = 'SportsClassification'  # target root dir
splits = ['train', 'test', 'validate']
split_ratio = [0.7, 0.2, 0.1]  # 70% train, 20% test, 10% val

# Make split folders
for split in splits:
    for class_name in os.listdir(source_dir):
        os.makedirs(os.path.join(output_base, split, class_name), exist_ok=True)

# Loop through each class and split
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    images = glob(os.path.join(class_path, '*.jpg'))
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratio[0])
    test_end = train_end + int(total * split_ratio[1])

    split_images = {
        'train': images[:train_end],
        'test': images[train_end:test_end],
        'validate': images[test_end:]
    }

    for split in splits:
        for img_path in tqdm(split_images[split], desc=f"Copying {split}/{class_name}"):
            dst = os.path.join(output_base, split, class_name, os.path.basename(img_path))
            shutil.copy(img_path, dst)

print("✅ Dataset split into train/test/validate.")
