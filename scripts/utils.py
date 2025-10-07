import os
import shutil
import random
from tqdm import tqdm

def split_dataset(original_dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits dataset from original_dataset_dir into train, val, and test folders inside output_dir.
    Assumes each class is inside its own subfolder under original_dataset_dir.

    Parameters:
    - train_ratio, val_ratio, test_ratio: Must sum to 1.
    - seed: for reproducibility.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    if os.path.exists(output_dir):
        print(f"[INFO] '{output_dir}' already exists. Skipping split.")
        return

    print("[INFO] Splitting dataset...")
    random.seed(seed)

    for phase in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, phase), exist_ok=True)

    # Loop over classes
    classes = os.listdir(original_dataset_dir)
    for cls in classes:
        cls_path = os.path.join(original_dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [img for img in os.listdir(cls_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        num_total = len(images)
        num_train = int(train_ratio * num_total)
        num_val = int(val_ratio * num_total)
        num_test = num_total - num_train - num_val

        splits = {
            'train': images[:num_train],
            'val': images[num_train:num_train+num_val],
            'test': images[num_train+num_val:]
        }

        for split_type, split_images in splits.items():
            split_dir = os.path.join(output_dir, split_type, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in tqdm(split_images, desc=f"Copying {cls} -> {split_type}", leave=False):
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)

    print(f"[INFO] Dataset split into train ({train_ratio*100}%), val ({val_ratio*100}%), test ({test_ratio*100}%) at '{output_dir}'")
