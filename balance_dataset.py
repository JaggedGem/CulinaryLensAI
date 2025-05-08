import os
import shutil
import random
import numpy as np
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm

def load_class_stats():
    """Load the dataset and count instances per class"""
    # Load class names from data.yaml
    with open('datase.v1i.yolov5-obb/dataset/data.yaml', 'r') as f:
        for line in f:
            if line.startswith('names:'):
                class_names = eval(line.split('names:')[1].strip())
                break

    # Count instances per class in training set
    class_counts = {class_name: 0 for class_name in class_names}
    
    # Read through all label files
    labels_dir = Path('datase.v1i.yolov5-obb/dataset/train/labels')
    image_dir = Path('datase.v1i.yolov5-obb/dataset/train/images')
    
    # Map from class_id to class_name
    id_to_name = {i: name for i, name in enumerate(class_names)}
    
    # Store files by class for later use
    files_by_class = {class_name: [] for class_name in class_names}
    
    print("Analyzing class distribution...")
    for label_file in tqdm(list(labels_dir.glob('*.txt'))):
        image_file = image_dir / (label_file.stem.split('.')[0] + '.jpg')
        if not image_file.exists():
            image_file = image_dir / (label_file.stem.split('.')[0] + '.png')
        
        if not image_file.exists():
            continue
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        # Count classes in this file
        class_ids_in_file = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                class_name = id_to_name[class_id]
                class_counts[class_name] += 1
                class_ids_in_file.add(class_id)
        
        # Store this file for each class it contains
        for class_id in class_ids_in_file:
            files_by_class[id_to_name[class_id]].append((str(image_file), str(label_file)))
    
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name:15s}: {count:4d} instances in {len(files_by_class[class_name]):3d} images")
    
    return class_counts, files_by_class, class_names

def apply_augmentation(image, labels, augmentation_level=3):
    """Apply augmentations to image and corresponding labels"""
    h, w = image.shape[:2]
    
    # Apply various augmentations based on level (1-5, 5 being most aggressive)
    augmented_images = []
    augmented_labels = []
    
    # Preserve original
    augmented_images.append(image.copy())
    augmented_labels.append(labels.copy())
    
    # Function to adjust bounding box coordinates after augmentation
    def adjust_bbox(bbox, trans_matrix=None, flip_code=None):
        class_id, x_center, y_center, width, height = bbox
        
        if trans_matrix is not None:
            # For affine transformations, transform the bounding box center
            x_center_px = x_center * w
            y_center_px = y_center * h
            pt = np.array([[x_center_px, y_center_px]], dtype=np.float32)
            transformed_pt = cv2.transform(np.array([pt]), trans_matrix)[0][0]
            x_center = max(0, min(1, transformed_pt[0] / w))
            y_center = max(0, min(1, transformed_pt[1] / h))
        
        if flip_code is not None:
            # For flips, adjust x coordinate for horizontal flip
            if flip_code == 1:  # horizontal flip
                x_center = 1 - x_center
            # For vertical flip, adjust y coordinate
            elif flip_code == 0:  # vertical flip
                y_center = 1 - y_center
        
        return [class_id, x_center, y_center, width, height]
    
    # 1. Horizontal Flip
    if augmentation_level >= 1:
        flipped_h = cv2.flip(image, 1)  # 1 for horizontal flip
        flipped_h_labels = [adjust_bbox(bbox, flip_code=1) for bbox in labels]
        augmented_images.append(flipped_h)
        augmented_labels.append(flipped_h_labels)
    
    # 2. Vertical Flip
    if augmentation_level >= 2:
        flipped_v = cv2.flip(image, 0)  # 0 for vertical flip
        flipped_v_labels = [adjust_bbox(bbox, flip_code=0) for bbox in labels]
        augmented_images.append(flipped_v)
        augmented_labels.append(flipped_v_labels)
    
    # 3. Rotation (small angle)
    if augmentation_level >= 2:
        for angle in [10, -10]:
            if augmentation_level >= 3:
                angle *= 1.5  # More extreme for level 3+
                
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            rotated_labels = [adjust_bbox(bbox, trans_matrix=rotation_matrix) for bbox in labels]
            augmented_images.append(rotated)
            augmented_labels.append(rotated_labels)
    
    # 4. Color jittering
    if augmentation_level >= 2:
        # Brightness adjustment
        brightness = np.random.uniform(0.7, 1.3)
        color_adjusted = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        augmented_images.append(color_adjusted)
        augmented_labels.append(labels.copy())  # Labels don't change
        
        # Contrast adjustment
        if augmentation_level >= 3:
            contrast = np.random.uniform(0.7, 1.3)
            mean = np.mean(image)
            color_adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=(1-contrast)*mean)
            augmented_images.append(color_adjusted)
            augmented_labels.append(labels.copy())  # Labels don't change
    
    # 5. Add noise
    if augmentation_level >= 3:
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        augmented_images.append(noisy)
        augmented_labels.append(labels.copy())  # Labels don't change
    
    # 6. Scale/zoom
    if augmentation_level >= 4:
        for scale in [0.8, 1.2]:
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            scaled = cv2.resize(image, (scaled_w, scaled_h))
            
            if scale < 1:  # Zoom out, need to pad
                pad_w, pad_h = (w - scaled_w) // 2, (h - scaled_h) // 2
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[pad_h:pad_h+scaled_h, pad_w:pad_w+scaled_w] = scaled
                
                # Adjust labels
                scaled_labels = []
                for bbox in labels:
                    class_id, x_center, y_center, width, height = bbox
                    # Scale and shift coordinates
                    x_center = x_center * scale + pad_w / w
                    y_center = y_center * scale + pad_h / h
                    width *= scale
                    height *= scale
                    scaled_labels.append([class_id, x_center, y_center, width, height])
                
                augmented_images.append(padded)
                augmented_labels.append(scaled_labels)
            else:  # Zoom in, need to crop
                crop_w, crop_h = (scaled_w - w) // 2, (scaled_h - h) // 2
                cropped = scaled[crop_h:crop_h+h, crop_w:crop_w+w]
                
                # Adjust labels
                scaled_labels = []
                for bbox in labels:
                    class_id, x_center, y_center, width, height = bbox
                    # Scale and shift coordinates
                    x_center = (x_center * scale - crop_w / w) / scale
                    y_center = (y_center * scale - crop_h / h) / scale
                    # Skip boxes that are now outside the image
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1:
                        width *= scale
                        height *= scale
                        scaled_labels.append([class_id, x_center, y_center, width, height])
                
                if scaled_labels:  # Only add if we have valid boxes
                    augmented_images.append(cropped)
                    augmented_labels.append(scaled_labels)
    
    return augmented_images, augmented_labels

def balance_dataset(target_count=200):
    """Balance the dataset by generating synthetic data for underrepresented classes"""
    class_counts, files_by_class, class_names = load_class_stats()
    
    # Find max count to determine how much augmentation each class needs
    max_count = max(class_counts.values())
    if target_count < max_count:
        target_count = max_count
    
    # Create augmentation directory
    aug_image_dir = Path('datase.v1i.yolov5-obb/dataset/train/images_augmented')
    aug_label_dir = Path('datase.v1i.yolov5-obb/dataset/train/labels_augmented')
    os.makedirs(aug_image_dir, exist_ok=True)
    os.makedirs(aug_label_dir, exist_ok=True)
    
    # Copy original files first
    print("\nCopying original files...")
    original_image_dir = Path('datase.v1i.yolov5-obb/dataset/train/images')
    original_label_dir = Path('datase.v1i.yolov5-obb/dataset/train/labels')
    
    for image_file in tqdm(list(original_image_dir.glob('*.*'))):
        if image_file.suffix in ['.jpg', '.jpeg', '.png']:
            shutil.copy(image_file, aug_image_dir / image_file.name)
    
    for label_file in original_label_dir.glob('*.txt'):
        shutil.copy(label_file, aug_label_dir / label_file.name)
    
    # Generate augmentations for each underrepresented class
    for i, class_name in enumerate(class_names):
        current_count = class_counts[class_name]
        if current_count >= target_count:
            continue
            
        needed_count = target_count - current_count
        print(f"\nGenerating {needed_count} synthetic examples for {class_name}...")
        
        # Determine augmentation level based on how much augmentation is needed
        if needed_count > current_count * 5:
            aug_level = 5  # Extreme augmentation
        elif needed_count > current_count * 3:
            aug_level = 4  # Heavy augmentation
        elif needed_count > current_count * 2:
            aug_level = 3  # Moderate augmentation
        elif needed_count > current_count:
            aug_level = 2  # Light augmentation
        else:
            aug_level = 1  # Minimal augmentation
            
        print(f"Using augmentation level {aug_level}")
        
        # Get files containing this class
        class_files = files_by_class[class_name]
        if not class_files:
            print(f"Warning: No files found for class {class_name}")
            continue
            
        # Repeat until we reach target count
        generated_count = 0
        while generated_count < needed_count:
            # Pick a random file containing this class
            image_path, label_path = random.choice(class_files)
            
            # Read image and labels
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Parse labels
            labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    labels.append([class_id, x_center, y_center, width, height])
            
            # Apply augmentations
            augmented_images, augmented_labels = apply_augmentation(image, labels, aug_level)
            
            # Save augmented images and labels
            base_name = Path(image_path).stem
            for j, (aug_image, aug_label) in enumerate(zip(augmented_images, augmented_labels)):
                if generated_count >= needed_count:
                    break
                    
                # Skip the first one as it's the original
                if j == 0:
                    continue
                
                # Save image
                aug_image_path = aug_image_dir / f"{base_name}_aug_{class_name}_{j}.jpg"
                cv2.imwrite(str(aug_image_path), aug_image)
                
                # Save label
                aug_label_path = aug_label_dir / f"{base_name}_aug_{class_name}_{j}.txt"
                with open(aug_label_path, 'w') as f:
                    for bbox in aug_label:
                        f.write(f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
                
                generated_count += 1
                if generated_count % 10 == 0:
                    print(f"Generated {generated_count}/{needed_count}")
    
    # Update data.yaml to use the augmented dataset
    data_yaml_path = 'datase.v1i.yolov5-obb/dataset/data_augmented.yaml'
    with open('datase.v1i.yolov5-obb/dataset/data.yaml', 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    data_yaml['train'] = 'train/images_augmented'
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"\nDataset balancing complete!")
    print(f"New data.yaml file created at: {data_yaml_path}")
    print("Use this file for training with more balanced data.")
    
    return data_yaml_path

if __name__ == "__main__":
    balanced_data_yaml = balance_dataset(target_count=200)
    print(f"Balanced dataset yaml: {balanced_data_yaml}") 