import os
import shutil
from pathlib import Path
import numpy as np

def convert_obb_to_yolo(obb_path, output_path, img_width=640, img_height=640):
    """
    Convert OBB format to YOLO format
    OBB format: x1 y1 x2 y2 x3 y3 x4 y4 class difficult
    YOLO format: class_id center_x center_y width height
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Process all label files
    count = 0
    for file in os.listdir(obb_path):
        if not file.endswith('.txt'):
            continue
        
        with open(os.path.join(obb_path, file), 'r') as f:
            lines = f.readlines()
        
        yolo_lines = []
        for line in lines:
            parts = line.strip().split()
            
            # Skip invalid lines
            if len(parts) < 9:
                continue
                
            # Get coordinates and class
            try:
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                class_name = parts[8]
                class_id = class_names.index(class_name)
                
                # Calculate center, width, height
                x_coords = [x1, x2, x3, x4]
                y_coords = [y1, y2, y3, y4]
                
                # Calculate the center using the average of coordinates
                center_x = sum(x_coords) / 4 / img_width
                center_y = sum(y_coords) / 4 / img_height
                
                # Find the width and height from the bounding box
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                width = (max_x - min_x) / img_width
                height = (max_y - min_y) / img_height
                
                # Ensure values are within range [0, 1]
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            except (ValueError, IndexError) as e:
                print(f"Error processing line '{line.strip()}' in file {file}: {e}")
                continue
        
        # Save the converted file
        if yolo_lines:
            with open(os.path.join(output_path, file), 'w') as f:
                f.write('\n'.join(yolo_lines))
            count += 1
    
    print(f"Converted {count} label files to YOLO format")

# Get class names from data.yaml
with open('datase.v1i.yolov5-obb/dataset/data.yaml', 'r') as f:
    for line in f:
        if line.startswith('names:'):
            class_names = eval(line.split('names:')[1].strip())
            break

# Directory paths
train_obb_labels = 'datase.v1i.yolov5-obb/dataset/train/labelTxt'
valid_obb_labels = 'datase.v1i.yolov5-obb/dataset/valid/labelTxt'
test_obb_labels = 'datase.v1i.yolov5-obb/dataset/test/labelTxt'

train_yolo_labels = 'datase.v1i.yolov5-obb/dataset/train/labels'
valid_yolo_labels = 'datase.v1i.yolov5-obb/dataset/valid/labels'
test_yolo_labels = 'datase.v1i.yolov5-obb/dataset/test/labels'

# Convert labels
print("Converting train labels...")
if os.path.exists(train_obb_labels):
    convert_obb_to_yolo(train_obb_labels, train_yolo_labels)

print("Converting validation labels...")
if os.path.exists(valid_obb_labels):
    convert_obb_to_yolo(valid_obb_labels, valid_yolo_labels)

print("Converting test labels...")
if os.path.exists(test_obb_labels):
    convert_obb_to_yolo(test_obb_labels, test_yolo_labels)

print("Label conversion complete!") 