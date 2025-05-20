import os
import argparse
from pathlib import Path
import tarfile
import shutil
from tqdm import tqdm
import urllib.request

def parse_args():
    parser = argparse.ArgumentParser(description='Download and prepare Food-101 dataset')
    parser.add_argument('--data-dir', type=str, default='./food-101', help='Directory to store the dataset')
    parser.add_argument('--download-only', action='store_true', help='Only download the dataset, don\'t prepare it')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare the dataset, don\'t download it')
    return parser.parse_args()

def download_file(url, save_path):
    """Download file from URL with progress bar"""
    with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
        file_size = int(response.info().get('Content-Length', 0))
        desc = f"Downloading {os.path.basename(save_path)}"
        
        with tqdm(total=file_size, desc=desc, unit='B', unit_scale=True) as pbar:
            while True:
                buffer = response.read(8192)
                if not buffer:
                    break
                    
                out_file.write(buffer)
                pbar.update(len(buffer))

def extract_tar(tar_path, extract_path):
    """Extract tar file with progress bar"""
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        print(f"Extracting {tar_path} to {extract_path}")
        for member in tqdm(members, desc=f"Extracting {os.path.basename(tar_path)}", unit='files'):
            tar.extract(member, extract_path)

def prepare_dataset(data_dir):
    """Prepare Food-101 dataset for training by organizing into train/test folders"""
    data_dir = Path(data_dir)
    food_dir = data_dir / 'food-101'
    
    # Check if dataset is already prepared
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    if train_dir.exists() and test_dir.exists():
        print("Dataset already prepared")
        return
    
    # Create directories
    train_dir.mkdir(exist_ok=True, parents=True)
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Get class names from meta/classes.txt
    classes_file = food_dir / 'meta' / 'classes.txt'
    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found at {classes_file}")
    
    class_names = [line.strip() for line in open(classes_file)]
    print(f"Found {len(class_names)} classes")
    
    # Process train and test splits
    meta_dir = food_dir / 'meta'
    images_dir = food_dir / 'images'
    
    # Process test data
    print("Processing test data...")
    test_split_file = meta_dir / 'test.txt'
    test_images = [line.strip() for line in open(test_split_file)]
    
    for image_path in tqdm(test_images, desc="Copying test images"):
        # Image path format: class/image_name
        class_name, image_name = image_path.split('/')
        
        # Create class directory if it doesn't exist
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Copy image
        src = images_dir / f"{image_path}.jpg"
        dst = test_dir / class_name / f"{image_name}.jpg"
        shutil.copy(src, dst)
    
    # Process train data
    print("Processing train data...")
    train_split_file = meta_dir / 'train.txt'
    train_images = [line.strip() for line in open(train_split_file)]
    
    for image_path in tqdm(train_images, desc="Copying train images"):
        # Image path format: class/image_name
        class_name, image_name = image_path.split('/')
        
        # Create class directory if it doesn't exist
        (train_dir / class_name).mkdir(exist_ok=True)
        
        # Copy image
        src = images_dir / f"{image_path}.jpg"
        dst = train_dir / class_name / f"{image_name}.jpg"
        shutil.copy(src, dst)
    
    print(f"Dataset prepared: {len(train_images)} training images, {len(test_images)} test images")
    
    # Verify the dataset structure
    train_classes = len(os.listdir(train_dir))
    test_classes = len(os.listdir(test_dir))
    
    print(f"Train classes: {train_classes}, Test classes: {test_classes}")
    
    # Count images in each split
    train_count = sum(len(os.listdir(train_dir / class_name)) for class_name in os.listdir(train_dir))
    test_count = sum(len(os.listdir(test_dir / class_name)) for class_name in os.listdir(test_dir))
    
    print(f"Train images: {train_count}, Test images: {test_count}")

def main():
    args = parse_args()
    
    # Create data directory if it doesn't exist
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Download dataset if needed
    if not args.prepare_only:
        url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        tar_path = data_dir / "food-101.tar.gz"
        
        if not tar_path.exists():
            print(f"Downloading Food-101 dataset from {url}")
            download_file(url, tar_path)
        else:
            print(f"Dataset archive already exists at {tar_path}")
        
        # Extract dataset if needed
        if not (data_dir / "food-101").exists():
            print("Extracting dataset...")
            extract_tar(tar_path, data_dir)
        else:
            print("Dataset already extracted")
    
    # Prepare dataset if needed
    if not args.download_only:
        print("Preparing dataset...")
        prepare_dataset(data_dir)
    
    print("Done!")
    
if __name__ == "__main__":
    main()
