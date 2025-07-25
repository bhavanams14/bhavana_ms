import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random
from PIL import Image, ImageEnhance, ImageFilter
import argparse

class DatasetPreparator:
    def __init__(self, source_path, target_path='dataset', img_size=(224, 224)):
        self.source_path = source_path
        self.target_path = target_path
        self.img_size = img_size
        
    def create_directory_structure(self):
        """Create the required directory structure"""
        directories = [
            os.path.join(self.target_path, 'Stone'),
            os.path.join(self.target_path, 'Non-Stone'),
            os.path.join(self.target_path, 'Augmented', 'Stone'),
            os.path.join(self.target_path, 'Augmented', 'Non-Stone')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def preprocess_image(self, image_path, output_path):
        """Preprocess a single image"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return False
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Denoise
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
            # Save preprocessed image
            cv2.imwrite(output_path, img)
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False
    
    def augment_image(self, image_path, output_dir, base_name, num_augmentations=5):
        """Create augmented versions of an image"""
        try:
            img = Image.open(image_path)
            augmented_images = []
            
            for i in range(num_augmentations):
                augmented = img.copy()
                
                # Random rotation
                if random.random() > 0.5:
                    angle = random.uniform(-15, 15)
                    augmented = augmented.rotate(angle, fillcolor=(0, 0, 0))
                
                # Random brightness
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Brightness(augmented)
                    factor = random.uniform(0.8, 1.2)
                    augmented = enhancer.enhance(factor)
                
                # Random contrast
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Contrast(augmented)
                    factor = random.uniform(0.8, 1.2)
                    augmented = enhancer.enhance(factor)
                
                # Random horizontal flip
                if random.random() > 0.5:
                    augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Random blur
                if random.random() > 0.3:
                    radius = random.uniform(0.5, 1.5)
                    augmented = augmented.filter(ImageFilter.GaussianBlur(radius))
                
                # Save augmented image
                output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
                augmented.save(output_path)
                augmented_images.append(output_path)
            
            return augmented_images
            
        except Exception as e:
            print(f"Error augmenting {image_path}: {str(e)}")
            return []
    
    def organize_dataset(self):
        """Organize images from source to target directory structure"""
        if not os.path.exists(self.source_path):
            print(f"Source path {self.source_path} does not exist")
            return False
        
        self.create_directory_structure()
        
        # Process each class
        for class_name in ['Stone', 'Non-Stone']:
            source_class_path = os.path.join(self.source_path, class_name)
            target_class_path = os.path.join(self.target_path, class_name)
            
            if not os.path.exists(source_class_path):
                print(f"Class directory {source_class_path} not found")
                continue
            
            print(f"Processing {class_name} images...")
            
            image_files = [f for f in os.listdir(source_class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for i, img_file in enumerate(image_files):
                source_img_path = os.path.join(source_class_path, img_file)
                
                # Create new filename
                base_name = f"{class_name.lower()}_{i:04d}"
                target_img_path = os.path.join(target_class_path, f"{base_name}.jpg")
                
                # Preprocess and save
                if self.preprocess_image(source_img_path, target_img_path):
                    print(f"Processed: {img_file} -> {base_name}.jpg")
                    
                    # Create augmented versions
                    aug_dir = os.path.join(self.target_path, 'Augmented', class_name)
                    self.augment_image(target_img_path, aug_dir, base_name, num_augmentations=3)
        
        return True
    
    def validate_dataset(self):
        """Validate the prepared dataset"""
        print("\nValidating dataset...")
        
        for split in ['', 'Augmented']:
            base_path = os.path.join(self.target_path, split) if split else self.target_path
            
            for class_name in ['Stone', 'Non-Stone']:
                class_path = os.path.join(base_path, class_name)
                
                if os.path.exists(class_path):
                    image_count = len([f for f in os.listdir(class_path) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    split_name = f" ({split})" if split else ""
                    print(f"{class_name}{split_name}: {image_count} images")
                else:
                    print(f"{class_name}{split_name}: Directory not found")
    
    def create_sample_dataset(self):
        """Create a sample dataset for testing"""
        print("Creating sample dataset...")
        
        # Create sample images (for demonstration)
        sample_dir = 'sample_dataset'
        os.makedirs(os.path.join(sample_dir, 'Stone'), exist_ok=True)
        os.makedirs(os.path.join(sample_dir, 'Non-Stone'), exist_ok=True)
        
        # Generate sample images (normally you would have real CT scan images)
        for class_name in ['Stone', 'Non-Stone']:
            class_dir = os.path.join(sample_dir, class_name)
            
            for i in range(10):  # Create 10 sample images per class
                # Create a random image (in practice, these would be real CT scans)
                img = np.random.randint(0, 255, (*self.img_size, 3), dtype=np.uint8)
                
                # Add some patterns to differentiate classes
                if class_name == 'Stone':
                    # Add bright spots to simulate stones
                    for _ in range(random.randint(1, 3)):
                        x, y = random.randint(50, 174), random.randint(50, 174)
                        cv2.circle(img, (x, y), random.randint(10, 30), (255, 255, 255), -1)
                
                img_path = os.path.join(class_dir, f"sample_{i:03d}.jpg")
                cv2.imwrite(img_path, img)
        
        print(f"Sample dataset created in {sample_dir}")
        return sample_dir

def main():
    parser = argparse.ArgumentParser(description='Prepare kidney stone dataset')
    parser.add_argument('--source', type=str, help='Source dataset path')
    parser.add_argument('--target', type=str, default='dataset', help='Target dataset path')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset')
    
    args = parser.parse_args()
    
    preparator = DatasetPreparator(
        source_path=args.source or 'raw_dataset',
        target_path=args.target
    )
    
    if args.create_sample:
        sample_path = preparator.create_sample_dataset()
        preparator.source_path = sample_path
    
    # Organize and prepare dataset
    if preparator.organize_dataset():
        preparator.validate_dataset()
        print("\nDataset preparation completed successfully!")
    else:
        print("Dataset preparation failed!")

if __name__ == "__main__":
    main()
