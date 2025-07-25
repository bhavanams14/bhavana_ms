import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class KidneyStoneModelTrainer:
    def __init__(self, dataset_path='dataset', img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.models = {}
        self.history = {}
        self.batch_size = self._optimize_batch_size()
        
    def _optimize_batch_size(self):
        """Optimize batch size based on available GPU memory"""
        try:
            gpu = tf.config.list_physical_devices('GPU')[0]
            memory_info = tf.config.experimental.get_memory_info(gpu)
            total_memory = memory_info['device_limit']
            # Use 80% of available memory
            available_memory = total_memory * 0.8
            # Estimate memory per image (224x224x3 float32)
            memory_per_image = 224 * 224 * 3 * 4  # 4 bytes per float32
            # Calculate optimal batch size
            optimal_batch_size = int(available_memory / memory_per_image)
            return min(optimal_batch_size, 64)  # Cap at 64
        except:
            return 32  # Default batch size if GPU not available

    def load_data(self):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        
        X, y = [], []
        class_names = ['Non-Stone', 'Stone']
        
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} not found")
                continue
                
            print(f"Loading {class_name} images...")
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, self.img_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        X.append(img)
                        y.append(class_idx)
        
        if len(X) == 0:
            raise ValueError("No images found in dataset")
            
        X = np.array(X, dtype='float32') / 255.0
        y = np.array(y)
        
        print(f"Loaded {len(X)} images")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def extract_hog_features(self, images):
        """Extract HOG features for SVM training"""
        print("Extracting HOG features...")
        features = []
        
        for img in images:
            # Convert to grayscale
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Extract HOG features
            hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=False)
            features.append(hog_features)
        
        return np.array(features)
    
    def create_cnn_model(self):
        """Create and compile CNN model with optimized architecture"""
        model = Sequential([
            # Input layer with mixed precision
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid', dtype='float32')  # Output layer must be float32
        ])
        
        # Use mixed precision optimizer
        optimizer = Adam(learning_rate=0.001)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_cnn(self, X, y, epochs=50, validation_split=0.2):
        """Train CNN model with optimized parameters"""
        print("Training CNN model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Optimized data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            dtype='float16'  # Use float16 for memory efficiency
        )
        
        # Create model
        model = self.create_cnn_model()
        
        # Optimized callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                mode='min'
            ),
            ModelCheckpoint(
                'models/best_cnn_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]
        
        # Train model with optimized parameters
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            workers=4,  # Optimize CPU workers
            use_multiprocessing=True  # Enable multiprocessing
        )
        
        # Save final model
        model.save('models/cnn_model.h5')
        
        # Evaluate
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"CNN Validation Accuracy: {val_accuracy:.4f}")
        
        self.models['cnn'] = model
        self.history['cnn'] = history
        
        return model, history
    
    def train_svm(self, X, y, test_size=0.2):
        """Train SVM model"""
        print("Training SVM model...")
        
        # Extract HOG features
        X_features = self.extract_hog_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = svm_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"SVM Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Stone', 'Stone']))
        
        # Save model and scaler
        joblib.dump(svm_model, 'models/svm_model.pkl')
        joblib.dump(scaler, 'models/svm_scaler.pkl')
        
        self.models['svm'] = svm_model
        
        return svm_model, accuracy
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'models/{model_name.lower()}_training_history.png')
        plt.show()
    
    def evaluate_models(self, X, y):
        """Evaluate both models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # Evaluate CNN
        if 'cnn' in self.models:
            cnn_pred = self.models['cnn'].predict(X_test)
            cnn_pred_binary = (cnn_pred > 0.5).astype(int).flatten()
            cnn_accuracy = accuracy_score(y_test, cnn_pred_binary)
            
            print(f"\nCNN Model:")
            print(f"Accuracy: {cnn_accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, cnn_pred_binary, target_names=['Non-Stone', 'Stone']))
            
            results['cnn'] = {
                'accuracy': cnn_accuracy,
                'predictions': cnn_pred_binary,
                'probabilities': cnn_pred.flatten()
            }
        
        # Evaluate SVM
        if 'svm' in self.models:
            X_test_features = self.extract_hog_features(X_test)
            scaler = joblib.load('models/svm_scaler.pkl')
            X_test_scaled = scaler.transform(X_test_features)
            
            svm_pred = self.models['svm'].predict(X_test_scaled)
            svm_accuracy = accuracy_score(y_test, svm_pred)
            
            print(f"\nSVM Model:")
            print(f"Accuracy: {svm_accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, svm_pred, target_names=['Non-Stone', 'Stone']))
            
            results['svm'] = {
                'accuracy': svm_accuracy,
                'predictions': svm_pred,
                'probabilities': self.models['svm'].predict_proba(X_test_scaled)[:, 1]
            }
        
        return results
    
    def train_all_models(self, epochs=50):
        """Train both CNN and SVM models"""
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Load data
        X, y = self.load_data()
        
        print(f"\nDataset loaded: {len(X)} images")
        print(f"Image shape: {X[0].shape}")
        print(f"Classes: {np.unique(y)}")
        
        # Train CNN
        print("\n" + "="*50)
        print("TRAINING CNN MODEL")
        print("="*50)
        cnn_model, cnn_history = self.train_cnn(X, y, epochs=epochs)
        self.plot_training_history(cnn_history, 'CNN')
        
        # Train SVM
        print("\n" + "="*50)
        print("TRAINING SVM MODEL")
        print("="*50)
        svm_model, svm_accuracy = self.train_svm(X, y)
        
        # Evaluate models
        results = self.evaluate_models(X, y)
        
        # Save training summary
        self.save_training_summary(results)
        
        return results
    
    def save_training_summary(self, results):
        """Save training summary to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            'timestamp': timestamp,
            'dataset_path': self.dataset_path,
            'image_size': self.img_size,
            'results': results
        }
        
        import json
        with open(f'models/training_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nTraining summary saved to models/training_summary_{timestamp}.json")

def main():
    """Main training function"""
    print("Kidney Stone Detection Model Training")
    print("="*50)
    
    # Initialize trainer
    trainer = KidneyStoneModelTrainer(dataset_path='dataset')
    
    # Train all models
    try:
        results = trainer.train_all_models(epochs=30)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        # Print final results
        for model_name, result in results.items():
            print(f"{model_name.upper()} Final Accuracy: {result['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()
