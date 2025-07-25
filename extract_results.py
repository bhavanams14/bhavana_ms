import numpy as np
import json
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import os
from sklearn.decomposition import PCA
import cv2
from skimage.feature import hog
from glob import glob

def load_and_preprocess_test_data():
    """Load and preprocess test images"""
    test_images = []
    test_labels = []
    
    # Load test images from dataset
    stone_images = glob('dataset/Stone/*.jpg') + glob('dataset/Stone/*.png')
    non_stone_images = glob('dataset/Non-Stone/*.jpg') + glob('dataset/Non-Stone/*.png')
    
    print(f"Found {len(stone_images)} stone images and {len(non_stone_images)} non-stone images")
    
    for img_path in stone_images:
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            img = cv2.resize(img, (224, 224))
            test_images.append(img)
            test_labels.append(1)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
        
    for img_path in non_stone_images:
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            img = cv2.resize(img, (224, 224))
            test_images.append(img)
            test_labels.append(0)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    if not test_images:
        raise ValueError("No valid images found in the dataset")
    
    return np.array(test_images), np.array(test_labels)

def extract_cnn_features(model, images):
    """Extract features using CNN model"""
    # Normalize images
    images = images.astype('float32') / 255.0
    return model.predict(images, verbose=0)

def extract_hog_features(images):
    """Extract HOG features from images"""
    hog_features = []
    for img in images:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys')
            hog_features.append(features)
        except Exception as e:
            print(f"Error extracting HOG features: {str(e)}")
            continue
    return np.array(hog_features)

def evaluate_models():
    """Evaluate all models and save results"""
    try:
        print("Loading models...")
        cnn_model = load_model('models/cnn_best_model.keras')
        svm_model = joblib.load('models/svm_model.pkl')
        svm_pca = joblib.load('models/svm_pca.pkl')
        
        print("Loading test data...")
        test_images, test_labels = load_and_preprocess_test_data()
        
        print("Extracting features...")
        cnn_features = extract_cnn_features(cnn_model, test_images)
        hog_features = extract_hog_features(test_images)
        
        if len(cnn_features) != len(hog_features):
            raise ValueError("Mismatch in number of features extracted")
        
        # Combine features for hybrid approach
        combined_features = np.concatenate([cnn_features, hog_features], axis=1)
        
        print("Making predictions...")
        # CNN predictions
        cnn_probs = cnn_model.predict(test_images, verbose=0)
        cnn_preds = (cnn_probs > 0.5).astype(int)
        
        # SVM predictions
        svm_features = svm_pca.transform(combined_features)
        svm_probs = svm_model.predict_proba(svm_features)[:, 1]
        svm_preds = (svm_probs > 0.5).astype(int)
        
        # Hybrid predictions (ensemble)
        hybrid_probs = (cnn_probs + svm_probs) / 2
        hybrid_preds = (hybrid_probs > 0.5).astype(int)
        
        print("Calculating metrics...")
        # Calculate metrics for each model
        results = {
            'cnn': {
                'accuracy': float(accuracy_score(test_labels, cnn_preds)),
                'precision': float(precision_score(test_labels, cnn_preds)),
                'recall': float(recall_score(test_labels, cnn_preds)),
                'f1_score': float(f1_score(test_labels, cnn_preds)),
                'confusion_matrix': confusion_matrix(test_labels, cnn_preds).tolist()
            },
            'svm': {
                'accuracy': float(accuracy_score(test_labels, svm_preds)),
                'precision': float(precision_score(test_labels, svm_preds)),
                'recall': float(recall_score(test_labels, svm_preds)),
                'f1_score': float(f1_score(test_labels, svm_preds)),
                'confusion_matrix': confusion_matrix(test_labels, svm_preds).tolist()
            },
            'hybrid': {
                'accuracy': float(accuracy_score(test_labels, hybrid_preds)),
                'precision': float(precision_score(test_labels, hybrid_preds)),
                'recall': float(recall_score(test_labels, hybrid_preds)),
                'f1_score': float(f1_score(test_labels, hybrid_preds)),
                'confusion_matrix': confusion_matrix(test_labels, hybrid_preds).tolist()
            }
        }
        
        # Calculate error analysis
        error_analysis = {
            'Calcification artifacts': 0,
            'Image noise': 0,
            'Small stones': 0,
            'Low contrast': 0,
            'Partial volume effect': 0
        }
        
        # Analyze false positives and false negatives
        for i, (true, pred) in enumerate(zip(test_labels, hybrid_preds)):
            if true != pred:
                # This is a simplified error analysis - in practice, you would need
                # more sophisticated analysis of the images
                if true == 0 and pred == 1:  # False positive
                    error_analysis['Calcification artifacts'] += 1
                elif true == 1 and pred == 0:  # False negative
                    if np.mean(test_images[i]) < 100:  # Low contrast
                        error_analysis['Low contrast'] += 1
                    elif np.std(test_images[i]) > 50:  # High noise
                        error_analysis['Image noise'] += 1
                    else:
                        error_analysis['Small stones'] += 1
        
        results['error_analysis'] = error_analysis
        
        # Save evaluation results
        with open('models/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save prediction results
        prediction_results = {
            'y_true': test_labels.tolist(),
            'cnn_probs': cnn_probs.flatten().tolist(),
            'svm_probs': svm_probs.tolist(),
            'hybrid_probs': hybrid_probs.flatten().tolist()
        }
        with open('models/prediction_results.json', 'w') as f:
            json.dump(prediction_results, f, indent=4)
        
        # Calculate feature importance
        feature_importance = {
            'cnn_importance': float(np.mean(np.abs(cnn_features))),
            'hog_importance': float(np.mean(np.abs(hog_features))),
            'confidence_scores': hybrid_probs.flatten().tolist()
        }
        with open('models/feature_analysis.json', 'w') as f:
            json.dump(feature_importance, f, indent=4)
        
        # Save CNN training history
        cnn_history = {
            'accuracy': [float(x) for x in cnn_model.history.history['accuracy']],
            'val_accuracy': [float(x) for x in cnn_model.history.history['val_accuracy']],
            'loss': [float(x) for x in cnn_model.history.history['loss']],
            'val_loss': [float(x) for x in cnn_model.history.history['val_loss']]
        }
        with open('models/cnn_training_history.json', 'w') as f:
            json.dump(cnn_history, f, indent=4)
        
        print("Results have been saved to the models directory.")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_models() 