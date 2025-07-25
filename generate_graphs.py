import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import joblib
from tensorflow.keras.models import load_model
import os
import json
from datetime import datetime

def load_training_results():
    """Load actual training results from saved models and logs"""
    results = {}
    
    # Load CNN model and get its history
    try:
        cnn_model = load_model('models/cnn_best_model.keras')
        # Load training history from the model's metadata
        with open('models/cnn_training_history.json', 'r') as f:
            cnn_history = json.load(f)
        results['cnn'] = {
            'model': cnn_model,
            'history': cnn_history
        }
    except Exception as e:
        print(f"Error loading CNN model: {str(e)}")
    
    # Load SVM model and its results
    try:
        svm_model = joblib.load('models/svm_model.pkl')
        svm_pca = joblib.load('models/svm_pca.pkl')
        results['svm'] = {
            'model': svm_model,
            'pca': svm_pca
        }
    except Exception as e:
        print(f"Error loading SVM model: {str(e)}")
    
    return results

def plot_model_performance_comparison(results):
    """Plot 1: Model Performance Comparison using actual results"""
    # Load actual metrics from evaluation results
    try:
        with open('models/evaluation_results.json', 'r') as f:
            eval_results = json.load(f)
            
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        cnn_values = [
            eval_results['cnn']['accuracy'],
            eval_results['cnn']['precision'],
            eval_results['cnn']['recall'],
            eval_results['cnn']['f1_score']
        ]
        svm_values = [
            eval_results['svm']['accuracy'],
            eval_results['svm']['precision'],
            eval_results['svm']['recall'],
            eval_results['svm']['f1_score']
        ]
        hybrid_values = [
            eval_results['hybrid']['accuracy'],
            eval_results['hybrid']['precision'],
            eval_results['hybrid']['recall'],
            eval_results['hybrid']['f1_score']
        ]
    except Exception as e:
        print(f"Error loading evaluation results: {str(e)}")
        return

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, cnn_values, width, label='CNN', color='#2ecc71')
    ax.bar(x, svm_values, width, label='SVM', color='#3498db')
    ax.bar(x + width, hybrid_values, width, label='Hybrid', color='#e74c3c')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for i, v in enumerate(cnn_values):
        ax.text(i - width, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(svm_values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(hybrid_values):
        ax.text(i + width, v + 0.01, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig('visualizations/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_progress(results):
    """Plot 2: Training Progress and Learning Curves using actual history"""
    if 'cnn' not in results or 'history' not in results['cnn']:
        print("CNN training history not available")
        return

    history = results['cnn']['history']
    epochs = range(1, len(history['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(epochs, history['accuracy'], label='Training Accuracy', color='#2ecc71')
    ax1.plot(epochs, history['val_accuracy'], label='Validation Accuracy', color='#e74c3c')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Loss plot
    ax2.plot(epochs, history['loss'], label='Training Loss', color='#2ecc71')
    ax2.plot(epochs, history['val_loss'], label='Validation Loss', color='#e74c3c')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add accuracy achieved at the end of the line for training accuracy
    final_train_accuracy = history['accuracy'][-1]
    ax1.text(epochs[-1], final_train_accuracy + 0.005, f'Acc: {final_train_accuracy:.4f}', 
             color='blue', fontsize=10, ha='right', va='bottom', fontweight='bold')

    # Add accuracy achieved at the end of the line for validation accuracy
    final_val_accuracy = history['val_accuracy'][-1]
    ax1.text(epochs[-1], final_val_accuracy + 0.005, f'Val Acc: {final_val_accuracy:.4f}', 
             color='red', fontsize=10, ha='right', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_and_pr_curves(results):
    """Plot 3: ROC and Precision-Recall Curves using actual predictions"""
    try:
        with open('models/prediction_results.json', 'r') as f:
            pred_results = json.load(f)
            
        # Get actual ROC and PR curve data
        fpr_cnn, tpr_cnn, _ = roc_curve(pred_results['y_true'], pred_results['cnn_probs'])
        fpr_svm, tpr_svm, _ = roc_curve(pred_results['y_true'], pred_results['svm_probs'])
        fpr_hybrid, tpr_hybrid, _ = roc_curve(pred_results['y_true'], pred_results['hybrid_probs'])
        
        precision_cnn, recall_cnn, _ = precision_recall_curve(pred_results['y_true'], pred_results['cnn_probs'])
        precision_svm, recall_svm, _ = precision_recall_curve(pred_results['y_true'], pred_results['svm_probs'])
        precision_hybrid, recall_hybrid, _ = precision_recall_curve(pred_results['y_true'], pred_results['hybrid_probs'])
        
        # Calculate AUC and AP scores
        auc_cnn = auc(fpr_cnn, tpr_cnn)
        auc_svm = auc(fpr_svm, tpr_svm)
        auc_hybrid = auc(fpr_hybrid, tpr_hybrid)
        
        ap_cnn = average_precision_score(pred_results['y_true'], pred_results['cnn_probs'])
        ap_svm = average_precision_score(pred_results['y_true'], pred_results['svm_probs'])
        ap_hybrid = average_precision_score(pred_results['y_true'], pred_results['hybrid_probs'])
        
    except Exception as e:
        print(f"Error loading prediction results: {str(e)}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # ROC curves
    ax1.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {auc_cnn:.4f})', color='#2ecc71')
    ax1.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.4f})', color='#3498db')
    ax1.plot(fpr_hybrid, tpr_hybrid, label=f'Hybrid (AUC = {auc_hybrid:.4f})', color='#e74c3c')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Precision-Recall curves
    ax2.plot(recall_cnn, precision_cnn, label=f'CNN (AP = {ap_cnn:.4f})', color='#2ecc71')
    ax2.plot(recall_svm, precision_svm, label=f'SVM (AP = {ap_svm:.4f})', color='#3498db')
    ax2.plot(recall_hybrid, precision_hybrid, label=f'Hybrid (AP = {ap_hybrid:.4f})', color='#e74c3c')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('visualizations/roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices_and_errors(results):
    """Plot 4: Confusion Matrices and Error Analysis using actual predictions"""
    try:
        with open('models/evaluation_results.json', 'r') as f:
            eval_results = json.load(f)
            
        # Get actual confusion matrices
        cnn_cm = np.array(eval_results['cnn']['confusion_matrix'])
        svm_cm = np.array(eval_results['svm']['confusion_matrix'])
        hybrid_cm = np.array(eval_results['hybrid']['confusion_matrix'])
        
        # Get actual error distribution
        error_types = eval_results['error_analysis']
        
    except Exception as e:
        print(f"Error loading evaluation results: {str(e)}")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Confusion matrices
    sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('CNN Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('SVM Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    sns.heatmap(hybrid_cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('Hybrid Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')

    # Error distribution pie chart
    ax4.pie(error_types.values(), labels=error_types.keys(), autopct='%1.1f%%')
    ax4.set_title('Error Distribution')

    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices_and_errors.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance_and_confidence(results):
    """Plot 5: Feature Importance and Confidence Analysis using actual data"""
    try:
        with open('models/feature_analysis.json', 'r') as f:
            feature_data = json.load(f)
            
        # Get actual feature importance
        cnn_features = feature_data['cnn_importance']
        hog_features = feature_data['hog_importance']
        
        # Get actual confidence scores
        confidence_scores = np.array(feature_data['confidence_scores'])
        
    except Exception as e:
        print(f"Error loading feature analysis data: {str(e)}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Feature importance
    features = ['CNN Features', 'HOG Features']
    importance = [cnn_features, hog_features]
    ax1.bar(features, importance, color=['#2ecc71', '#3498db'])
    ax1.set_title('Feature Importance')
    ax1.set_ylabel('Importance Score')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Confidence distribution
    ax2.hist(confidence_scores, bins=20, color='#e74c3c', alpha=0.7)
    ax2.set_title('Prediction Confidence Distribution')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('visualizations/feature_importance_and_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()

def load_db_training_results():
    with open('db_training_results.json', 'r') as f:
        data = json.load(f)
    return data

def plot_model_performance_comparison(db_results):
    """Plot model performance comparison using real data"""
    plt.figure(figsize=(8, 5))  # Reduced figure size
    
    # Extract final accuracies from training logs and epoch logs
    cnn_accuracy = db_results['epoch_logs'][-1]['accuracy']  # Use final epoch accuracy
    svm_accuracy = next(log['accuracy'] for log in db_results['training_logs'] if log['model_type'] == 'SVM')
    
    # Create bar chart
    models = ['CNN', 'SVM']
    accuracies = [cnn_accuracy, svm_accuracy]
    
    bars = plt.bar(models, accuracies, color=['#2ecc71', '#3498db'])
    
    # Add value labels on top of bars (ensure correct formatting)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracies[i]:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('Model Performance Comparison', fontsize=12, pad=15)
    plt.ylabel('Accuracy', fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/db_model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cnn_training_progress(db_results):
    """Plot CNN training progress using real data"""
    plt.figure(figsize=(8, 4))  # Reduced figure size
    
    # Extract epoch data
    epochs = range(1, len(db_results['epoch_logs']) + 1)
    accuracies = [log['accuracy'] for log in db_results['epoch_logs']]
    losses = [log['loss'] for log in db_results['epoch_logs']]
    val_accuracies = [log['val_accuracy'] for log in db_results['epoch_logs']]
    val_losses = [log['val_loss'] for log in db_results['epoch_logs']]
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracies, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accuracies, 'r--', label='Validation Accuracy', linewidth=2)
    plt.title('CNN Training Progress - Accuracy', fontsize=10)
    plt.xlabel('Epoch', fontsize=9)
    plt.ylabel('Accuracy', fontsize=9)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add accuracy achieved at the end of the line for training accuracy
    final_train_accuracy = accuracies[-1]
    plt.text(epochs[-1], final_train_accuracy + 0.005, f'Acc: {final_train_accuracy:.4f}', 
             color='blue', fontsize=10, ha='right', va='bottom', fontweight='bold')

    # Add accuracy achieved at the end of the line for validation accuracy
    final_val_accuracy = val_accuracies[-1]
    plt.text(epochs[-1], final_val_accuracy + 0.005, f'Val Acc: {final_val_accuracy:.4f}', 
             color='red', fontsize=10, ha='right', va='bottom', fontweight='bold')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)
    plt.title('CNN Training Progress - Loss', fontsize=10)
    plt.xlabel('Epoch', fontsize=9)
    plt.ylabel('Loss', fontsize=9)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add loss achieved at the end of the line for training loss
    final_train_loss = losses[-1]
    plt.text(epochs[-1], final_train_loss + 0.005, f'Loss: {final_train_loss:.4f}', 
             color='blue', fontsize=10, ha='right', va='bottom', fontweight='bold')

    # Add loss achieved at the end of the line for validation loss
    final_val_loss = val_losses[-1]
    plt.text(epochs[-1], final_val_loss + 0.005, f'Val Loss: {final_val_loss:.4f}', 
             color='red', fontsize=10, ha='right', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/db_cnn_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_svm_training_progress(db_results):
    """Plot SVM training progress using real data"""
    plt.figure(figsize=(6, 4))  # Reduced figure size
    
    # Extract SVM final accuracy
    svm_accuracy = next(log['accuracy'] for log in db_results['training_logs'] if log['model_type'] == 'SVM')
    
    # Create bar chart for single epoch result
    plt.bar(['SVM'], [svm_accuracy], color='#3498db')
    
    # Add value label
    plt.text(0, svm_accuracy, f'{svm_accuracy:.4f}', 
             ha='center', va='bottom', fontsize=10)
    
    plt.title('SVM Training Progress', fontsize=12, pad=15)
    plt.ylabel('Accuracy', fontsize=10)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/db_svm_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(db_results):
    """Plot confusion matrices for all three models using user-specified values."""
    plt.figure(figsize=(15, 5))
    
    # User-specified confusion matrices
    cm_cnn = np.array([[146, 1], [3, 146]])   # TN, FP | FN, TP
    cm_svm = np.array([[140, 9], [11, 140]])
    cm_hybrid = np.array([[148, 2], [2, 148]])
    totals = [296, 300, 300]
    accs = [0.9795, 0.9342, 0.9895]
    
    for i, (cm, model, acc, total) in enumerate(zip([cm_cnn, cm_svm, cm_hybrid],
                                                     ['CNN', 'SVM', 'Hybrid'],
                                                     accs, totals)):
        plt.subplot(1, 3, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{model} Confusion Matrix\nAccuracy: {acc:.4f}, Total: {total}', fontsize=12)
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('visualizations/db_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_time(db_results):
    """Plot total training time for CNN and SVM using provided hh:mm:ss values."""
    plt.figure(figsize=(8, 5))
    
    # Provided times in hh:mm:ss
    model_names = ['CNN', 'SVM']
    times_hms = ['11:51:09', '03:59:48']
    # Convert to total seconds for plotting
    def hms_to_seconds(hms):
        h, m, s = map(int, hms.split(':'))
        return h * 3600 + m * 60 + s
    times_sec = [hms_to_seconds(t) for t in times_hms]
    
    bars = plt.bar(model_names, times_sec, color=['#2ecc71', '#3498db'])
    
    # Add value labels in hh:mm:ss
    for bar, hms in zip(bars, times_hms):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(times_sec)*0.02,
                 hms, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Total Model Training Time', fontsize=14, pad=15)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/db_training_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the database results
    with open('db_training_results.json', 'r') as f:
        db_results = json.load(f)
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Generate all graphs
    plot_model_performance_comparison(db_results)
    plot_cnn_training_progress(db_results)
    plot_svm_training_progress(db_results)
    plot_confusion_matrices(db_results)
    plot_training_time(db_results)
    
    print("All graphs have been generated and saved in the visualizations/ directory.")

if __name__ == "__main__":
    main() 