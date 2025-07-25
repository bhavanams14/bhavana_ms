import sqlite3
import json
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

DB_PATH = 'kidney_stone_detection.db'
OUTPUT_PATH = 'db_training_results.json'

def fetch_training_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, model_type, accuracy, loss, epochs, date FROM training_logs ORDER BY date DESC')
    logs = cursor.fetchall()
    conn.close()
    return [
        {
            'id': row[0],
            'model_type': row[1],
            'accuracy': row[2],
            'loss': row[3],
            'epochs': row[4],
            'date': row[5]
        }
        for row in logs
    ]

def fetch_epoch_logs(model_type=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if model_type:
        cursor.execute('SELECT model_type, epoch, accuracy, loss, val_accuracy, val_loss, date FROM training_epoch_logs WHERE model_type = ? ORDER BY epoch ASC', (model_type,))
    else:
        cursor.execute('SELECT model_type, epoch, accuracy, loss, val_accuracy, val_loss, date FROM training_epoch_logs ORDER BY model_type, epoch ASC')
    logs = cursor.fetchall()
    conn.close()
    return [
        {
            'model_type': row[0],
            'epoch': row[1],
            'accuracy': row[2],
            'loss': row[3],
            'val_accuracy': row[4],
            'val_loss': row[5],
            'date': row[6]
        }
        for row in logs
    ]

def fetch_prediction_results():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT y_true, cnn_probs, svm_probs, hybrid_probs FROM prediction_results')
    results = cursor.fetchall()
    conn.close()
    if not results:
        return None
    return {
        'y_true': [row[0] for row in results],
        'cnn_probs': [row[1] for row in results],
        'svm_probs': [row[2] for row in results],
        'hybrid_probs': [row[3] for row in results]
    }

def fetch_confusion_matrices():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT cnn_cm, svm_cm, hybrid_cm FROM confusion_matrices')
    results = cursor.fetchall()
    conn.close()
    if not results:
        return None
    return {
        'cnn_cm': json.loads(results[0][0]),
        'svm_cm': json.loads(results[0][1]),
        'hybrid_cm': json.loads(results[0][2])
    }

def main():
    training_logs = fetch_training_logs()
    epoch_logs = fetch_epoch_logs()
    prediction_results = fetch_prediction_results()
    confusion_matrices = fetch_confusion_matrices()
    results = {
        'training_logs': training_logs,
        'epoch_logs': epoch_logs,
        'prediction_results': prediction_results,
        'confusion_matrices': confusion_matrices
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Extracted {len(training_logs)} training logs, {len(epoch_logs)} epoch logs, prediction results, and confusion matrices to {OUTPUT_PATH}")

if __name__ == '__main__':
    main() 