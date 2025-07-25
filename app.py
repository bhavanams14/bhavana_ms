from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import pickle
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from skimage.feature import hog
import base64
import io
from PIL import Image
from functools import wraps
from tqdm import tqdm
from sklearn.decomposition import PCA
from flask_socketio import SocketIO, emit
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
app.secret_key = 'kidney_stone_detection_secret_key_2025'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
IMG_SIZE = (224, 224)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('user_type') != 'doctor':
            flash('Access denied. Doctor privileges required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def init_db():
    """Initialize SQLite database with enhanced schema"""
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            user_type TEXT NOT NULL CHECK (user_type IN ('patient', 'doctor')),
            full_name TEXT NOT NULL,
            phone TEXT,
            date_of_birth DATE,
            gender TEXT,
            address TEXT,
            specialization TEXT,
            license_number TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create patients table (extended profile for patients)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            medical_history TEXT,
            allergies TEXT,
            current_medications TEXT,
            emergency_contact_name TEXT,
            emergency_contact_phone TEXT,
            blood_group TEXT,
            height REAL,
            weight REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create predictions table (enhanced)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT NOT NULL,
            symptoms TEXT,
            pain_level INTEGER,
            additional_notes TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id)
        )
    ''')
    
    # Create doctor recommendations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctor_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            diagnosis TEXT NOT NULL,
            treatment_plan TEXT NOT NULL,
            medications TEXT,
            lifestyle_recommendations TEXT,
            follow_up_date DATE,
            urgency_level TEXT CHECK (urgency_level IN ('low', 'medium', 'high', 'critical')),
            status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'cancelled')),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id),
            FOREIGN KEY (patient_id) REFERENCES users (id)
        )
    ''')
    
    # Create patient progress table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recommendation_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            progress_notes TEXT,
            pain_level INTEGER,
            symptoms_improvement TEXT,
            medication_compliance TEXT,
            side_effects TEXT,
            next_appointment DATE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (recommendation_id) REFERENCES doctor_recommendations (id),
            FOREIGN KEY (patient_id) REFERENCES users (id)
        )
    ''')
    
    # Create appointments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            appointment_date DATETIME NOT NULL,
            appointment_type TEXT,
            status TEXT DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'completed', 'cancelled', 'rescheduled')),
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    ''')
    
    # Create training_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            accuracy REAL NOT NULL,
            loss REAL,
            epochs INTEGER,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create training_epoch_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_epoch_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT,
            epoch INTEGER,
            accuracy REAL,
            loss REAL,
            val_accuracy REAL,
            val_loss REAL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin doctor if not exists
    cursor.execute('SELECT COUNT(*) FROM users WHERE user_type = "doctor"')
    if cursor.fetchone()[0] == 0:
        admin_password = generate_password_hash('admin123')
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, user_type, full_name, specialization, license_number)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('admin_doctor', 'admin@hospital.com', admin_password, 'doctor', 'Dr. Admin', 'Urology', 'DOC001'))
    
    conn.commit()
    conn.close()

def preprocess_image(image_path, target_size=IMG_SIZE):
    """Preprocess image for model prediction"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return img

def extract_hog_features(image_path):
    """Extract HOG features for SVM"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Reduce image size for HOG
    img = cv2.resize(img, (64, 64))
    features = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), visualize=False)
    return features

def create_cnn_model():
    """Create CNN model for feature extraction and classification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        Flatten(),
        Dropout(0.4),
        Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def save_prediction_to_db(patient_id, filename, prediction, confidence, model_used, symptoms=None, pain_level=None, notes=None):
    """Save prediction results to database"""
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (patient_id, filename, prediction, confidence, model_used, symptoms, pain_level, additional_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, filename, prediction, confidence, model_used, symptoms, pain_level, notes))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('kidney_stone_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, password_hash, user_type, full_name FROM users WHERE username = ? AND is_active = 1', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['user_type'] = user[2]
            session['full_name'] = user[3]
            flash(f'Welcome back, {user[3]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']
        full_name = request.form['full_name']
        phone = request.form.get('phone')
        
        # Additional fields for doctors
        specialization = request.form.get('specialization') if user_type == 'doctor' else None
        license_number = request.form.get('license_number') if user_type == 'doctor' else None
        
        conn = sqlite3.connect('kidney_stone_detection.db')
        cursor = conn.cursor()
        
        # Check if username or email already exists
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            flash('Username or email already exists', 'error')
            conn.close()
            return render_template('register.html')
        
        # Create user
        password_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, user_type, full_name, phone, specialization, license_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, email, password_hash, user_type, full_name, phone, specialization, license_number))
        
        user_id = cursor.lastrowid
        
        # Create patient profile if user is patient
        if user_type == 'patient':
            cursor.execute('INSERT INTO patient_profiles (user_id) VALUES (?)', (user_id,))
        
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_type = session.get('user_type')
    user_id = session.get('user_id')
    
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    if user_type == 'patient':
        # Get patient's recent predictions
        cursor.execute('''
            SELECT id, filename, prediction, confidence, timestamp
            FROM predictions 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''', (user_id,))
        recent_predictions = cursor.fetchall()
        
        # Get patient's recommendations
        cursor.execute('''
            SELECT dr.id, dr.diagnosis, dr.treatment_plan, dr.urgency_level, dr.status, dr.created_at,
                   u.full_name as doctor_name
            FROM doctor_recommendations dr
            JOIN users u ON dr.doctor_id = u.id
            WHERE dr.patient_id = ? AND dr.status = 'active'
            ORDER BY dr.created_at DESC
        ''', (user_id,))
        recommendations = cursor.fetchall()
        
        # Get upcoming appointments
        cursor.execute('''
            SELECT a.id, a.appointment_date, a.appointment_type, a.status,
                   u.full_name as doctor_name
            FROM appointments a
            JOIN users u ON a.doctor_id = u.id
            WHERE a.patient_id = ? AND a.status = 'scheduled'
            ORDER BY a.appointment_date ASC
            LIMIT 3
        ''', (user_id,))
        appointments = cursor.fetchall()
        
        conn.close()
        return render_template('patient_dashboard.html', 
                             recent_predictions=recent_predictions,
                             recommendations=recommendations,
                             appointments=appointments)
    
    elif user_type == 'doctor':
        # Get pending cases for review
        cursor.execute('''
            SELECT p.id, p.filename, p.prediction, p.confidence, p.timestamp,
                   u.full_name as patient_name, u.id as patient_id
            FROM predictions p
            JOIN users u ON p.patient_id = u.id
            LEFT JOIN doctor_recommendations dr ON p.id = dr.prediction_id
            WHERE dr.id IS NULL
            ORDER BY p.timestamp DESC
            LIMIT 10
        ''', ())
        pending_cases = cursor.fetchall()
        
        # Get doctor's active recommendations
        cursor.execute('''
            SELECT dr.id, dr.diagnosis, dr.urgency_level, dr.status, dr.created_at,
                   u.full_name as patient_name
            FROM doctor_recommendations dr
            JOIN users u ON dr.patient_id = u.id
            WHERE dr.doctor_id = ? AND dr.status = 'active'
            ORDER BY dr.created_at DESC
            LIMIT 10
        ''', (user_id,))
        active_recommendations = cursor.fetchall()
        
        # Get today's appointments
        cursor.execute('''
            SELECT a.id, a.appointment_date, a.appointment_type, a.status,
                   u.full_name as patient_name
            FROM appointments a
            JOIN users u ON a.patient_id = u.id
            WHERE a.doctor_id = ? AND DATE(a.appointment_date) = DATE('now')
            ORDER BY a.appointment_date ASC
        ''', (user_id,))
        todays_appointments = cursor.fetchall()
        
        conn.close()
        return render_template('doctor_dashboard.html',
                             pending_cases=pending_cases,
                             active_recommendations=active_recommendations,
                             todays_appointments=todays_appointments)
    
    conn.close()
    return render_template('dashboard.html')

@app.route('/upload')
@login_required
def upload_page():
    if session.get('user_type') != 'patient':
        flash('Only patients can upload CT scans', 'error')
        return redirect(url_for('dashboard'))
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if session.get('user_type') != 'patient':
        return jsonify({'error': 'Only patients can upload CT scans'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'cnn')
    symptoms = request.form.get('symptoms', '')
    pain_level = request.form.get('pain_level', type=int)
    notes = request.form.get('notes', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if model_type == 'cnn':
                # Load CNN model
                model_path = 'models/cnn_best_model.keras'
                if os.path.exists(model_path):
                    model = load_model(model_path)
                    img = preprocess_image(filepath)
                    if img is not None:
                        img = np.expand_dims(img, axis=0)
                        prediction = model.predict(img)[0][0]
                        confidence = float(prediction)
                        result = 'Stone' if prediction > 0.5 else 'Non-Stone'
                        
                        prediction_id = save_prediction_to_db(
                            session['user_id'], filename, result, confidence, 'CNN',
                            symptoms, pain_level, notes
                        )
                        
                        return jsonify({
                            'prediction': result,
                            'confidence': confidence,
                            'model': 'CNN',
                            'filename': filename,
                            'prediction_id': prediction_id
                        })
                else:
                    return jsonify({'error': 'CNN model not found. Please contact administrator.'})
            
            elif model_type == 'svm':
                # Load SVM model
                if os.path.exists('models/svm_model.pkl'):
                    svm_model = joblib.load('models/svm_model.pkl')
                    # Load PCA object
                    pca_path = 'models/svm_pca.pkl'
                    if not os.path.exists(pca_path):
                        return jsonify({'error': 'SVM PCA object not found. Please retrain the SVM model.'})
                    pca = joblib.load(pca_path)
                    features = extract_hog_features(filepath)
                    if features is not None:
                        features = features.reshape(1, -1)
                        features_pca = pca.transform(features)
                        prediction = svm_model.predict(features_pca)[0]
                        confidence = float(max(svm_model.predict_proba(features_pca)[0]))
                        result = 'Stone' if prediction == 1 else 'Non-Stone'
                        prediction_id = save_prediction_to_db(
                            session['user_id'], filename, result, confidence, 'SVM',
                            symptoms, pain_level, notes
                        )
                        return jsonify({
                            'prediction': result,
                            'confidence': confidence,
                            'model': 'SVM',
                            'filename': filename,
                            'prediction_id': prediction_id
                        })
                else:
                    return jsonify({'error': 'SVM model not found. Please contact administrator.'})
            
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'})
    
    return jsonify({'error': 'Invalid file format'})

@app.route('/doctor/cases')
@doctor_required
def doctor_cases():
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    # Get all predictions without recommendations
    cursor.execute('''
        SELECT p.id, p.filename, p.prediction, p.confidence, p.model_used, p.symptoms, 
               p.pain_level, p.additional_notes, p.timestamp,
               u.full_name as patient_name, u.id as patient_id, u.email, u.phone
        FROM predictions p
        JOIN users u ON p.patient_id = u.id
        LEFT JOIN doctor_recommendations dr ON p.id = dr.prediction_id
        WHERE dr.id IS NULL
        ORDER BY p.timestamp DESC
    ''')
    
    pending_cases = cursor.fetchall()
    conn.close()
    
    return render_template('doctor_cases.html', cases=pending_cases)

@app.route('/doctor/case/<int:case_id>')
@doctor_required
def view_case(case_id):
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    # Get case details
    cursor.execute('''
        SELECT p.id, p.filename, p.prediction, p.confidence, p.model_used, p.symptoms, 
               p.pain_level, p.additional_notes, p.timestamp,
               u.full_name as patient_name, u.id as patient_id, u.email, u.phone,
               u.date_of_birth, u.gender, u.address,
               pp.medical_history, pp.allergies, pp.current_medications, pp.blood_group,
               pp.height, pp.weight
        FROM predictions p
        JOIN users u ON p.patient_id = u.id
        LEFT JOIN patient_profiles pp ON u.id = pp.user_id
        WHERE p.id = ?
    ''', (case_id,))
    
    case = cursor.fetchone()
    
    if not case:
        flash('Case not found', 'error')
        return redirect(url_for('doctor_cases'))
    
    # Get patient's previous predictions
    cursor.execute('''
        SELECT id, filename, prediction, confidence, timestamp
        FROM predictions
        WHERE patient_id = ? AND id != ?
        ORDER BY timestamp DESC
        LIMIT 5
    ''', (case[11], case_id))
    
    previous_predictions = cursor.fetchall()
    conn.close()
    
    return render_template('case_detail.html', case=case, previous_predictions=previous_predictions)

@app.route('/doctor/recommend/<int:case_id>', methods=['POST'])
@doctor_required
def add_recommendation(case_id):
    diagnosis = request.form['diagnosis']
    treatment_plan = request.form['treatment_plan']
    medications = request.form.get('medications', '')
    lifestyle_recommendations = request.form.get('lifestyle_recommendations', '')
    follow_up_date = request.form.get('follow_up_date')
    urgency_level = request.form['urgency_level']
    
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    # Get patient ID from prediction
    cursor.execute('SELECT patient_id FROM predictions WHERE id = ?', (case_id,))
    result = cursor.fetchone()
    
    if not result:
        flash('Case not found', 'error')
        return redirect(url_for('doctor_cases'))
    
    patient_id = result[0]
    
    # Insert recommendation
    cursor.execute('''
        INSERT INTO doctor_recommendations 
        (prediction_id, doctor_id, patient_id, diagnosis, treatment_plan, medications, 
         lifestyle_recommendations, follow_up_date, urgency_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (case_id, session['user_id'], patient_id, diagnosis, treatment_plan, 
          medications, lifestyle_recommendations, follow_up_date, urgency_level))
    
    conn.commit()
    conn.close()
    
    flash('Recommendation added successfully', 'success')
    return redirect(url_for('doctor_cases'))

@app.route('/patient/recommendations')
@login_required
def patient_recommendations():
    if session.get('user_type') != 'patient':
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT dr.id, dr.diagnosis, dr.treatment_plan, dr.medications, 
               dr.lifestyle_recommendations, dr.follow_up_date, dr.urgency_level, 
               dr.status, dr.created_at, dr.updated_at,
               u.full_name as doctor_name, u.specialization,
               p.filename, p.prediction, p.confidence
        FROM doctor_recommendations dr
        JOIN users u ON dr.doctor_id = u.id
        JOIN predictions p ON dr.prediction_id = p.id
        WHERE dr.patient_id = ?
        ORDER BY dr.created_at DESC
    ''', (session['user_id'],))
    
    recommendations = cursor.fetchall()
    conn.close()
    
    return render_template('patient_recommendations.html', recommendations=recommendations)

@app.route('/patient/progress/<int:recommendation_id>', methods=['GET', 'POST'])
@login_required
def patient_progress(recommendation_id):
    if session.get('user_type') != 'patient':
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    if request.method == 'POST':
        progress_notes = request.form['progress_notes']
        pain_level = request.form.get('pain_level', type=int)
        symptoms_improvement = request.form['symptoms_improvement']
        medication_compliance = request.form['medication_compliance']
        side_effects = request.form.get('side_effects', '')
        
        cursor.execute('''
            INSERT INTO patient_progress 
            (recommendation_id, patient_id, progress_notes, pain_level, symptoms_improvement, 
             medication_compliance, side_effects)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (recommendation_id, session['user_id'], progress_notes, pain_level, 
              symptoms_improvement, medication_compliance, side_effects))
        
        conn.commit()
        flash('Progress updated successfully', 'success')
    
    # Get recommendation details
    cursor.execute('''
        SELECT dr.id, dr.diagnosis, dr.treatment_plan, dr.medications, 
               dr.lifestyle_recommendations, dr.follow_up_date, dr.urgency_level,
               u.full_name as doctor_name
        FROM doctor_recommendations dr
        JOIN users u ON dr.doctor_id = u.id
        WHERE dr.id = ? AND dr.patient_id = ?
    ''', (recommendation_id, session['user_id']))
    
    recommendation = cursor.fetchone()
    
    if not recommendation:
        flash('Recommendation not found', 'error')
        return redirect(url_for('patient_recommendations'))
    
    # Get progress history
    cursor.execute('''
        SELECT progress_notes, pain_level, symptoms_improvement, medication_compliance, 
               side_effects, created_at
        FROM patient_progress
        WHERE recommendation_id = ? AND patient_id = ?
        ORDER BY created_at DESC
    ''', (recommendation_id, session['user_id']))
    
    progress_history = cursor.fetchall()
    conn.close()
    
    return render_template('patient_progress.html', 
                         recommendation=recommendation, 
                         progress_history=progress_history)

@app.route('/doctor/patients')
@doctor_required
def doctor_patients():
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    # Get all patients with recommendations from this doctor
    cursor.execute('''
        SELECT DISTINCT u.id, u.full_name, u.email, u.phone, 
               COUNT(dr.id) as total_recommendations,
               MAX(dr.created_at) as last_recommendation
        FROM users u
        JOIN doctor_recommendations dr ON u.id = dr.patient_id
        WHERE dr.doctor_id = ?
        GROUP BY u.id, u.full_name, u.email, u.phone
        ORDER BY last_recommendation DESC
    ''', (session['user_id'],))
    
    patients = cursor.fetchall()
    conn.close()
    
    return render_template('doctor_patients.html', patients=patients)

@app.route('/doctor/patient/<int:patient_id>')
@doctor_required
def view_patient(patient_id):
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    # Get patient details
    cursor.execute('''
        SELECT u.id, u.full_name, u.email, u.phone, u.date_of_birth, u.gender, u.address,
               pp.medical_history, pp.allergies, pp.current_medications, pp.blood_group,
               pp.height, pp.weight, pp.emergency_contact_name, pp.emergency_contact_phone
        FROM users u
        LEFT JOIN patient_profiles pp ON u.id = pp.user_id
        WHERE u.id = ? AND u.user_type = 'patient'
    ''', (patient_id,))
    
    patient = cursor.fetchone()
    
    if not patient:
        flash('Patient not found', 'error')
        return redirect(url_for('doctor_patients'))
    
    # Get patient's recommendations from this doctor
    cursor.execute('''
        SELECT id, diagnosis, treatment_plan, medications, lifestyle_recommendations,
               follow_up_date, urgency_level, status, created_at
        FROM doctor_recommendations
        WHERE patient_id = ? AND doctor_id = ?
        ORDER BY created_at DESC
    ''', (patient_id, session['user_id']))
    
    recommendations = cursor.fetchall()
    
    # Get patient's progress reports
    cursor.execute('''
        SELECT pp.progress_notes, pp.pain_level, pp.symptoms_improvement, 
               pp.medication_compliance, pp.side_effects, pp.created_at,
               dr.diagnosis
        FROM patient_progress pp
        JOIN doctor_recommendations dr ON pp.recommendation_id = dr.id
        WHERE pp.patient_id = ? AND dr.doctor_id = ?
        ORDER BY pp.created_at DESC
        LIMIT 10
    ''', (patient_id, session['user_id']))
    
    progress_reports = cursor.fetchall()
    
    # Get patient's predictions
    cursor.execute('''
        SELECT id, filename, prediction, confidence, model_used, symptoms, 
               pain_level, timestamp
        FROM predictions
        WHERE patient_id = ?
        ORDER BY timestamp DESC
        LIMIT 10
    ''', (patient_id,))
    
    predictions = cursor.fetchall()
    
    conn.close()
    
    return render_template('patient_detail.html', 
                         patient=patient, 
                         recommendations=recommendations,
                         progress_reports=progress_reports,
                         predictions=predictions)

@app.route('/train')
@doctor_required
def train_page():
    return render_template('train.html')

@app.route('/train_model', methods=['POST'])
@doctor_required
def train_model():
    model_type = request.form.get('model_type', 'cnn')
    epochs = int(request.form.get('epochs', 10))
    
    # Check if dataset exists
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset folder not found'})
    
    try:
        if model_type == 'cnn':
            # Train CNN model
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
            
            train_generator = datagen.flow_from_directory(
                dataset_path,
                target_size=IMG_SIZE,
                batch_size=32,
                class_mode='binary',
                subset='training'
            )
            
            validation_generator = datagen.flow_from_directory(
                dataset_path,
                target_size=IMG_SIZE,
                batch_size=32,
                class_mode='binary',
                subset='validation'
            )
            
            model = create_cnn_model()
            
            # Custom callback for real-time updates with error handling
            class TrainingCallback(tf.keras.callbacks.Callback):
                def __init__(self):
                    super().__init__()
                    self.current_epoch = 0
                
                def on_epoch_begin(self, epoch, logs=None):
                    self.current_epoch = epoch
                    try:
                        socketio.emit('training_status', {
                            'type': 'epoch_start',
                            'epoch': epoch + 1,
                            'total_epochs': epochs
                        })
                    except Exception as e:
                        print(f"Error in epoch begin callback: {str(e)}")
                
                def on_epoch_end(self, epoch, logs=None):
                    try:
                        socketio.emit('training_update', {
                            'type': 'training_update',
                            'epoch': epoch + 1,
                            'total_epochs': epochs,
                            'train_accuracy': float(logs['accuracy']),
                            'val_accuracy': float(logs['val_accuracy']),
                            'loss': float(logs['loss'])
                        })
                    except Exception as e:
                        print(f"Error in epoch end callback: {str(e)}")
                
                def on_train_end(self, logs=None):
                    try:
                        socketio.emit('training_status', {
                            'type': 'training_complete',
                            'final_accuracy': float(logs['val_accuracy']),
                            'final_loss': float(logs['val_loss'])
                        })
                    except Exception as e:
                        print(f"Error in train end callback: {str(e)}")
            
            # Initialize training status
            socketio.emit('training_status', {
                'type': 'training_start',
                'total_epochs': epochs
            })
            
            # Add Early Stopping and Model Checkpoint callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )

            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                'models/cnn_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_format='h5'
            )

            # Add ReduceLROnPlateau callback
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=0.0001
            )

            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                callbacks=[TrainingCallback(), early_stopping, model_checkpoint, reduce_lr],
                verbose=1
            )
            
            # Save training log
            try:
                conn = sqlite3.connect('kidney_stone_detection.db')
                cursor = conn.cursor()

                # Get the final training accuracy and loss
                final_accuracy = history.history['accuracy'][-1]
                final_loss = history.history['loss'][-1]

                # Insert final summary log
                cursor.execute('''
                    INSERT INTO training_logs (model_type, accuracy, loss, epochs)
                    VALUES (?, ?, ?, ?)
                ''', ('CNN', final_accuracy, final_loss, epochs))

                # Insert per-epoch results
                for epoch in range(epochs):
                    acc = history.history['accuracy'][epoch]
                    loss = history.history['loss'][epoch]
                    val_acc = history.history['val_accuracy'][epoch]
                    val_loss = history.history['val_loss'][epoch]
                    cursor.execute('''
                        INSERT INTO training_epoch_logs (model_type, epoch, accuracy, loss, val_accuracy, val_loss)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', ('CNN', epoch+1, acc, loss, val_acc, val_loss))

                conn.commit()
                print("CNN training logs saved successfully.")
            except Exception as db_error:
                conn.rollback() # Rollback changes if saving fails
                import traceback
                print("\n" + "="*50)
                print("Error saving CNN training logs to database")
                print("="*50)
                print(f"Database Error: {str(db_error)}")
                print("Full Traceback:")
                traceback.print_exc()
                print("="*50 + "\n")
                # Do not return error to frontend here, as training completed. Log and continue.
            finally:
                conn.close()
            
            return jsonify({
                'success': True,
                'message': f'CNN model trained successfully with {final_accuracy:.4f} final accuracy',
                'accuracy': final_accuracy,
                'loss': final_loss
            })
            
        elif model_type == 'svm':
            # Train SVM model (implementation from previous code)
            X, y = [], []
            # Load and process images
            for class_name in ['Non-Stone', 'Stone']:
                class_path = os.path.join(dataset_path, class_name)
                if os.path.exists(class_path):
                    label = 0 if class_name == 'Non-Stone' else 1
                    img_files = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    for img_file in tqdm(img_files, desc=f"Extracting features for {class_name}"):
                        img_path = os.path.join(class_path, img_file)
                        features = extract_hog_features(img_path)
                        if features is not None:
                            X.append(features)
                            y.append(label)
            if len(X) == 0:
                return jsonify({'error': 'No valid images found in dataset'})
            X = np.array(X)
            y = np.array(y)
            # Apply PCA to reduce dimensionality
            pca = PCA(n_components=200, random_state=42)
            X_reduced = pca.fit_transform(X)
            # Save PCA object
            joblib.dump(pca, 'models/svm_pca.pkl')
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
            # Train SVM
            svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            svm_model.fit(X_train, y_train)
            # Evaluate
            y_pred = svm_model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            # Save model
            joblib.dump(svm_model, 'models/svm_model.pkl')
            # Save training log
            try:
                conn = sqlite3.connect('kidney_stone_detection.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO training_logs (model_type, accuracy, epochs)
                    VALUES (?, ?, ?)
                ''', ('SVM', accuracy, 1))
                conn.commit()
                print("SVM training logs saved successfully.")
            except Exception as db_error:
                conn.rollback()
                import traceback
                print("\n" + "="*50)
                print("Error saving SVM training logs to database")
                print("="*50)
                print(f"Database Error: {str(db_error)}")
                print("Full Traceback:")
                traceback.print_exc()
                print("="*50 + "\n")
            finally:
                conn.close()
            
            return jsonify({
                'success': True,
                'message': f'SVM model trained successfully with {accuracy:.4f} accuracy',
                'accuracy': accuracy
            })
            
    except Exception as e:
        # Catch any exception during training and log full traceback
        import traceback
        error_message = f'Training failed: {str(e)}'
        print("\n" + "="*50)
        print("CNN Training Error")
        print("="*50)
        print(error_message)
        print("Full Traceback:")
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'error': error_message})

@app.route('/history')
@login_required
def history():
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    
    if session.get('user_type') == 'patient':
        cursor.execute('''
            SELECT filename, prediction, confidence, model_used, timestamp
            FROM predictions
            WHERE patient_id = ?
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (session['user_id'],))
    else:
        cursor.execute('''
            SELECT p.filename, p.prediction, p.confidence, p.model_used, p.timestamp,
                   u.full_name as patient_name
            FROM predictions p
            JOIN users u ON p.patient_id = u.id
            ORDER BY p.timestamp DESC
            LIMIT 50
        ''')
    
    predictions = cursor.fetchall()
    conn.close()
    
    return render_template('history.html', predictions=predictions)

@app.route('/get_training_history')
def get_training_history():
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, model_type, accuracy, loss, epochs, date 
        FROM training_logs 
        ORDER BY date DESC
    ''')
    history = cursor.fetchall()
    conn.close()
    
    return jsonify([{
        'id': row[0],
        'model_type': row[1],
        'accuracy': row[2],
        'loss': row[3],
        'epochs': row[4],
        'date': datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S').timestamp() * 1000 # Convert to Unix timestamp in milliseconds
    } for row in history])

@app.route('/get_training_details/<int:log_id>')
def get_training_details(log_id):
    conn = sqlite3.connect('kidney_stone_detection.db')
    cursor = conn.cursor()

    # Get final training log details
    cursor.execute('''
        SELECT id, model_type, accuracy, loss, epochs, date
        FROM training_logs
        WHERE id = ?
    ''', (log_id,))
    final_log = cursor.fetchone()

    if not final_log:
        conn.close()
        return jsonify({'error': 'Training log not found'}), 404

    # Get per-epoch logs (only for CNN, if available)
    epoch_logs = []
    if final_log[1] == 'CNN': # Check if model type is CNN
        cursor.execute('''
            SELECT epoch, accuracy, loss, val_accuracy, val_loss, date
            FROM training_epoch_logs
            WHERE model_type = ?
            ORDER BY epoch ASC
        ''', (final_log[1],))
        epoch_logs_raw = cursor.fetchall()
        epoch_logs = [{
            'epoch': row[0],
            'accuracy': row[1],
            'loss': row[2],
            'val_accuracy': row[3],
            'val_loss': row[4],
            'date': datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S').timestamp() * 1000
        } for row in epoch_logs_raw]

    conn.close()

    # Format final log date as Unix timestamp
    final_log_formatted = {
        'id': final_log[0],
        'model_type': final_log[1],
        'accuracy': final_log[2],
        'loss': final_log[3],
        'epochs': final_log[4],
        'date': datetime.strptime(final_log[5], '%Y-%m-%d %H:%M:%S').timestamp() * 1000
    }

    return jsonify({
        'final_log': final_log_formatted,
        'epoch_logs': epoch_logs
    })

if __name__ == '__main__':
    init_db()
    socketio.run(app, debug=True)
