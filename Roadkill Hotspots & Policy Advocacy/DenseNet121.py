import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import cv2

# Define Image Size & Paths
img_size = (224, 224)
dataset_dir = "/kaggle/input/roadkill-hotspot-and-policy-advocacy/RHP"
train_dir = "/kaggle/working/train"
test_dir = "/kaggle/working/test"

# Ensure train and test directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
for category in ["dead", "alive"]:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Get all images
dead_images = [os.path.join(dataset_dir, "dead", img) for img in os.listdir(os.path.join(dataset_dir, "dead"))]
alive_images = [os.path.join(dataset_dir, "alive", img) for img in os.listdir(os.path.join(dataset_dir, "alive"))]

# Split dataset
train_dead, test_dead = train_test_split(dead_images, test_size=0.2, random_state=42)
train_alive, test_alive = train_test_split(alive_images, test_size=0.2, random_state=42)

# Move images to respective directories
def move_images(image_list, destination_folder):
    for img_path in image_list:
        shutil.copy(img_path, os.path.join(destination_folder, os.path.basename(img_path)))

move_images(train_dead, os.path.join(train_dir, "dead"))
move_images(test_dead, os.path.join(test_dir, "dead"))
move_images(train_alive, os.path.join(train_dir, "alive"))
move_images(test_alive, os.path.join(test_dir, "alive"))

# Data Augmentation & Normalization
datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=32, class_mode='binary')
test_generator = datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=32, class_mode='binary', shuffle=False)

# Load Pretrained DenseNet121
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build CNN Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_generator, validation_data=test_generator, epochs=20)

# Plot Training & Validation Accuracy with Smoothing
from scipy.interpolate import make_interp_spline

epochs = np.arange(1, len(history.history['accuracy']) + 1)

# Create smooth curves
epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)

# Smooth training accuracy
spl_train_acc = make_interp_spline(epochs, history.history['accuracy'], k=3)
train_acc_smooth = spl_train_acc(epochs_smooth)

# Smooth validation accuracy
spl_val_acc = make_interp_spline(epochs, history.history['val_accuracy'], k=3)
val_acc_smooth = spl_val_acc(epochs_smooth)

# Smooth training loss
spl_train_loss = make_interp_spline(epochs, history.history['loss'], k=3)
train_loss_smooth = spl_train_loss(epochs_smooth)

# Smooth validation loss
spl_val_loss = make_interp_spline(epochs, history.history['val_loss'], k=3)
val_loss_smooth = spl_val_loss(epochs_smooth)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_smooth, train_acc_smooth, label='Training Accuracy', linewidth=2.5, color='#1f77b4')
plt.plot(epochs_smooth, val_acc_smooth, label='Validation Accuracy', linewidth=2.5, color='#ff7f0e')
plt.scatter(epochs, history.history['accuracy'], alpha=0.5, s=30, color='#1f77b4')
plt.scatter(epochs, history.history['val_accuracy'], alpha=0.5, s=30, color='#ff7f0e')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

# Plot Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_smooth, train_loss_smooth, label='Training Loss', linewidth=2.5, color='#2ca02c')
plt.plot(epochs_smooth, val_loss_smooth, label='Validation Loss', linewidth=2.5, color='#d62728')
plt.scatter(epochs, history.history['loss'], alpha=0.5, s=30, color='#2ca02c')
plt.scatter(epochs, history.history['val_loss'], alpha=0.5, s=30, color='#d62728')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Evaluate Model
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Function to Predict Dog Status
def predict_dog_status(image_path, model, img_size):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)[0][0]
    status = "Dead" if prediction > 0.5 else "Alive"
    print(f"Prediction: {status} (Probability: {prediction:.2f})")
    return status

# Load the trained model
model_path = "/kaggle/working/dog_status_model.h5"
model = load_model(model_path)

# Path to the image you want to predict
image_path = "/kaggle/working/test/alive/8.jpeg"

# Predict Dog Status
predict_dog_status(image_path, model, img_size)

# Classification Report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=['Alive', 'Dead']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Alive', 'Dead'], yticklabels=['Alive', 'Dead'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Confusion Matrix')
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 7))
plt.plot(fpr, tpr, label=f'Model (AUC = {roc_auc:.3f})', linewidth=3, color='#1f77b4')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier (AUC = 0.50)', linewidth=2)
plt.fill_between(fpr, tpr, alpha=0.2, color='#1f77b4')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()

# Model Summary
print("\n" + "="*50)
print("MODEL SUMMARY")
print("="*50)
model.summary()

# Final Evaluation Metrics
print("\n" + "="*50)
print("FINAL MODEL EVALUATION")
print("="*50)
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Prediction Distribution
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(y_pred_probs, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Probabilities')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
class_counts = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
plt.bar(['Alive', 'Dead'], class_counts, color=['green', 'red'], alpha=0.7, edgecolor='black')
plt.ylabel('Count')
plt.title('Predicted Class Distribution')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Save Model
model_save_path = "/kaggle/working/dog_status_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")