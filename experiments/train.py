import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, mixed_precision
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. GPU & MIXED PRECISION CONFIGURATION
# ==========================================
# enable mixed precision to speed up training on t4 gpu
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# dataset path
data_path = '/content/drive/MyDrive/Layer1_DS/TRAIN_layer1_datasets '

# ==========================================
# 2. DATA GENERATORS & AUGMENTATION
# ==========================================
# data augmentation to break green color bias and force model to learn structural features
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.15  # 15% validation split
)

# training generator
train_gen = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=64, # optimal batch size for t4 gpu
    class_mode='binary',
    subset='training',
    shuffle=True,
    color_mode='rgb'
)

# validation generator
val_gen = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    subset='validation',
    shuffle=False, # mandatory false for accurate evaluation matching
    color_mode='rgb'
)

# ==========================================
# 3. MODEL ARCHITECTURE & FINE-TUNING
# ==========================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# unfreeze last 20 layers so the model learns fine geometric textures instead of memorizing colors
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4), # extra protection against color overfitting
    layers.Dense(1, activation='sigmoid', dtype='float32')
])

# compile model with a low learning rate for safe fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# class weights and checkpoint configuration
final_weights = {0: 1.43, 1: 0.77}
checkpoint_path = "/content/drive/MyDrive/best_plant_model_T4_final.keras"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_loss', # validation loss monitoring for academic reliability
    mode='min',
    verbose=1
)

# ==========================================
# 4. MODEL TRAINING
# ==========================================
print("--- Starting Training on Nvidia T4 ---")
history = model.fit(
    train_gen,
    epochs=12,
    validation_data=val_gen,
    class_weight=final_weights,
    callbacks=[checkpoint_callback]
)

# plot training & validation accuracy / loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# ==========================================
# 5. EVALUATION & METRICS GENERATION
# ==========================================
print("\n🔄 Generating predictions for evaluation...")
predictions = model.predict(val_gen).flatten()
y_true = val_gen.classes
y_pred = (predictions > 0.50).astype(int)
target_names = list(val_gen.class_indices.keys())

# print classification report and accuracy
print(f"\n🚀 Validation Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
print("\n📊 --- Official Classification Report ---")
print(classification_report(y_true, y_pred, target_names=target_names))

# plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# plot advanced evaluation curves (precision-recall, recall-confidence, f1-confidence)
print("\n📈 --- Generating Evaluation Curves ---")
precisions, recalls, thresholds = precision_recall_curve(y_true, predictions)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

plt.figure(figsize=(7, 16))

# curve 1: precision-recall curve
plt.subplot(3, 1, 1)
plt.plot(recalls, precisions, color='purple', lw=2)
plt.fill_between(recalls, precisions, alpha=0.15, color='purple')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True, linestyle='--', alpha=0.5)

# curve 2: recall-confidence curve
plt.subplot(3, 1, 2)
plt.plot(thresholds, recalls[:-1], color='forestgreen', lw=2)
plt.title('Recall-Confidence Curve')
plt.xlabel('Confidence Threshold')
plt.ylabel('Recall')
plt.grid(True, linestyle='--', alpha=0.5)

# curve 3: f1-confidence curve
plt.subplot(3, 1, 3)
plt.plot(thresholds, f1_scores[:-1], color='dodgerblue', lw=2)
plt.title('F1-Confidence Curve')
plt.xlabel('Confidence Threshold')
plt.ylabel('F1 Score')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
