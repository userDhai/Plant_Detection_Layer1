

import os
import zipfile
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

zip_path = '/content/drive/MyDrive/Layer1_DS/Plant_Detection_Backup.zip'
extract_path = '/content/fast_data'

print("Extracting dataset ZIP file locally...")
!rm -rf /content/fast_data
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Extraction completed successfully.")

from pathlib import Path
import pandas as pd

work_path = Path("/content/fast_data")

#retrieve all image files across supported extensions
files_list = [str(p) for p in work_path.rglob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]

#label classification mapping
cleaned_data = []
for f in files_list:
    path_lower = f.lower()
    if any(x in path_lower for x in ["non", "other", "background"]):
        cleaned_data.append((f, "Non_Plants"))
    else:
        cleaned_data.append((f, "Plants"))

#create master dataframe
master_df = pd.DataFrame(cleaned_data, columns=['image_path', 'target_label'])

print(f"Total images processed: {len(master_df)}")
print(master_df['target_label'].value_counts())

#stratified split (85% train, 15% validation)
train_df, val_df = train_test_split(
    master_df, test_size=0.15, stratify=master_df['target_label'], random_state=42
)

#initialize data generators with augmentation and preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, zoom_range=0.2, brightness_range=[0.7, 1.3],
    horizontal_flip=True, fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='image_path', y_col='target_label', target_size=(224, 224),
    batch_size=64, class_mode='binary', shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col='image_path', y_col='target_label', target_size=(224, 224),
    batch_size=64, class_mode='binary', shuffle=False
)
print("Data generators initialized successfully.")

#build the mobilenetv2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
#unfreeze the last 20 layers for domain adaptation
for layer in base_model.layers[:-20]: 
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid') #binary output layer
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

#handle class imbalance using class weights during training
final_weights = {0: 1.43, 1: 0.77}
checkpoint_path = "/content/drive/MyDrive/best_plant_model_T4_final3.keras"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

print("Starting model training...")
history = model.fit(train_gen, epochs=5, validation_data=val_gen, class_weight=final_weights, callbacks=[checkpoint_callback])
