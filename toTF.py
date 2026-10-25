import tensorflow as tf

model = tf.keras.models.load_model('/content/best_plant_model_T4_final3.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_path = '/content/plant_model_mobile.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

from google.colab import files
files.download(tflite_model_path)
