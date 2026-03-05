import tensorflowjs as tfjs
import tensorflow as tf

# 1. Mevcut modelini yükle
model = tf.keras.models.load_model('medical_binary_model.h5')

# 2. React projenin public klasörüne dönüştür
tfjs.converters.save_keras_model(model, '../public/model')

print("✅ Model başarıyla dönüştürüldü ve ./public/model klasörüne kaydedildi!")