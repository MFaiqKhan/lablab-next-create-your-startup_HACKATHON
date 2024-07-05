import tensorflow as tf
import tf2onnx
from keras.src.legacy.saving import legacy_h5_format

model = legacy_h5_format.load_model_from_hdf5("raw_models\inceptionv3_tf.h5", custom_objects={'MSE': 'MSE'})

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path, opset=14)
print(f"Model Succesfully converted to ONNX format and saved at {output_path}")