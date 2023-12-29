import tensorflow as tf
from tensorflow.python.keras import backend
from packaging import version

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras import layers
    from tensorflow import keras
else:
    from keras import layers
    import keras
from tf_common import load_model_from_ckpt

from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from model_compression_toolkit.qat.keras.quantizer.configs.weight_quantizer_config import WeightQuantizeConfig

qat_model = tf.keras.models.load_model('/home/fsw/Documents/codes/foot_robot_tf/qat_model',
                                       custom_objects={"QuantizeWrapper": QuantizeWrapper,
                                                       "WeightQuantizeConfig": WeightQuantizeConfig})


