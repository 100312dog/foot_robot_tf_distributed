# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This tutorial demonstrates how the Model Compression Toolkit (MCT) prepares a model for Quantization Aware
Training. A model is trained on the MNIST dataset and then quantized and being QAT-ready by the MCT and
returned to the user. A QAT-ready model is a model with certain layers wrapped by a QuantizeWrapper with
the requested quantizers.
The user can now Fine-Tune the QAT-ready model. Finally, the model is finalized by the MCT which means the
MCT replaces the QuantizeWrappers with their native layers and quantized weights.
"""

import argparse
import tensorflow as tf
from packaging import version
from model import PoseHighResolutionNet, extra32
if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras import Model, layers, datasets, mixed_precision
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    import tensorflow.keras as keras
else:
    from keras.datasets import mnist
    from keras import Model, layers, datasets, optimizers, mixed_precision
    from keras.applications.resnet_v2 import ResNet50V2
    import keras
import model_compression_toolkit as mct
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from dataset import prepare
from loss import AELoss
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from model_compression_toolkit.qat.keras.quantizer.configs.weight_quantizer_config import WeightQuantizeConfig

# mixed_precision.set_global_policy('mixed_float16')
from keras.datasets import mnist


def get_dataset(batch_size):
    # ds = tfds.load("foot_robot", split="train[:500]")
    ds, info = tfds.load("foot_robot", split="train", with_info=True)
    ds = prepare(ds, batch_size, shuffle=True)
    return ds, info.splits['train'].num_examples // batch_size


def gen_representative_dataset(_images, num_calibration_iterations):
    # Return a Callable representative dataset for calibration purposes.
    # The function should be called without any arguments, and should return a list numpy arrays (array
    # for each model's input).
    # In this tutorial, each time the representative dataset is called it returns a list containing a single
    # MNIST image of shape (1, 28, 28, 1).
    def _generator():
        for i,(_img,_label) in enumerate(_images):
            # yield [_img[np.newaxis, ...]]
            if i == num_calibration_iterations:
                break
            yield [_img]
    return _generator


def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size for model training.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs for model training.')
    parser.add_argument('--num_calibration_iterations', type=int, default=10,
                        help='number of iterations for calibration - model quantization before fine-tuning.')
    return parser.parse_args()


@tf.function
def train_step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        # predictions = model(x, training=True)
        predictions = model(x)
        loss = loss_fn(y, predictions)
        hm_loss, pull_loss, push_loss, off_loss = tf.unstack(loss, 4)
        final_loss = hm_loss + pull_loss + push_loss + off_loss

    #     scaled_loss = optimizer.get_scaled_loss(final_loss)
    # scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    gradients = tape.gradient(final_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return final_loss, hm_loss, pull_loss, push_loss, off_loss

@tf.function
def eval_step(model,loss_fn, x, y):
    predictions = model(x)
    loss = loss_fn(y, predictions)
    hm_loss, pull_loss, push_loss, off_loss = tf.unstack(loss, 4)
    final_loss = hm_loss + pull_loss + push_loss + off_loss
    return final_loss


if __name__ == "__main__":
    """
    The code below is an example code of a user for fine tuning a float model with the MCT Quantization
    Aware Training API. 
    """

    # # # Parse arguments
    args = argument_handler()

    train_db, db_len = get_dataset(args.batch_size)
    representative_dataset = gen_representative_dataset(train_db, args.num_calibration_iterations)

    loss_fn = AELoss()
    num_steps = db_len * args.num_epochs
    scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-5, decay_steps=num_steps, alpha=0.01)
    # scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=)
    optimizer = optimizers.Adam(learning_rate=scheduler)



    # num_classes = 2
    # x = layers.Input(shape=(512, 512, 3))
    # out = PoseHighResolutionNet.func(x, extra32, heads={"hm": num_classes, "reg": 2, "em": 2})
    # model = keras.Model(x, out)
    # checkpoint = tf.train.Checkpoint(model=model)
    # checkpoint.restore(tf.train.latest_checkpoint("/home/fsw/Downloads/mmm")).expect_partial()
    # # model = tf.keras.models.load_model("/home/fsw/Documents/codes/foot_robot_tf_distributed/3model_")
    #
    #
    # total_loss = 0
    # step = 0
    # for x_batch_train, y_batch_train in train_db:
    #     loss = eval_step(model, loss_fn, x_batch_train, y_batch_train)
    #     total_loss += loss
    #     step += 1
    # total_loss /= step
    # print("EVAL_MODEL", total_loss)
    #
    # tf.keras.backend.clear_session()
    #
    # # prepare model for QAT with MCT and return to user for fine-tuning. Due to the relatively easy
    # # task of quantizing model trained on MNIST, a custom TPC is used in this example to demonstrate
    # # the degradation caused by post training quantization.
    # target_platform_cap = mct.get_target_platform_capabilities('tensorflow', 'tflite', 'latest')
    # eight_bits = mct.QuantizationConfig(
    #     weights_per_channel_threshold=False
    # )
    # qat_model, quantization_info, custom_objects = mct.keras_quantization_aware_training_init(model,
    #                                                              representative_dataset,
    #                                                              core_config=mct.CoreConfig(quantization_config=eight_bits),
    #                                                                                         # quantization_config=mct.QuantizationConfig(
    #                                                                                         #     activation_error_method=mct.QuantizationErrorMethod.NOCLIPPING,
    #                                                                                         # )),
    #                                                              target_platform_capabilities=target_platform_cap)
    #
    # qat_model.save("qat_model2")

    qat_model = tf.keras.models.load_model('qat_model_train/50',
                                           custom_objects={"QuantizeWrapper":QuantizeWrapper, "WeightQuantizeConfig":WeightQuantizeConfig})

    # qat_model.summary()


    total_loss = 0
    step = 0
    for x_batch_train, y_batch_train in train_db:
        loss = eval_step(qat_model, loss_fn, x_batch_train, y_batch_train)
        total_loss += float(loss)
        step += 1
    total_loss /= step
    print("EVAL_QAT_MODEL", total_loss)




    # # fine-tune QAT model from MCT to recover the lost accuracy.
    #
    # for epoch in range(1, args.num_epochs+ 1):
    #     print("\nStart of epoch %d" % (epoch))
    #
    #     total_loss, total_hm_loss, total_pull_loss, total_push_loss, total_off_loss = 0, 0, 0, 0, 0
    #     # Iterate over the batches of the dataset.
    #     for step, (x_batch_train, y_batch_train) in enumerate(train_db):
    #
    #         loss, hm_loss, pull_loss, push_loss, off_loss = train_step(qat_model, optimizer, loss_fn, x_batch_train, y_batch_train)
    #
    #         # # # Log every 200 batches.
    #         # if step % 20 == 0:
    #         #     print(
    #         #         f"Training loss (for one batch) at step {step}: {float(loss)}, {float(hm_loss)}, {float(pull_loss)}, {float(push_loss)}, {float(off_loss)}")
    #         #     print("Seen so far: %s samples" % ((step + 1) * args.batch_size))
    #
    #         total_loss += float(loss)
    #         total_hm_loss += float(hm_loss)
    #         total_pull_loss += float(pull_loss)
    #         total_push_loss += float(push_loss)
    #         total_off_loss += float(off_loss)
    #
    #     total_loss /= step
    #     total_hm_loss /= step
    #     total_pull_loss /= step
    #     total_push_loss /= step
    #     total_off_loss /= step
    #     print(
    #         f"Training loss for epoch {epoch}: {total_loss}, {total_hm_loss}, {total_pull_loss}, {total_push_loss}, {total_off_loss}",
    #         f"learning rate {optimizer._decayed_lr(tf.float32)}"
    #     )
    #
    #     if epoch % 10 == 0:
    #         qat_model.save(f"qat_model_train2/{epoch}")


    # qat_model.save("qat_model_finetuned")

    # # Create TFLite model.
    # converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # converter._experimental_disable_per_channel = True
    # converter.inference_type = tf.uint8
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # # converter.quantized_input_stats = {"x" : (0., 1.)}
    # quantized_tflite_model = converter.convert()
    #
    # with open("model_wrapper.tflite", 'wb') as f:
    #     f.write(quantized_tflite_model)

    # # # Finalize QAT model: remove QuantizeWrappers and keep weights quantized as fake-quant values
    # quantized_model = mct.keras_quantization_aware_training_finalize(qat_model)
    # print(quantized_model(x))
