from model import PoseHighResolutionNet, extra32
from tensorflow import keras

from packaging import version
import tensorflow as tf

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras import layers, optimizers, mixed_precision
else:
    from keras import layers, optimizers, mixed_precision

import tensorflow_datasets as tfds

# mixed_precision.set_global_policy('mixed_float16')

from loss import AELoss
from dataset import prepare
import datetime

# `tf.distribute.MirroredStrategy` constructor, they will be auto-detected.
strategy = tf.distribute.MirroredStrategy()
num_devices = strategy.num_replicas_in_sync
print('Number of devices: {}'.format(num_devices))

with strategy.scope():
    # A model, an optimizer, and a checkpoint must be created under `strategy.scope`.

    # loss
    loss_fn = AELoss()


    # def compute_loss(label, prediction, num_devices=num_devices):
    #     # gloabl_batch_size= batch_size * num_devices
    #     # here loss is already per batch, so gloabl_batch_size=num_devices
    #     loss = loss_fn(label, prediction)
    #     # loss / num_devices
    #     # loss = tf.nn.compute_average_loss(loss,
    #     #                                   global_batch_size=num_devices)
    #     return loss / num_devices

    # model
    def create_model(num_classes=2):
        x = layers.Input(shape=(512, 512, 3))
        out = PoseHighResolutionNet.func(x, extra32, heads={"hm": num_classes, "reg": 2, "em": 2})
        model = keras.Model(x, out)
        return model


    model = create_model()

    # optimizer
    optimizer = optimizers.Adam(learning_rate=num_devices * 1e-4)
    # optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


def train_step(x, y):
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        stacked_loss = loss_fn(y, prediction)
        # compute_average_loss will sum up the stacked_loss and divided by global_batch_size
        # sum(stacked_loss) / global_batch_size
        loss = tf.nn.compute_average_loss(stacked_loss,
                                          global_batch_size=num_devices)

        # hm_loss, pull_loss, push_loss, off_loss = tf.split(loss, 4)
        # final_loss = hm_loss + pull_loss + push_loss + off_loss

    #     scaled_loss = optimizer.get_scaled_loss(final_loss)
    # scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # return final_loss, hm_loss, pull_loss, push_loss, off_loss
    return stacked_loss


@tf.function
def distributed_train_step(x, y):
    per_replica_losses = strategy.run(train_step, args=(x, y))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)


# model = tf.keras.models.load_model('/home/fsw/Documents/codes/foot_robot_tf/model4/80')


batch_size = 4
ds, info = tfds.load("foot_robot", with_info=True)
print(info.splits['train'].num_examples)
ds = prepare(ds['train'], batch_size, shuffle=True)
dist_ds = strategy.experimental_distribute_dataset(ds)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

epochs = 300
for epoch in range(1, epochs + 1):
    print("\nStart of epoch %d" % (epoch))

    total_loss, total_hm_loss, total_pull_loss, total_push_loss, total_off_loss = 0, 0, 0, 0, 0
    # Iterate over the batches of the dataset.
    for step, (x, y) in enumerate(dist_ds, 1):
        # loss, hm_loss, pull_loss, push_loss, off_loss = train_step(model, optimizer, loss_fn, x_batch_train, y_batch_train)
        stacked_loss = distributed_train_step(x, y)
        hm_loss, pull_loss, push_loss, off_loss = tf.split(stacked_loss, 4)
        loss = hm_loss + pull_loss + push_loss + off_loss

        # # # Log every 200 batches.
        # if step % 20 == 0:
        #     print(f"Training loss (for one batch) at step {step}: {float(loss)}, {float(hm_loss)}, {float(pull_loss)}, {float(push_loss)}, {float(off_loss)}")
        #     print("Seen so far: %s samples" % ((step + 1) * batch_size))

        total_loss += float(loss)
        total_hm_loss += float(hm_loss)
        total_pull_loss += float(pull_loss)
        total_push_loss += float(push_loss)
        total_off_loss += float(off_loss)

    total_loss /= step
    total_hm_loss /= step
    total_pull_loss /= step
    total_push_loss /= step
    total_off_loss /= step
    print(
        f"Training loss for epoch {epoch}: {total_loss}, {total_hm_loss}, {total_pull_loss}, {total_push_loss}, {total_off_loss}")

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', float(total_loss), step=epoch)
        tf.summary.scalar('hm_loss', float(total_hm_loss), step=epoch)
        tf.summary.scalar('pull_loss', float(total_pull_loss), step=epoch)
        tf.summary.scalar('push_loss', float(total_push_loss), step=epoch)
        tf.summary.scalar('off_loss', float(total_off_loss), step=epoch)

    if (epoch) % 20 == 0:
        checkpoint.save(f"model/{epoch}")


