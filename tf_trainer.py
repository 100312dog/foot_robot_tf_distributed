import tensorflow as tf
from tensorflow.python.keras import backend
from packaging import version
from functools import partial

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras import layers, optimizers
    from tensorflow import keras
else:
    from keras import layers, optimizers
    import keras
import datetime
from loss import AELoss
from model import PoseHighResolutionNet, extra32
import tensorflow_datasets as tfds
from dataset import prepare
from scheduler import WarmstartCosineDecay

from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from model_compression_toolkit.qat.keras.quantizer.configs.weight_quantizer_config import WeightQuantizeConfig


class BaseTrainer():
    def __init__(self):
        self.setup()

    def get_loss(self):
        pass

    def get_model(self):
        pass

    def get_optimizer(self):
        pass

    def get_dataset(self):
        pass

    def initialize(self):
        pass

    def setup(self):
        self.initialize()
        # `tf.distribute.MirroredStrategy` constructor, they will be auto-detected.
        self.strategy = tf.distribute.MirroredStrategy()
        self.num_devices = self.strategy.num_replicas_in_sync
        print('Number of devices: {}'.format(self.num_devices))
        self.dataset = self.strategy.experimental_distribute_dataset(self.get_dataset())
        with self.strategy.scope():
            self.loss = self.get_loss()
            self.model = self.get_model()
            self.optimizer = self.get_optimizer()
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    def train_step(self, x, y):
        pass

    @tf.function
    def distributed_train_step(self, x, y):
        per_replica_losses = self.strategy.run(self.train_step, args=(x, y))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)

    def test_step(self, x, y):
        pass

    @tf.function
    def distributed_test_step(self, x, y):
        per_replica_losses = self.strategy.run(self.test_step, args=(x, y))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)


class FootRobotTrainer(BaseTrainer):
    def __init__(self):
        super(FootRobotTrainer, self).__init__()

    def initialize(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/model/' + current_time + "/train"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.batch_size = 4
        self.epochs = 300
        self.model_path = "model"

    def get_loss(self):
        return AELoss()

    def get_model(self):
        num_classes = 2
        x = layers.Input(shape=(512, 512, 3))
        out = PoseHighResolutionNet.func(x, extra32, heads={"hm": num_classes, "reg": 2, "em": 2})
        model = keras.Model(x, out)
        return model

    def get_optimizer(self):
        num_steps = self.dataset_size // self.batch_size * self.epochs
        scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=self.num_devices * 1e-4,
                                                              decay_steps=num_steps, alpha=0.01)
        return optimizers.Adam(learning_rate=scheduler)

    def get_dataset(self):
        ds, info = tfds.load("foot_robot", with_info=True)
        ds = prepare(ds['train'], self.batch_size, shuffle=True)
        self.dataset_size = info.splits['train'].num_examples
        return ds

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x, training=True)
            stacked_loss = self.loss(y, prediction)
            loss = tf.nn.compute_average_loss(stacked_loss,
                                              global_batch_size=self.num_devices)

        #     scaled_loss = optimizer.get_scaled_loss(final_loss)
        # scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # return final_loss, hm_loss, pull_loss, push_loss, off_loss
        return stacked_loss

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print("\nStart of epoch %d" % (epoch))

            total_loss, total_hm_loss, total_pull_loss, total_push_loss, total_off_loss = 0, 0, 0, 0, 0
            # Iterate over the batches of the dataset.
            for step, (x, y) in enumerate(self.dataset, 1):
                # loss, hm_loss, pull_loss, push_loss, off_loss = train_step(model, optimizer, loss_fn, x_batch_train, y_batch_train)
                stacked_loss = self.distributed_train_step(x, y)
                stacked_loss /= self.num_devices
                hm_loss, pull_loss, push_loss, off_loss = tf.unstack(stacked_loss)
                loss = hm_loss + pull_loss + push_loss + off_loss

                # if step % 100 == 0:
                #     print(f"Training loss epoch{epoch} step {step}: {loss.numpy()}, {hm_loss.numpy()}, {pull_loss.numpy()}, {push_loss.numpy()}, {off_loss.numpy()}")
                #     print(f"learning rate: {optimizer._decayed_lr(tf.float32).numpy()}")

                total_loss += loss
                total_hm_loss += hm_loss
                total_pull_loss += pull_loss
                total_push_loss += push_loss
                total_off_loss += off_loss

            total_loss /= step
            total_hm_loss /= step
            total_pull_loss /= step
            total_push_loss /= step
            total_off_loss /= step
            # print(f"Training loss for epoch {epoch}: {total_loss}, {total_hm_loss}, {total_pull_loss}, {total_push_loss}, {total_off_loss}")
            print(
                f"Training loss epoch{epoch}: {total_loss.numpy()}, {total_hm_loss.numpy()}, {total_pull_loss.numpy()}, {total_push_loss.numpy()}, {total_off_loss.numpy()}")
            print(f"learning rate: {self.optimizer._decayed_lr(tf.float32).numpy()}")

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', total_loss, step=epoch)
                tf.summary.scalar('hm_loss', total_hm_loss, step=epoch)
                tf.summary.scalar('pull_loss', total_pull_loss, step=epoch)
                tf.summary.scalar('push_loss', total_push_loss, step=epoch)
                tf.summary.scalar('off_loss', total_off_loss, step=epoch)
                tf.summary.scalar('lr', self.optimizer._decayed_lr(tf.float32), step=epoch)

            if (epoch) % 20 == 0:
                self.checkpoint.save(f"{self.model_path}/{epoch}")


class QATFootRobotTrainer(FootRobotTrainer):

    def initialize(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/qat_model/' + current_time + "/train"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.batch_size = 2
        self.epochs = 100
        self.model_path = "qat_model_train"

    def get_optimizer(self):
        warmup_num_steps = self.dataset_size * 10
        decay_num_steps = self.dataset_size * (self.epochs - 10)
        scheduler = WarmstartCosineDecay(warmstart_learning_rate=1e-7, initial_learning_rate=1e-6,
                                         warmstart_steps=warmup_num_steps, decay_steps=decay_num_steps, alpha=0.1)
        return optimizers.Adam(learning_rate=scheduler)
    
    def get_model(self):
        return tf.keras.models.load_model('qat_model',
                                        custom_objects={"QuantizeWrapper":QuantizeWrapper, "WeightQuantizeConfig":WeightQuantizeConfig})







