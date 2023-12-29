import tensorflow as tf
import math

class WarmstartCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self, warmstart_learning_rate, initial_learning_rate, warmstart_steps, decay_steps, alpha=0.0, name=None
    ):
        super().__init__()

        self.warmstart_learning_rate = warmstart_learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmstart_steps = warmstart_steps
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmstartCosineDecay"):
            warmstart_learning_rate = tf.convert_to_tensor(
                self.warmstart_learning_rate, name="warmstart_learning_rate"
            )
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )

            dtype = initial_learning_rate.dtype

            def compute_lr():
                decay_steps = tf.cast(self.decay_steps, dtype)

                global_step_recomp = tf.cast(step - self.warmstart_steps, dtype)
                global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
                completed_fraction = global_step_recomp / decay_steps
                cosine_decayed = 0.5 * (
                        1.0
                        + tf.cos(tf.constant(math.pi, dtype=dtype) * completed_fraction)
                )

                decayed = (1 - self.alpha) * cosine_decayed + self.alpha
                return tf.multiply(initial_learning_rate, decayed)

            lr = tf.cond(tf.less(step, self.warmstart_steps),
                         lambda: warmstart_learning_rate + step * (initial_learning_rate - warmstart_learning_rate) / (
                                     self.warmstart_steps - 1),
                         compute_lr)
            return lr