import tensorflow as tf
from tensorflow import keras

class SimplyConnected(keras.layers.Layer):
    def __init__(self, input_dim=32, la=0):
        super(SimplyConnected, self).__init__()
        w_init = tf.random_normal_initializer()
        self.la = la
        
    def build(self, input_dim):
        self.w = self.add_weight(
            shape=(input_dim[-1], ),
            initializer="random_normal",
            regularizer=tf.keras.regularizers.l2(self.la),
            trainable=True,
        )
        
    def call(self, inputs):
        return tf.math.multiply(inputs, self.w)
        
class TibLinearLasso(tf.keras.layers.Layer):
  def __init__(self, input_dim, num_outputs=1, use_bias=False, la=0):
    super(TibLinearLasso, self).__init__()
    self.num_outputs = num_outputs
    self.la = la
    self.fc = tf.keras.layers.Dense(input_shape = (input_dim,), units = 1, use_bias=use_bias, bias_regularizer=None, activation=None, kernel_regularizer=tf.keras.regularizers.l2(self.la))
    self.sc = SimplyConnected(input_dim = input_dim, la=la)

  def call(self, input):
    return self.fc(self.sc(input))
