import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.regularizers as reg

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
  def __init__(self, input_dim, num_outputs=1, use_bias=False, la=0, name="tib_lasso"):
    super(TibLinearLasso, self).__init__()
    self.num_outputs = num_outputs
    self.la = la
    self.fc = tf.keras.layers.Dense(input_shape = (input_dim-1,), units = 1, use_bias=False, bias_regularizer=None, activation=None, kernel_regularizer=tf.keras.regularizers.l2(self.la))
    self.intercept = tf.keras.layers.Dense(input_shape = (1,), units = 1, use_bias=False, bias_regularizer=None, activation=None, kernel_regularizer=None)
    self.sc = SimplyConnected(input_dim = input_dim, la=la)
    self._name = name

  def call(self, input):
    return self.fc(self.sc(input[:,1:])) + self.intercept(input[:,0:1])
    
class group_lasso_pen(reg.Regularizer):

    def __init__(self, la):
        self.la = la

    def __call__(self, x):
        return self.la * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(x), 1)))

class TibLinearLassoMC(tf.keras.layers.Layer):
  def __init__(self, input_dim, num_outputs=1, use_bias=False, la=0, name="tib_lasso_MC"):
    super(TibLinearLassoMC, self).__init__()
    self.num_outputs = num_outputs
    self.la = la
    self.fc = tf.keras.layers.Dense(input_shape = (input_dim-1,), units = num_outputs, use_bias=False, bias_regularizer=None, activation=None, kernel_regularizer=group_lasso_pen(self.la))
    self.intercept = tf.keras.layers.Dense(input_shape = (1,), units = num_outputs, use_bias=False, bias_regularizer=None, activation=None, kernel_regularizer=None)
    self.sc = SimplyConnected(input_dim = input_dim, la=la)
    self._name = name

  def call(self, input):
    return self.fc(self.sc(input[:,1:])) + self.intercept(input[:,0:1])
