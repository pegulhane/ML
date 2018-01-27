import tensorflow as tf


class Model:
  def create_model(feature_dim):
  # batch x features
    self.x = tf.placeholder(tf.float32, shape=[None, feature_dim], name="X")
    self.y = tf.placeholder(tf.int, shape=[None, 1], name="Y")

    W = tf.get_variable("W", shape=[feature_dim, 1])
    b = tf.get_variable("b", shape=[None, 1])

    y_pred = tf.matmul(X,W) + b

    # Compute loss, divinding by 2 just for loss gradient computation
    self.loss = tf.reduce_mean(tf.pow((y-y_pred), 2))/2

    self.optimiser = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    self.init = tf.global_variable_intializer()
    

  def train():
    
    
