import tensorflow as tf
import numpy as np

class Model:
  def create_model(self, feature_dim):
    self.feature_dim = feature_dim

    # batch x features
    self.x = tf.placeholder(tf.float32, shape=[None, feature_dim], name="X")
    self.y = tf.placeholder(tf.float32, shape=[None, 1], name="Y")

    W = tf.get_variable("W", shape=[feature_dim, 1])
    b = tf.get_variable("b", shape=[1, 1])

    y_pred = tf.matmul(self.x, W) + b

    # Compute loss, divinding by 2 just for loss gradient computation
    self.loss = tf.reduce_mean(tf.pow((self.y-y_pred), 2))/2

    self.optimiser = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    self.init = tf.global_variables_initializer()

  def run_test(self, batch_size):
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                              7.042, 10.791, 5.313, 7.997, 5.654, 9.27]).reshape(-1, batch_size, self.feature_dim)
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                              2.827, 3.465, 1.65, 2.904, 2.42, 2.94])[:train_X.shape[0]*batch_size].reshape(-1, batch_size,1)

    self.train(train_X, train_Y)


  def train(self, train_x, train_y):
    with tf.Session() as session:
      session.run(self.init)
      for i in range(50):
        for (x, y) in zip(train_x, train_y):
            pred_loss, _ = session.run([self.loss, self.optimiser],
                                       feed_dict={self.x: x,
                                                  self.y: y})

        print(pred_loss)
    
if __name__ == '__main__':
  model = Model()
  model.create_model(2)
  model.run_test(2)