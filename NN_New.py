import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, optimizers, Model, initializers, regularizers
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow import keras
disable_eager_execution()

class Multiply_By_Const(layers.Layer):
    def __init__(self, **kwargs):
        super(Multiply_By_Const, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sigma = self.add_weight(name='sigma', shape=(1,),
                                     initializer=initializers.random_uniform(), trainable=True)
        super(Multiply_By_Const, self).build(input_shape)

    def call(self, input_data):
        return tf.multiply(input_data, tf.broadcast_to(self.sigma, tf.shape(input_data)))

    def compute_output_shape(self, input_shape):
        return input_shape

class Variable(layers.Layer):
    def __init__(self, initial_value, **kwargs):
        self.initial_value = initial_value
        self.output_shape2 = np.shape(initial_value)
        super(Variable, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='weights',
                                      shape=self.output_shape2, trainable=True, initializer=initializers.constant(self.initial_value))
        super(Variable, self).build(input_shape)

    def call(self, input_data):
        return tf.broadcast_to(self.kernel, tf.shape(input_data))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_shape2[0], self.output_shape2[1])

def RBF_Model(RBFs, input_dimensions, number_of_RBFs):
    input = layers.Input(batch_input_shape=(None, None, None, input_dimensions), dtype="float32")

    cluster_centers = tf.constant(RBFs, dtype="float32")
    b = tf.tile(tf.expand_dims(input, axis=-2), [1, 1, 1, number_of_RBFs, 1])
    distances = layers.Subtract()([tf.broadcast_to(cluster_centers, tf.shape(b)), b])
    squares = tf.square(distances)
    reduce = tf.reduce_sum(squares, axis=-1)
    sigma_out = -Multiply_By_Const()(reduce)
    exp_out = tf.exp(sigma_out)
    dense_1_out = layers.Dense(100, activation="sigmoid")(exp_out)
    linear_out = layers.Dense(1, activation=None)(dense_1_out)

    model = Model(inputs=input, outputs=linear_out)
    model.compile(loss=losses.MSE, optimizer=optimizers.Adam(learning_rate=1e-5), metrics=[])
    model.summary(line_length = 150)
    return model

def RBF_Model_2(RBFs, input_dimensions, number_of_RBFs):
    input = layers.Input(batch_input_shape=(None, None, None, input_dimensions), dtype="float32")

    b = tf.tile(tf.expand_dims(input, axis=-2), [1, 1, 1, number_of_RBFs, 1])
    cluster_centers = Variable(RBFs)(b)
    distances = layers.Subtract()([cluster_centers, b])
    squares = tf.square(distances)
    reduce = tf.reduce_sum(squares, axis=-1)
    #sigma = Variable(np.ones(number_of_RBFs)*0.3)(reduce)
    #mult_out = layers.Multiply()([sigma, reduce])

    sigma_out = -Multiply_By_Const()(reduce)
    exp_out = tf.exp(sigma_out)
    dense_1_out = layers.Dense(100, activation="sigmoid")(exp_out)
    linear_out = layers.Dense(1, activation=None)(dense_1_out)

    model = Model(inputs=input, outputs=linear_out)
    model.compile(loss=losses.MSE, optimizer=optimizers.Adam(learning_rate=1e-5), metrics=[])
    model.summary(line_length = 150)
    return model


def model_wrapper(x, model):
    model.inputs = x
    return model.outputs

def train_over_whole_year_model(score_model):
    input_earnings = layers.Input(batch_shape=(None, None, None))

    scores = score_model.outputs
    reshaped = tf.squeeze(scores, [0, 4])

    mins = tf.broadcast_to(tf.reduce_min(reshaped, axis=-1, keepdims=True), tf.shape(reshaped))
    reshaped = layers.Subtract()([reshaped, mins])

    maxs = tf.broadcast_to(tf.reduce_max(reshaped, axis=-1, keepdims=True), tf.shape(reshaped))
    reshaped = tf.math.divide_no_nan(reshaped, maxs)

    reshaped = tf.multiply(reshaped, tf.constant(3.0))

    softmax_scores = layers.Softmax(axis=-1)(reshaped)

    earnings = layers.Multiply()([softmax_scores, input_earnings])
    earnings_average_day = tf.reduce_sum(earnings, axis=-1)
    earnings_average_overall = tf.reduce_prod(earnings_average_day, axis=-1, keepdims=True)

    model = Model(inputs=[score_model.inputs, input_earnings], outputs=earnings_average_overall)
    model.compile(loss=losses.MSE, optimizer=optimizers.Adam(learning_rate=1e-4), metrics=[])
    model.summary(line_length = 150)
    return model

def score_output_NN(score_model):
    input_earnings = layers.Input(batch_shape=(None, None, None))

    scores = score_model.outputs
    reshaped = tf.squeeze(scores, [0, 4])

    mins = tf.broadcast_to(tf.reduce_min(reshaped, axis=-1, keepdims=True), tf.shape(reshaped))
    reshaped = layers.Subtract()([reshaped, mins])

    maxs = tf.broadcast_to(tf.reduce_max(reshaped, axis=-1, keepdims=True), tf.shape(reshaped))
    reshaped = tf.math.divide_no_nan(reshaped, maxs)

    reshaped = tf.multiply(reshaped, tf.constant(3.0))

    softmax_scores = layers.Softmax(axis=-1)(reshaped)

    model = Model(inputs=[score_model.inputs, input_earnings], outputs=softmax_scores)
    model.compile(loss=losses.MSE, optimizer=optimizers.Adam(learning_rate=1e-4), metrics=[])
    model.summary(line_length=150)
    return model

if __name__ == '__main__':
    RBF_Model(np.zeros(10), 10, 1)