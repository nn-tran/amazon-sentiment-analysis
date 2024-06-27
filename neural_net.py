import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomEmbedding(Layer):
    def __init__(self, input_dim, output_dim, input_length, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddings = self.add_weight(shape=(self.input_dim, self.output_dim),
                                          initializer='uniform',
                                          trainable=True)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

class CustomLSTM(Layer):
    def __init__(self, units, **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)
        self.units = units
        self.state_size = [tf.TensorShape([self.units]), tf.TensorShape([self.units])]

    def build(self, input_shape):
        self.Wf = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        self.Uf = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
        self.bf = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        self.Wi = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        self.Ui = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
        self.bi = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        self.Wc = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        self.Uc = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
        self.bc = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        self.Wo = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        self.Uo = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
        self.bo = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs, states):
        h_t, c_t = states
        f_t = tf.sigmoid(tf.matmul(inputs, self.Wf) + tf.matmul(h_t, self.Uf) + self.bf)
        i_t = tf.sigmoid(tf.matmul(inputs, self.Wi) + tf.matmul(h_t, self.Ui) + self.bi)
        c_hat_t = tf.tanh(tf.matmul(inputs, self.Wc) + tf.matmul(h_t, self.Uc) + self.bc)
        c_t = f_t * c_t + i_t * c_hat_t
        o_t = tf.sigmoid(tf.matmul(inputs, self.Wo) + tf.matmul(h_t, self.Uo) + self.bo)
        h_t = o_t * tf.tanh(c_t)
        return h_t, [h_t, c_t]

class CustomDense(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)



class CustomModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, output_dim, max_sequence_length):
        super(CustomModel, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding = CustomEmbedding(vocab_size, embedding_dim, input_length=self.max_sequence_length)
        self.lstm = CustomLSTM(lstm_units)
        self.dense = CustomDense(output_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        states = [tf.zeros((tf.shape(inputs)[0], self.lstm.units)), tf.zeros((tf.shape(inputs)[0], self.lstm.units))]
        for t in range(self.max_sequence_length):
            x_t = x[:, t, :]

            h_t, states = self.lstm(x_t, states)
        output = self.dense(h_t)
        return output