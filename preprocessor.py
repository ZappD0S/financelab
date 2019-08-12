import numpy as np
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model


class Preprocessor(Layer):
    def __init__(self, time_steps, batch_size, **kwargs):
        self.time_steps = time_steps
        self.batch_size = batch_size
        super(Preprocessor, self).__init__(**kwargs)
        self.trainable = False

    def build(self, input_shape):
        N = input_shape[1]
        self.n_samples = N - self.time_steps + 1
        self.n_batches = self.n_samples // self.batch_size
        self.remainder_batch_size = self.n_batches % self.batch_size
        self.count = K.variable(0, dtype='int32')
        self.indices = K.variable(K.arange(self.n_samples))
        self.indices = K.update(self.indices, tf.random.shuffle(self.indices))
        super(Preprocessor, self).build(input_shape)

    def call(self, x):
        last_batch = K.equal(self.count, self.n_batches)
        batch_size = K.switch(last_batch, lambda: self.remainder_batch_size, lambda: self.batch_size)
        indices = self.indices[self.count * batch_size:(self.count + 1) * batch_size]
        indices = K.reshape(K.arange(batch_size), (-1, 1)) + K.reshape(K.arange(self.time_steps), (1, -1))
        out = tf.gather(x, indices, axis=-1)
        self.count = K.switch(last_batch,
                              lambda: K.update(self.count, 0),
                              lambda: K.update_add(self.count, 1))

        self.indices = K.switch(last_batch,
                                lambda: K.update(self.indices, tf.random.shuffle(self.indices)),
                                lambda: K.identity(self.indices))

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.time_steps)


if __name__ == "__main__":
    inputs = Input(shape=(500,))
    out = Preprocessor(20, 4, name='preprocessor')(inputs)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(inputs=[inputs], outputs=out)
    model.compile(loss='mse', optimizer='adam')
    x = np.random.rand(500).reshape(1, -1)
    y = np.random.rand(500).reshape(1, -1)
    for i in range(500):
        model.train_on_batch(x=x, y=y[:, i])
        # print(K.eval(model.get_layer('preprocessor').count))
    # print(K.eval(model.get_layer('preprocessor').count))
