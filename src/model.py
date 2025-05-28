import tensorflow as tf
from keras import regularizers

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(1, kernel_regularizer=regularizers.l2(0.01))
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='mean_squared_error')
    return model