import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

def buildq_network(n_actions, learning_rate=0.00001, input_shape=(84, 84), history_length=4):

    """Builds a dueling DQN as a Keras model
    Arguments:
        n_actions:      Number of possible action the agent can take.
        learning_rate:  Learning rate (nolmally the value is 0.99).
        input_shape:    Shape of the preprocessed frame the model sees.
        history_length: Number of historical frames the agent can see.
    Returns:
        A compiled Keras model
    """
    # Variable to store the values of the shape
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    # Lambda is used to transform the input data using an expression or function
    x = Lambda(lambda layer: layer / 255)(model_input)  # normalize the RGB values by dividing the value between 255

    x = Conv2D(32, (8, 8),   strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (4, 4),   strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3),   strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer

    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    # tf.reduce_mean Reduces input_tensor along the dimensions given in axis
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean

    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model