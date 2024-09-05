import sys
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import pydot
import graphviz


def prepare_data():
    mnist = tf.keras.datasets.mnist
    # Prepare data for training
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # bring values between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
    )
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
    )
    print("loaded data")
    return mnist, x_train, y_train, x_test, y_test


def train_model(x_train, y_train):
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            64,  # how many filters
            (3, 3),  # size of filters
            activation="relu",
            input_shape=(28, 28, 1)  # each image is 28x28 pixels, one channel (grayscale)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(), # make one flat layer (vector)

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),  # avoid overfitting!

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(10, activation="softmax")  # 10 digits
        # softmax: probability distribution over the 10 digits
    ])
    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=10)
    print("trained model")
    return model


if __name__ == '__main__':
    # Load data
    mnist, x_train, y_train, x_test, y_test = prepare_data()
    # Train neural network
    model = train_model(x_train, y_train)

    plot_model(model, show_shapes=True)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")
