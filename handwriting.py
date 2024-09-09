import sys
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import pydot
import graphviz

# Definición de función para preparar los datos
def prepare_data():
    # Carga del dataset MNIST desde Keras
    mnist = tf.keras.datasets.mnist
    # Preparar los datos de entrenamiento y prueba
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalización de los valores de las imágenes, llevándolos entre 0 y 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Convertir las etiquetas en una representación categórica (one-hot encoding)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    # Ajustar las dimensiones de las imágenes para adaptarlas a la entrada de la red convolucional
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    
    # Mensaje de confirmación de carga de datos
    print("loaded data")
    
    # Devolver el conjunto de datos cargado y preprocesado
    return mnist, x_train, y_train, x_test, y_test

# Definición de función para construir y entrenar el modelo de red convolucional
def train_model(x_train, y_train):
    # Definir un modelo secuencial
    model = tf.keras.models.Sequential([
        # Capa convolucional: Aprender 64 filtros de 3x3 con activación ReLU
        tf.keras.layers.Conv2D(
            64,  # Número de filtros
            (3, 3),  # Tamaño de los filtros
            activation="relu",
            input_shape=(28, 28, 1)  # Dimensiones de las imágenes de entrada (28x28 píxeles, 1 canal - escala de grises)
        ),
        
        # Capa de Max-Pooling: Reduce las dimensiones usando ventanas de 2x2
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Aplanar los datos a una dimensión
        tf.keras.layers.Flatten(),  # Convierte el tensor 2D en un vector 1D
        
        # Añadir una capa densa con 128 unidades y activación ReLU
        tf.keras.layers.Dense(128, activation="relu"),
        
        # Aplicar dropout (desconexión aleatoria de neuronas) con un 50% de probabilidad para evitar sobreajuste
        tf.keras.layers.Dropout(0.5),
        
        # Añadir la capa de salida con 10 neuronas (una por dígito) y activación softmax para obtener probabilidades
        tf.keras.layers.Dense(10, activation="softmax")  # Capa de salida con 10 clases (dígitos del 0 al 9)
    ])
    
    # Compilar el modelo usando el optimizador Adam y la función de pérdida de entropía cruzada categórica
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Entrenar el modelo con los datos de entrenamiento durante 10 épocas
    model.fit(x_train, y_train, epochs=10)
    print("trained model")
    
    # Devolver el modelo entrenado
    return model

# Ejecución del código principal
if __name__ == '__main__':
    # Cargar los datos
    mnist, x_train, y_train, x_test, y_test = prepare_data()
    
    # Entrenar el modelo
    model = train_model(x_train, y_train)
    
    # Visualizar la arquitectura del modelo, mostrando las formas de las capas
    plot_model(model, show_shapes=True)
    
    # Evaluar el modelo en los datos de prueba y mostrar el rendimiento
    model.evaluate(x_test, y_test, verbose=2)
    
    # Guardar el modelo entrenado si se proporciona un nombre de archivo como argumento de la línea de comandos
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")
