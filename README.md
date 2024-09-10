# MNIST Convolutional Neural Network (CNN)

Este proyecto implementa una red neuronal convolucional (CNN) para clasificar imágenes del dataset MNIST. El modelo está construido usando TensorFlow y Keras, y el conjunto de datos MNIST contiene imágenes de dígitos escritos a mano (del 0 al 9). 

## Requisitos

Este proyecto requiere Python 3 y las siguientes dependencias:

- TensorFlow
- Keras (incluido con TensorFlow)
- Pydot
- Graphviz

Puedes instalarlos usando el siguiente comando:

```bash
pip install tensorflow pydot graphviz
````
## Estructura del Código

El código está organizado de la siguiente manera:

1. **prepare_data()**: 
   - Carga y prepara el conjunto de datos MNIST.
   - Normaliza los valores de píxeles de las imágenes, que originalmente son enteros de 0 a 255, para que estén entre 0 y 1.
   - Aplica una transformación a las etiquetas (one-hot encoding) para que el modelo pueda trabajar con ellas.
   - Ajusta las dimensiones de las imágenes para que tengan el formato correcto para la red convolucional (28x28x1).
   
2. **train_model(x_train, y_train)**:
   - Define y entrena un modelo CNN utilizando una arquitectura secuencial.
   - El modelo tiene una capa convolucional, una capa de max-pooling, una capa densa y una capa de salida con 10 neuronas (una para cada dígito del 0 al 9).
   - El modelo se entrena con el optimizador Adam y la función de pérdida de entropía cruzada categórica.

3. **Selección de Hiperparámetros y Justificación**:
   - **Número de filtros en la capa convolucional (64)**: Se seleccionó un valor relativamente estándar para este tipo de problemas de clasificación de imágenes. Este número de filtros ofrece un buen balance entre la capacidad de extraer características y la eficiencia computacional.
   - **Tamaño del filtro (3x3)**: Este tamaño es comúnmente utilizado en redes convolucionales debido a su capacidad para capturar características locales importantes en imágenes pequeñas como las del dataset MNIST.
   - **Función de activación ReLU**: ReLU es ideal para la mayoría de las capas ocultas de redes neuronales profundas porque introduce no linealidad y ayuda a mitigar problemas de desvanecimiento del gradiente.
   - **Max-Pooling (2x2)**: Esta técnica reduce las dimensiones de las características aprendidas mientras conserva información relevante. También ayuda a reducir el riesgo de sobreajuste, ya que simplifica la representación espacial de la imagen.
   - **Dropout (0.5)**: Se utiliza un valor de dropout del 50% en la capa densa para prevenir el sobreajuste, desconectando aleatoriamente la mitad de las neuronas durante el entrenamiento.
   - **Optimizer Adam**: Adam es un optimizador robusto que combina las ventajas de otros métodos (como RMSprop y SGD con momentum). Su capacidad de ajuste automático de la tasa de aprendizaje lo hace ideal para este tipo de problemas.

4. **Generalización del Modelo**:
   - El modelo está diseñado para generalizar bien en el conjunto de datos de prueba debido al uso de técnicas como la normalización de los datos, el dropout para prevenir el sobreajuste, y la reducción dimensional a través de max-pooling.
   - Tras el entrenamiento, el modelo se evalúa con datos de prueba que no ha visto antes, demostrando una alta capacidad de generalización al obtener buenos resultados tanto en precisión como en la métrica de pérdida.

5. **Visualización del Modelo**:
   - Visualiza la arquitectura de la CNN, mostrando las dimensiones de las capas mediante `plot_model`.
   - Visualiza la matriz de confusión.
   - Visualiza el contenido de la base de datos.

6. **Evaluación del modelo**:
   - Evalúa el modelo en el conjunto de prueba y muestra la precisión obtenida.

7. **Guardar el modelo entrenado**:
   - Si se proporciona un argumento en la línea de comandos, el modelo entrenado se guardará con el nombre de archivo proporcionado.

## Ejecución

Para ejecutar el código, sigue estos pasos:

1. Asegúrate de que todas las dependencias estén instaladas. Puedes instalarlas ejecutando:

```bash
pip install tensorflow pydot graphviz
```

2. Ejecuta el script principal usando el siguiente comando:

```bash
python handwriting.py nombre_del_modelo.h5
````

3. Para ejecutar el modelo y probarlo, se debe de ejecutar la siguiente línea en la terminal:

```bash
python recognition.py nombre_del_modelo.h5
````


