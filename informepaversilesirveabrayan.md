## Informe de Clasificador No Lineal (Red Neuronal con Una Capa Oculta)

### Descripción General

Este informe documenta la implementación de una red neuronal desde cero en Python, diseñada como un clasificador no lineal con una capa oculta. Esta red fue implementada en el contexto de una evaluación de Computación Blanda y aplicada a un problema de clasificación multiclase.

### Estructura de la Red

La red consta de:
- Una **capa de entrada** con dos neuronas (correspondiente a las coordenadas X, Y de entrada).
- Una **capa oculta** con 10 neuronas y función de activación ReLU.
- Una **capa de salida** con 3 neuronas (una por clase: purple, orange, green), con activación Softmax.

### Inicialización de Pesos

Se emplea la inicialización de He:

$$ w \sim \mathcal{N}(0, \sqrt{\frac{2}{n}}) $$

Donde `n` es el número de entradas a la capa. Esta técnica mejora la convergencia al entrenar redes profundas con ReLU.

### Funciones de Activación

#### ReLU (Rectified Linear Unit)

$$ \text{ReLU}(x) = \max(0, x) $$

La derivada usada en el backpropagation es:

$$
\text{ReLU}'(x) =
\begin{cases}
1 & \text{si } x > 0 \\
0 & \text{si } x \leq 0
\end{cases}
$$

#### Softmax

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Convierte la salida de la última capa en probabilidades.

### Cálculo de la Pérdida

Se utiliza la función de pérdida **log loss** para clasificación multiclase:

$$
\mathcal{L}(y, p) = - y \log(p) - (1 - y) \log(1 - p)
$$

Donde `y` es el vector objetivo y `p` la predicción.

### Propagación Hacia Adelante (Forward Pass)

1. Z1 = X * W1 + b1  
2. A1 = ReLU(Z1)  
3. Z2 = A1 * W2 + b2  
4. A2 = Softmax(Z2) => Predicciones

### Retropropagación (Backpropagation)

Se actualizan los pesos usando el descenso por gradiente estocástico:

1. Error en salida:  
$$ dZ2 = A2 - Y $$
$$ dW2 = A1^T \cdot dZ2 $$ 
$$ db2 = \sum dZ2 $$

2. Error en capa oculta:  
$$ dZ1 = (dZ2 \cdot W2) \odot \text{ReLU}'(Z1) $$ 
$$ dW1 = X^T \cdot dZ1 $$ 
$$ db1 = \sum dZ1 $$

3. Actualización de pesos:  
$$ W = W - \eta \cdot dW $$  
$$ b = b - \eta \cdot db $$

Donde $\eta$ es la tasa de aprendizaje.

### Entrenamiento

- Se realiza por **epoch**, iterando sobre los datos en batches de tamaño 1 (SGD).
- En cada iteración, se calcula el error y se actualizan los pesos.

### Resultados

Tras 1000 iteraciones de entrenamiento con tasa de aprendizaje 0.01, la red fue capaz de generalizar sobre un conjunto de prueba con entradas no vistas, clasificando con buena precisión las clases purple, orange y green.

### Predicciones
A continuación se muestran las predicciones que realizó la red neuronal después de ser entrenada:

| Target       | Predicción           |
|--------------|----------------------|
| (1, 0, 0)     | [1.00, 0.00, 0.00]   |
| (1, 0, 0)     | [0.78, 0.00, 0.22]   |
| (1, 0, 0)     | [1.00, 0.00, 0.00]   |
| (1, 0, 0)     | [0.98, 0.00, 0.02]   |
| (1, 0, 0)     | [1.00, 0.00, 0.00]   |
| (1, 0, 0)     | [0.39, 0.01, 0.60]   |
| (1, 0, 0)     | [1.00, 0.00, 0.00]   |
| (1, 0, 0)     | [1.00, 0.00, 0.00]   |
| (1, 0, 0)     | [1.00, 0.00, 0.00]   |
| (1, 0, 0)     | [0.95, 0.05, 0.00]   |
| (0, 1, 0)     | [0.00, 0.98, 0.02]   |
| (0, 1, 0)     | [0.00, 1.00, 0.00]   |
| (0, 1, 0)     | [0.00, 0.49, 0.51]   |
| (0, 1, 0)     | [0.00, 1.00, 0.00]   |
| (0, 1, 0)     | [0.00, 1.00, 0.00]   |
| (0, 1, 0)     | [0.00, 1.00, 0.00]   |
| (0, 1, 0)     | [0.00, 0.55, 0.45]   |
| (0, 1, 0)     | [0.00, 0.88, 0.12]   |
| (0, 1, 0)     | [0.07, 0.93, 0.00]   |
| (0, 1, 0)     | [0.00, 0.91, 0.09]   |
| (0, 0, 1)     | [0.05, 0.00, 0.95]   |
| (0, 0, 1)     | [0.00, 0.00, 1.00]   |
| (0, 0, 1)     | [0.00, 0.00, 1.00]   |
| (0, 0, 1)     | [0.00, 0.00, 1.00]   |
| (0, 0, 1)     | [0.00, 0.00, 1.00]   |
| (0, 0, 1)     | [0.00, 0.00, 1.00]   |
| (0, 0, 1)     | [0.00, 0.01, 0.99]   |
| (0, 0, 1)     | [0.00, 0.10, 0.90]   |
| (0, 0, 1)     | [0.00, 0.00, 1.00]   |
| (0, 0, 1)     | [0.00, 0.02, 0.98]   |

### Conclusión

La red neuronal implementada cumple con las características de un clasificador no lineal con una capa oculta. Utiliza ReLU, softmax, inicialización de He y entrenamiento mediante descenso por gradiente estocástico, cubriendo los fundamentos teóricos requeridos por la cátedra de Computación Blanda.

---

### Archivos Implementados

#### `RN.py`

Contiene la clase `CBNN`, con la arquitectura de la red neuronal, funciones de activación, forward pass, backpropagation y entrenamiento.

#### `punto1.py`

Define los datos de entrada y objetivos, instancia la red y ejecuta el entrenamiento y pruebas.

---

## Informe de Reconocimiento de Dígitos Manuscritos (MNIST)

### Descripción General

En este segundo punto se implementó una red neuronal completamente conectada, diseñada desde cero en Python, para reconocer dígitos escritos a mano utilizando el conjunto de datos **MNIST**. La red cuenta con una arquitectura de tres capas: entrada, oculta y salida.

### Estructura de la Red

- **Capa de entrada**: 784 neuronas (28x28 píxeles).
- **Capa oculta**: 20 neuronas, activación ReLU.
- **Capa de salida**: 10 neuronas (una por cada dígito del 0 al 9), activación softmax.

### Inicialización de Pesos

Los pesos de las capas se inicializan según la distribución:

$$
w \sim \mathcal{U}(-0.5, 0.5)
$$

No obstante, se implementó internamente la **inicialización de He** para mejorar la estabilidad del entrenamiento:

$$
w \sim \mathcal{N}(0, \sqrt{\frac{2}{n}})
$$

### Funciones de Activación

- **ReLU** en la capa oculta: favorece la propagación de gradientes.
- **Softmax** en la capa de salida: convierte los valores en probabilidades para clasificación multiclase.

### Función de Pérdida

Se emplea la pérdida de **log loss** (entropía cruzada multiclase):

$$
\mathcal{L}(y, p) = - y \log(p) - (1 - y) \log(1 - p)
$$

### Propagación hacia Adelante

1. $$ Z1 = X \cdot W1 + b1 $$
2. $$ A1 = \text{ReLU}(Z1) $$ 
3. $$ Z2 = A1 \cdot W2 + b2 $$
4. $$ A2 = \text{Softmax}(Z2) $$

### Retropropagación y Actualización de Pesos

- Se calcula el error de la salida y se propaga hacia atrás.
- Se actualizan los pesos usando el gradiente descendente con batches.

### Entrenamiento

- Se entrenó por 100 épocas con un **batch size de 32** y **tasa de aprendizaje 0.01**.
- Se imprimió la pérdida promedio por época.

### Resultados

El modelo fue capaz de aprender a reconocer dígitos manuscritos, realizando predicciones correctas en la mayoría de los casos sobre el conjunto de prueba de MNIST.

Se imprimieron las predicciones para una muestra de imágenes de test, comparando el valor real con la predicción del modelo.

### Prediciones

A continuación se muestra las predicciones de la red neuronal despues del entrenamiento con 20 datos del set de testeo, en el archivo punto2.py toma las prediciones que son una una matriz con 10 columnas donde cada columna determina un numero segun la posicion, y se toma el numero de la columna la cual tiene la mayor probabilidad entre todas las columnas.

| Target | Predicción |
|--------|------------|
| 7      | 7          |
| 2      | 2          |
| 1      | 1          |
| 0      | 0          |
| 4      | 4          |
| 1      | 1          |
| 4      | 4          |
| 9      | 9          |
| 5      | 5          |
| 9      | 9          |
| 0      | 0          |
| 6      | 6          |
| 9      | 9          |
| 0      | 0          |
| 1      | 1          |
| 5      | 5          |
| 9      | 9          |
| 7      | 7          |
| 3      | 3          |
| 4      | 4          |

### Conclusión

La red neuronal cumple con los requisitos del enunciado:
- Usa 784 neuronas de entrada, 20 ocultas y 10 de salida.
- Utiliza funciones de activación ReLU y Softmax.
- Entrenada desde cero con batch training y log loss.
- Reconoce correctamente dígitos manuscritos del dataset MNIST.

---

### Archivos Implementados

#### `RN.py`

Contiene la clase `CBNN`, donde se define la estructura de la red, funciones de activación, propagación hacia adelante, retropropagación y actualización de pesos.

#### `punto2.py`

Carga el conjunto de datos MNIST, normaliza las imágenes, define el modelo, entrena y realiza predicciones.

---