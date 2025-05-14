from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from RN import CBNN
import numpy as np
import matplotlib.pyplot as plt


# Carga datos: ya vienen divididos en entrenamiento y prueba
(x_train, targets), (test_inputs, test_targets) = mnist.load_data()


# Mostrar la primera imagen del conjunto de entrenamiento
# print()
# plt.imshow(x_train[0], cmap='gray')
# plt.title(f"Etiqueta: {targets[0]}")
# plt.axis('off')
# plt.show()


print(x_train.shape)  # (60000, 28, 28)
print(targets.shape)  # (60000,)
# print(x_train[0])

x_train = x_train/255
test_inputs = test_inputs/255

x_train = x_train.reshape(-1, 784)
test_inputs = test_inputs.reshape(-1, 784)
# print(x_train[0])
test_show = test_targets
targets = to_categorical(targets, num_classes=10)
test_targets = to_categorical(test_targets, num_classes=10)


print(x_train.shape)

rn = CBNN(x_train, targets, 1000, 100, 20, 0.01)

rn.training()

print('Predictions:')
prediction = rn.predict(np.array(test_inputs))

print(f'targets        prediction')
for i in range(len(test_targets)):
    indice = np.argmax(prediction[i])
    print(f'{test_show[i]}:    {indice}') 