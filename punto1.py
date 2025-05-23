from RN import CBNN
import numpy as np
import matplotlib.pyplot as plt

x_train = [ (0.0000, 0.3929), (0.5484, 0.7500), (0.0645, 0.5714), (0.5806,0.5714),
            (0.2258, 0.8929), (0.4839, 0.2500), (0.3226, 0.2143), (0.7742, 0.8214),
            (0.4516, 0.5000), (0.4194, 0.0357), (0.4839, 0.2500), (0.3226, 0.7143),
            (0.5806, 0.5000), (0.5484, 0.1071), (0.6129, 0.6429), (0.6774, 0.1786),
            (0.2258, 0.8214), (0.7419, 0.1429), (0.6452, 1.0000), (0.8387, 0.2500),
            (0.9677, 0.3214), (0.3226, 0.4643), (0.3871, 0.5357), (0.3548, 0.1429),
            (0.3548, 0.6429), (0.1935, 0.4643), (0.4516, 0.3929), (0.4839, 0.6071),
            (0.6129, 0.6786), (0.2258, 0.6071), (0.5161, 0.3214), (0.5484, 0.6786),
            (0.3871, 0.8571), (0.6452, 0.6071), (0.1935, 0.3929), (0.6452, 0.3929),
            (0.6774, 0.4643), (0.3226, 0.2857), (0.7419, 0.7143), (0.7419, 0.3214),
            (1.0000, 0.3929), (0.8065, 0.3929), (0.1935, 0.5000), (0.1613, 0.8214),
            (0.2903, 0.9286), (0.3548, 0.0000), (0.2903, 0.6786), (0.5484, 0.9643),
            (0.4194, 0.1786), (0.2581, 0.2500), (0.3226, 0.7143), (0.5161, 0.3929),
            (0.2903, 0.6429), (0.5484, 0.9286), (0.2581, 0.3214), (0.0968, 0.5000),
            (0.6129, 0.7857), (0.0968, 0.3214), (0.6452, 0.9286), (0.8065, 0.7500)]

purple = (1, 0, 0)
orange = (0, 1, 0)
green = (0, 0, 1)
targets = [ purple, orange, purple, orange, green,  purple, 
            purple, green,  orange, purple, purple, green, 
            orange, purple, orange, purple, green,  purple, 
            green,  purple, purple, orange, orange, purple, 
            orange, purple, orange, orange, orange, green, 
            orange, orange, green,  orange, purple, orange, 
            orange, purple, orange, orange, purple, orange, 
            green,  green,  green,  purple, green,  green, 
            purple, purple, green,  orange, green,  green, 
            purple, purple, green,  purple, green,  green]

rn = CBNN(x_train = x_train, targets = targets, n_iter = 1000, n_hidden = 10, lr = 0.01)

rn.training(batch_size=1)

test_inputs = [ (0.0000, 0.3929), (0.0645, 0.5714), (0.0968, 0.3214),
                (0.0968, 0.5000), (0.2581, 0.3214), (0.1935, 0.4643), 
                (0.2581, 0.2500), (0.1935, 0.3929), (0.3226, 0.2143), 
                (0.4839, 0.2500), (0.3226, 0.4643), (0.3871, 0.5357), 
                (0.3548, 0.6429), (0.4516, 0.5000), (0.4516, 0.3929),
                (0.5161, 0.3929), (0.5484, 0.7500), (0.6129, 0.6786), 
                (0.5161, 0.3214), (0.5484, 0.6786), (0.1935, 0.5000), 
                (0.2258, 0.6071), (0.3226, 0.7143), (0.2903, 0.6786), 
                (0.3226, 0.7143), (0.2258, 0.8214), (0.2903, 0.6429),
                (0.6129, 0.7857), (0.7742, 0.8214), (0.8065, 0.7500)]

test_targets = [purple, purple, purple, 
                purple, purple, purple, 
                purple, purple, purple, 
                purple, orange, orange, 
                orange, orange, orange, 
                orange, orange, orange,
                orange, orange, green, 
                green,  green,  green, 
                green,  green,  green, 
                green,  green,  green]

print('Predictions:')
prediction = rn.predict(np.array(test_inputs))

print(f'targets        prediction')
for i in range(len(test_targets)):
    print(f'{test_targets[i]}:    {np.round(prediction[i], decimals= 2)}') 