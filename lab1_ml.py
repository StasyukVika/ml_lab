import numpy as np

# Cигмоїдальна функція активації 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Похідна сигмоїдальної функції активації 
def sigmoid_derivative(x):
    return x * (1 - x)

# Частина 1: Реалізація класичного нейрону
class ClassicalNeuron:
    def __init__(self):
        self.weight = np.random.uniform(-1, 1)
        self.bias = np.random.uniform(-1, 1)

    def train(self, x, y, learning_rate=0.1):
        output = sigmoid(x * self.weight + self.bias)
        error = y - output
        
        # Оновлення ваги та зміщення
        self.weight += learning_rate * error * sigmoid_derivative(output) * x
        self.bias += learning_rate * error * sigmoid_derivative(output)

    def predict(self, x):
        return sigmoid(x * self.weight + self.bias)

# Частина 2: Елементарний двошаровий персептрон із структурою 1-1-1
class SimplePerceptron:
    def __init__(self):
        self.weight_1 = np.random.uniform(-1, 1)
        self.bias_1 = np.random.uniform(-1, 1)
        self.weight_2 = np.random.uniform(-1, 1)
        self.bias_2 = np.random.uniform(-1, 1)

    def train(self, x, y, learning_rate=0.1):
        hidden_output = sigmoid(x * self.weight_1 + self.bias_1)
        final_output = sigmoid(hidden_output * self.weight_2 + self.bias_2)
        error = y - final_output
        
        # Оновлення ваги та зміщення
        d_output = error * sigmoid_derivative(final_output)
        d_hidden = d_output * self.weight_2 * sigmoid_derivative(hidden_output)
        
        self.weight_2 += learning_rate * d_output * hidden_output
        self.bias_2 += learning_rate * d_output
        self.weight_1 += learning_rate * d_hidden * x
        self.bias_1 += learning_rate * d_hidden

    def predict(self, x):
        hidden_output = sigmoid(x * self.weight_1 + self.bias_1)
        return sigmoid(hidden_output * self.weight_2 + self.bias_2)

# Частина 3: Двошаровий персептрон із структурою 2-3-1
class TwoLayerPerceptron:
    def __init__(self):
        self.weights_1 = np.random.uniform(-1, 1, (2, 3))
        self.biases_1 = np.random.uniform(-1, 1, 3)
        self.weights_2 = np.random.uniform(-1, 1, 3)
        self.bias_2 = np.random.uniform(-1, 1)

    def train(self, x, y, learning_rate=0.1):
        hidden_input = np.dot(x, self.weights_1) + self.biases_1
        hidden_output = sigmoid(hidden_input)
        final_output = sigmoid(np.dot(hidden_output, self.weights_2) + self.bias_2)
        error = y - final_output
        
        # Оновлення ваги та зміщення
        d_output = error * sigmoid_derivative(final_output)
        d_hidden = d_output * self.weights_2 * sigmoid_derivative(hidden_output)
        
        self.weights_2 += learning_rate * d_output * hidden_output
        self.bias_2 += learning_rate * d_output
        
        for i in range(3):
            self.weights_1[:, i] += learning_rate * d_hidden[i] * x
        self.biases_1 += learning_rate * d_hidden

    def predict(self, x):
        hidden_input = np.dot(x, self.weights_1) + self.biases_1
        hidden_output = sigmoid(hidden_input)
        return sigmoid(np.dot(hidden_output, self.weights_2) + self.bias_2)

# Приклад використання для піддослідної функції х1+х2=у 
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 2])

# Навчання двошарового персептрону
perceptron = TwoLayerPerceptron()
for epoch in range(10000):
    for x, y in zip(x_train, y_train):
        perceptron.train(x, y)

# Тестування
for x in x_train:  
    print(f"Input x: {x}, Predicted: {perceptron.predict(x):.3f}")
