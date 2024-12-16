import numpy as np

class ProbabilisticNeuralNetwork:
    def __init__(self, sigma=1.0):
        # sigma - параметр розсіювання для функції Гаусса 
        self.sigma = sigma
        self.training_data = None
        self.training_labels = None

    def gaussian_kernel(self, x, xi):
        # Гауссова функція ядра.
        # x - поточний вектор входу.
        # xi - тренувальний вектор.
        distance = np.linalg.norm(x - xi)
        return np.exp(-distance**2 / (2 * self.sigma**2))

    def train(self, X_train, y_train):
        # Збереження тренувальних даних.
        self.training_data = X_train
        self.training_labels = y_train

    def predict(self, x_predict):
        # Виконання розпізнавання для вхідних даних.
        predictions = []
        for x in x_predict:
            # Вектор ймовірностей для кожного тренувального зразка
            probabilities = np.array([
                self.gaussian_kernel(x, xi) for xi in self.training_data
            ])
            weighted_outputs = probabilities * self.training_labels
            predicted_value = np.sum(weighted_outputs) / np.sum(probabilities)
            predictions.append(predicted_value)

        return np.array(predictions)

# Функція для генерації навчальних даних на основі функції у=х1+х2
def generate_training_data(n_samples=1000):
    np.random.seed(50)  # Створення одних і тих самих чисел
    X_train = np.random.rand(n_samples, 2)  
    y_train = np.sum(X_train, axis=1) 
    return X_train, y_train

# Генеруємо дані
X_train, y_train = generate_training_data()

# Ініціалізуємо та тренуємо PNN
pnn = ProbabilisticNeuralNetwork(sigma=0.1)
pnn.train(X_train, y_train)

# Тестові дані
X_test = np.array([[0.2, 0.3], [0.4, 0.5], [0.9, 0.1], [0.6, 0.2], [0.7, 0.8]])
predictions = pnn.predict(X_test)

# Виведення результатів
print("Тестові дані:\n", X_test)
print("Прогнозовані результати:\n", predictions)
