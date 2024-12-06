import numpy as np

# Función de activación
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptrón simple
class Perceptron:
    def __init__(self, input_size, learning_rate=0.8):
        # Inicialización de pesos aleatorios (entre -1 y 1)
        self.weights = np.random.uniform(-1, 1, input_size + 1) 
        
        self.learning_rate = learning_rate

    # Función de predicción
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  
        return step_function(summation)  

    # Función de entrenamiento
    def train(self, training_data, labels, epochs=100):
        for _ in range(epochs):  
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction 

                if error != 0: 
                    self.weights[1:] += self.learning_rate * error * inputs 
                    self.weights[0] += self.learning_rate * error  

# Datos de entrenamiento para compuertas lógicas
logic_gates = {
    "AND": {
        "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "labels": np.array([0, 0, 0, 1])
    },
    "OR": {
        "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "labels": np.array([0, 1, 1, 1])
    },
    "XNOR": {
        "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "labels": np.array([1, 0, 0, 0])
    }
}

# Entrenamiento para cada compuerta lógica
for gate, data in logic_gates.items():
    print(f"\nEntrenando para compuerta {gate}")
    perceptron = Perceptron(input_size=2)
    perceptron.train(data["inputs"], data["labels"], epochs=10)  # Entrenamiento del perceptrón
    print(f"Pesos finales: {perceptron.weights}")

    # Verificación
    print("Resultados:")
    for input_data in data["inputs"]:
        print(f"Entrada: {input_data}, Salida: {perceptron.predict(input_data)}")
