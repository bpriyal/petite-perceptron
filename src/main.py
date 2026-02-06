import numpy as np
from perceptron import Perceptron

def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, epochs=10)
    model.fit(X, y)

    print("Weights:", model.weights)
    print("Bias:", model.bias)

    predictions = model.predict(X)
    for x, p, t in zip(X, predictions, y):
        print(f"{x} -> {p} (target={t})")

if __name__ == "__main__":
    main()
