import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('BankNote_Authentication.csv')

X = df[['variance', 'skewness', 'curtosis', 'entropy']].values
y = df['class'].values.reshape(-1, 1)

#Normalización de los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

#División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Datos de inicializacion
w = np.zeros((X.shape[1], 1))  # (4,1)
b = 0
alpha = 0.1
epochs = 200
m = X_train.shape[0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


train_loss = []

for epoch in range(epochs):
    z = np.dot(X_train, w) + b
    a = sigmoid(z)

    # Gradientes
    dw = (1/m) * np.dot(X_train.T, (a - y_train))
    db = (1/m) * np.sum(a - y_train)

    # Actualizar
    w -= alpha * dw
    b -= alpha * db

    # Calcular error de clasificación
    pred_labels = (a >= 0.5).astype(int)
    error = np.mean(pred_labels != y_train)
    train_loss.append(error)

    if epoch % 20 == 0:
        print(f"Época {epoch}: error de clasificación = {error:.4f}")

plt.plot(range(epochs), train_loss, color='purple')
plt.xlabel('Época')
plt.ylabel('Error de clasificación')
plt.title('Error por época - Gradiente descendente multivariable')
plt.grid(True)
plt.show()


z_test = np.dot(X_test, w) + b
y_pred_test = sigmoid(z_test)
y_pred_labels = (y_pred_test >= 0.5).astype(int)
test_error = np.mean(y_pred_labels != y_test)

print(f"\n🔍 Error en conjunto de prueba: {test_error:.4f}")


def predecir_billete(variance, skewness, curtosis, entropy):
    
    datos = np.array([[variance, skewness, curtosis, entropy]])
    datos_normalizados = scaler.transform(datos)
    
    z = np.dot(datos_normalizados, w) + b
    prob = 1 / (1 + np.exp(-z))
    clase = int(prob >= 0.5)

    print(f"\nProbabilidad de que sea VERDADERO: {prob[0][0]:.4f}")
    if clase == 1:
        print("Resultado: BILLETE VERDADERO")
    else:
        print("Resultado: BILLETE FALSO")
        
print("===============================================")
print("\nPredicción de billete:")
variance = input("\nIngresa el valor de variance: ")
skewness = input("\nIngresa el valor de skewness: ")
curtosis = input("\nIngresa el valor de curtosis: ")
entropy = input("\nIngresa el valor de entropy: ")

predecir_billete(variance, skewness, curtosis, entropy)

