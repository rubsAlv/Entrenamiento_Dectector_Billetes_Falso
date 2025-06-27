import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('BankNote_Authentication.csv')
print(df.head())

#Para ver cuales pares se correlacionan más con la clase
sns.pairplot(df, hue='class')
plt.show()
correlations = df.corr(numeric_only=True)['class'].abs().sort_values(ascending=False)
print(correlations)


X = df[['variance']].values  
y = df['class'].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


w = np.zeros((1, 1))
b = 0
alpha = 0.1
epochs = 300
train_loss = []
weight_history = []
bias_history = []

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

plt.ion()
fig, ax = plt.subplots()
for epoch in range(epochs):
    z = np.dot(X_train, w) + b
    a = sigmoid(z)

    # Gradientes
    m = len(y_train)
    dw = (1/m) * np.dot(X_train.T, (a - y_train))
    db = (1/m) * np.sum(a - y_train)

    # Actualización
    w -= alpha * dw
    b -= alpha * db

    # Error
    pred_labels = (a >= 0.5).astype(int)
    error = np.mean(pred_labels != y_train)
    train_loss.append(error)

    # Visualización de la curva sigmoide
    ax.clear()
    ax.scatter(X_train, y_train, c=y_train, cmap='bwr', label='Datos')
    x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_vals = sigmoid(np.dot(x_vals, w) + b)
    ax.plot(x_vals, y_vals, color='black', label=f'Epoch {epoch+1}')
    ax.set_title(f'Curva sigmoide - Epoch {epoch+1}')
    ax.set_xlabel('Variance (normalizado)')
    ax.set_ylabel('Probabilidad de clase 1')
    ax.legend()
    plt.pause(0.01)

plt.ioff()
plt.show()

#Graficar la evolucion con cortes
plt.plot(range(epochs), train_loss, color='purple')
plt.xlabel('Época')
plt.ylabel('Error de clasificación')
plt.title('Error por época - Clasificador univariable')
plt.grid(True)
plt.show()

z_test = np.dot(X_test, w) + b
y_pred_test = sigmoid(z_test)
y_pred_labels = (y_pred_test >= 0.5).astype(int)
test_error = np.mean(y_pred_labels != y_test)
print(f"\n🔍 Error en conjunto de prueba: {test_error:.4f}")

def predecir_billete(variance):
    datos = np.array([[float(variance)]])
    datos_normalizados = scaler.transform(datos)
    z = np.dot(datos_normalizados, w) + b
    prob = sigmoid(z)
    clase = int(prob >= 0.5)

    print(f"\nProbabilidad de que sea VERDADERO: {prob[0][0]:.4f}")
    if clase == 1:
        print("Resultado: BILLETE VERDADERO")
    else:
        print("Resultado: BILLETE FALSO")

print("\n=== Predicción de billete ===")
valor = input("Ingresa el valor de variance: ")
predecir_billete(valor)