import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('BankNote_Authentication.csv')

"""
#Para ver cuales pares se correlacionan más con la clase
sns.pairplot(df, hue='class')
plt.show()
correlations = df.corr(numeric_only=True)['class'].abs().sort_values(ascending=False)
print(correlations)
"""

X = df[['variance', 'curtosis']].values 
# X = df[['variance', 'entropy']].values
#X = df[['skewness', 'variance']].values
y = df['class'].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


w = np.zeros((X.shape[1], 1))  
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
    z = X_train.dot(w) + b
    a = sigmoid(z)
    weight_history.append(w.copy())
    bias_history.append(b)

    m = len(y_train)
    dw = (1/m) * X_train.T.dot(a - y_train)
    db = (1/m) * np.sum(a - y_train)

    w -= alpha * dw
    b -= alpha * db
    pred_labels = (a >= 0.5).astype(int)
    error = np.mean(pred_labels != y_train)
    train_loss.append(error)

    ax.clear()
    ax.scatter(X_train[y_train.ravel() == 0, 0], X_train[y_train.ravel() == 0, 1], color='blue', label='Clase 0')
    ax.scatter(X_train[y_train.ravel() == 1, 0], X_train[y_train.ravel() == 1, 1], color='red', label='Clase 1')

    x_vals = np.array(ax.get_xlim())
    if w[1] != 0: 
        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, 'k--', label=f'Epoch {epoch+1}')
    
    ax.set_title(f"Frontera de decisión - Epoch {epoch+1}")
    ax.set_xlabel('variance (normalizado)')
    ax.set_ylabel('skewness (normalizado)')
    ax.legend()
    plt.pause(0.01)

plt.ioff()
plt.show()

#Graficar la evolucion con cortes

x_vals = np.linspace(-2, 2, 100) 
plt.figure(figsize=(8,6))
plt.scatter(X_train[y_train.ravel() == 0, 0], X_train[y_train.ravel() == 0, 1], color='blue', label='Clase 0')
plt.scatter(X_train[y_train.ravel() == 1, 0], X_train[y_train.ravel() == 1, 1], color='red', label='Clase 1')

for i in range(0, epochs, 100):  
    w_tmp = weight_history[i]
    b_tmp = bias_history[i]
    if w_tmp[1] != 0:
        y_vals = -(w_tmp[0] * x_vals + b_tmp) / w_tmp[1]
        plt.plot(x_vals, y_vals, '--', label=f'Epoch {i+1}')

plt.title('Evolución de la frontera de decisión')
plt.xlabel('Variance')
plt.ylabel('Curtosis')
plt.legend()
plt.grid(True)
plt.show()

#Graficar el error de entrenamiento
plt.plot(range(epochs), train_loss, color='purple')
plt.xlabel('Época')
plt.ylabel('Error de clasificación')
plt.title('Error por época')
plt.grid(True)
plt.show()

z_test = np.dot(X_test, w) + b
y_pred_test = sigmoid(z_test)
y_pred_labels = (y_pred_test >= 0.5).astype(int)
test_error = np.mean(y_pred_labels != y_test)

print(f"\nError en conjunto de prueba: {test_error:.4f}")



"""
def predecir_billete(variance, curtosis):
    # Normalizar los datos como lo hiciste con los datos de entrenamiento
    datos = np.array([[variance, curtosis]])
    datos_normalizados = scaler.transform(datos)

    # Calcular la predicción
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
curtosis = input("\nIngresa el valor de curtosis: ")
# Supón que el usuario ingresa estos valores:
predecir_billete(variance, curtosis)

"""