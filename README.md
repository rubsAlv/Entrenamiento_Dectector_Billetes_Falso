# Entrenamiento_Dectector_Billetes_Falso
Clasificador lineal con descenso de gradiente
## 📊 Dataset

Este proyecto utiliza el dataset [Banknote Authentication - UCI](https://www.kaggle.com/datasets/shantanuss/banknote-authentication-uci) disponible en Kaggle, creado por Shantanu.

Créditos: Shantanu (Kaggle), basado en datos del UCI Machine Learning Repository.

Explicación del Dataset: 

Varience (Varianza):  refleja qué tan variados son los píxeles en ciertas zonas del billete.
Skewness (Asimetría): detecta desequilibrios en la intensidad de la imagen.
Curtosis: detecta si la imagen tiene muchos valores extremos o picos en la intensidad.
Entropy (entropy): indica la cantidad de información o detalle en la textura del billete.

Se cargan los datos desde el archivo .csv. 
Posteriormente, se utiliza sns.pairplot para visualizar las relaciones entre las variables y analizar qué pares de atributos presentan una mayor correlación con la clase objetivo.

Se asignan las variables predictoras a X y la variable objetivo a Y, eligiendo las características ‘variance’ y ‘curtosis’. Los datos se normalizan para asegurar una escala uniforme. Luego, se divide el conjunto de datos utilizando train_test_split, reservando el 70% para entrenamiento y el 30% para prueba.

Para cada época se inicializa z que es la combinación lineal de entradas y se aplica la función sigmoide a z, dando una probabilidad de pertenecer a la clase 1.
Se guardan los pesos y bias actuales.
Calcula los gradientes del error con respecto a los pesos y el bias.
Se actualizan los parámetros usando el descenso de gradiente.
Convierte las probabilidades a clases 0 o 1. Calcula el porcentaje de errores y lo guarda.


Se calcula la salida del modelo (z_test) utilizando los pesos entrenados, se aplica la función sigmoide para obtener probabilidades, y se convierte esa salida en etiquetas binarias.
 Finalmente, se compara la predicción con las etiquetas reales del conjunto de prueba (y_test) y se calcula el error de clasificación, que representa el porcentaje de predicciones incorrectas.