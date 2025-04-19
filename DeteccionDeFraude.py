import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carga solo una parte del dataset para evitar cuelgues
df = pd.read_csv('c:/Users/DrakoS3M/OneDrive/Desktop/programacion/proyectos/c++/.vscode/PY/Detección de fraude/creditcard.csv', nrows=10000)

# Separar características y etiquetas
X = df.drop('Class', axis=1)
y = df['Class']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo más liviano (menos árboles)
clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)

# Resultados
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
