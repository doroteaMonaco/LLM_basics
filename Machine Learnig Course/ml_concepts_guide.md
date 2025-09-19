# Guida ai Concetti di Machine Learning: Quando Usare Cosa

## Indice
1. [Preprocessing dei Dati](#preprocessing-dei-dati)
2. [Visualizzazione dei Dati](#visualizzazione-dei-dati)
3. [Clustering](#clustering)
4. [Regressione](#regressione)
5. [Classificazione](#classificazione)
6. [Valutazione del Modello](#valutazione-del-modello)
7. [Matrice di Decisione Riassuntiva](#matrice-di-decisione-riassuntiva)

---

## Preprocessing dei Dati

### Scopo
Pulire, trasformare e preparare i dati grezzi per gli algoritmi di machine learning.

### Quando Usarlo
- **Sempre** - Primo passo in qualsiasi pipeline ML
- Quando si hanno valori mancanti
- Quando le feature hanno scale diverse
- Quando i dati categorici necessitano di codifica
- Quando i dati contengono outlier

### Tecniche Principali

#### StandardScaler
**Usare quando:**
- Le feature hanno unità/scale diverse
- Si usano algoritmi sensibili alla scala (SVM, KNN, Reti Neurali)
- I dati seguono una distribuzione normale

**Vantaggi:**
- Media zero, varianza unitaria
- Preserva le relazioni tra osservazioni

**Svantaggi:**
- Sensibile agli outlier
- Assume distribuzione normale

**Codice:**
```python
from sklearn.preprocessing import StandardScaler

# Inizializzazione
scaler = StandardScaler()

# Fit sui dati di training
scaler.fit(X_train)

# Trasformazione
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit e transform in un passaggio (solo su training)
X_train_scaled = scaler.fit_transform(X_train)
```

#### MinMaxScaler
**Usare quando:**
- Servono valori limitati (range 0-1)
- I dati non seguono distribuzione normale
- Si lavora con reti neurali

**Vantaggi:**
- Preserva i valori zero
- Range di output limitato

**Svantaggi:**
- Sensibile agli outlier
- Può comprimere i dati normali

**Codice:**
```python
from sklearn.preprocessing import MinMaxScaler

# Inizializzazione con range personalizzato
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit e transform
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inversione della trasformazione
X_original = scaler.inverse_transform(X_scaled)
```

#### RobustScaler
**Usare quando:**
- I dati contengono molti outlier
- Serve un metodo di scaling robusto

**Vantaggi:**
- Usa mediana e IQR (resistente agli outlier)
- Migliore per dati asimmetrici

**Svantaggi:**
- Potrebbe non scalare nel range 0-1
- Meno interpretabile

**Codice:**
```python
from sklearn.preprocessing import RobustScaler

# Inizializzazione
scaler = RobustScaler()

# Fit e transform
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Tecniche di Codifica

**Label Encoding:**
- Usare per dati categorici ordinali
- Quando le categorie hanno un ordine naturale

**Codice:**
```python
from sklearn.preprocessing import LabelEncoder

# Inizializzazione
encoder = LabelEncoder()

# Fit e transform
y_encoded = encoder.fit_transform(y)

# Classi disponibili
print(encoder.classes_)

# Inversione
y_original = encoder.inverse_transform(y_encoded)
```

**One-Hot Encoding:**
- Usare per dati categorici nominali
- Quando le categorie sono indipendenti
- Evitare con feature ad alta cardinalità

**Codice:**
```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Con sklearn
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = encoder.fit_transform(X_categorical)

# Con pandas (più semplice)
X_encoded = pd.get_dummies(df, columns=['categoria'], drop_first=True)
```

#### Gestione Valori Mancanti

**Codice:**
```python
from sklearn.impute import SimpleImputer

# Strategie: 'mean', 'median', 'most_frequent', 'constant'
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Per valori categorici
imputer_cat = SimpleImputer(strategy='most_frequent')
X_cat_imputed = imputer_cat.fit_transform(X_categorical)
```

---

## Visualizzazione dei Dati

### Scopo
Comprendere pattern, relazioni e distribuzioni nei dati.

### Quando Usare Ogni Tipo di Grafico

#### Grafici Base di Matplotlib

**Line Plot:**
- Dati di serie temporali
- Relazioni continue
- Analisi di trend

**Codice:**
```python
import matplotlib.pyplot as plt

# Grafico base
plt.plot(x, y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Titolo')
plt.grid(True)
plt.show()

# Multiple linee
plt.plot(x, y1, label='Serie 1')
plt.plot(x, y2, label='Serie 2')
plt.legend()
plt.show()
```

**Scatter Plot:**
- Relazione tra due variabili continue
- Rilevazione di outlier
- Analisi di correlazione

**Codice:**
```python
# Scatter base
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Con colori per categorie
plt.scatter(x, y, c=categories, cmap='viridis')
plt.colorbar()
plt.show()

# Dimensione variabile
plt.scatter(x, y, s=sizes, alpha=0.6)
plt.show()
```

**Istogramma:**
- Distribuzione di una singola variabile
- Comprensione della dispersione dei dati
- Rilevazione di asimmetria

**Codice:**
```python
# Istogramma base
plt.hist(data, bins=30, alpha=0.7)
plt.xlabel('Valori')
plt.ylabel('Frequenza')
plt.show()

# Multiple distribuzioni
plt.hist([data1, data2], bins=30, alpha=0.7, label=['Gruppo 1', 'Gruppo 2'])
plt.legend()
plt.show()
```

**Bar Plot:**
- Confronto di dati categorici
- Conteggi di frequenza
- Confronti tra gruppi

**Codice:**
```python
# Bar plot verticale
plt.bar(categories, values)
plt.xlabel('Categorie')
plt.ylabel('Valori')
plt.show()

# Bar plot orizzontale
plt.barh(categories, values)
plt.show()

# Bar plot raggruppate
x = np.arange(len(categories))
plt.bar(x - 0.2, values1, 0.4, label='Gruppo 1')
plt.bar(x + 0.2, values2, 0.4, label='Gruppo 2')
plt.xticks(x, categories)
plt.legend()
plt.show()
```

**Box Plot:**
- Riassunto della distribuzione (mediana, quartili)
- Rilevazione di outlier
- Confronto di distribuzioni tra gruppi

**Codice:**
```python
# Box plot singolo
plt.boxplot(data)
plt.ylabel('Valori')
plt.show()

# Multiple box plot
plt.boxplot([data1, data2, data3], labels=['Gruppo 1', 'Gruppo 2', 'Gruppo 3'])
plt.show()
```

#### Visualizzazioni Avanzate

**Heatmap:**
- Matrici di correlazione
- Matrici di confusione
- Pattern di dati 2D

**Codice:**
```python
import seaborn as sns

# Heatmap correlazione
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()

# Heatmap matrice confusione
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.show()
```

**Pair Plot:**
- Relazioni tra multiple variabili
- Selezione di feature
- Scoperta di pattern

**Codice:**
```python
import seaborn as sns

# Pair plot base
sns.pairplot(df)
plt.show()

# Con categorie
sns.pairplot(df, hue='target_column')
plt.show()

# Solo alcune colonne
sns.pairplot(df[['col1', 'col2', 'col3', 'target']], hue='target')
plt.show()
```

**Grafici 3D:**
- Relazioni tridimensionali
- Strutture dati complesse
- Visualizzazione di cluster

**Codice:**
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter 3D
ax.scatter(x, y, z, c=colors)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Surface plot
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()
```

---

## Clustering

### Scopo
Raggruppare punti dati simili senza esempi etichettati (apprendimento non supervisionato).

### Quando Usare il Clustering
- Segmentazione clienti
- Esplorazione dati
- Rilevazione anomalie
- Compressione dati
- Preprocessing per apprendimento supervisionato

### Algoritmi

#### K-Means
**Usare quando:**
- Si aspettano cluster sferici
- Si conosce il numero approssimativo di cluster
- Servono risultati veloci
- Dataset grandi

**Vantaggi:**
- Semplice e veloce
- Funziona bene con cluster sferici
- Scala bene su dataset grandi

**Svantaggi:**
- Necessario specificare k
- Sensibile all'inizializzazione
- Assume cluster sferici
- Sensibile agli outlier

**Codice:**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Inizializzazione
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)

# Fit e predizione
labels = kmeans.fit_predict(X)

# Solo fit
kmeans.fit(X)
labels = kmeans.labels_

# Centroidi
centroids = kmeans.cluster_centers_

# Inerzia (somma distanze quadrate dai centroidi)
inertia = kmeans.inertia_

# Metodo del gomito per trovare k ottimale
inertias = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Numero di cluster (k)')
plt.ylabel('Inerzia')
plt.show()

# Silhouette score
silhouette_avg = silhouette_score(X, labels)
```

#### DBSCAN
**Usare quando:**
- I cluster hanno forme arbitrarie
- Densità di cluster variabili
- Necessaria rilevazione outlier
- Non si conosce il numero di cluster

**Vantaggi:**
- Trova cluster di forma arbitraria
- Determina automaticamente i cluster
- Robusto agli outlier
- Identifica punti di rumore

**Svantaggi:**
- Sensibile agli iperparametri
- Difficoltà con densità variabili
- Intensivo in memoria per dataset grandi

**Codice:**
```python
from sklearn.cluster import DBSCAN
import numpy as np

# Inizializzazione
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fit e predizione
labels = dbscan.fit_predict(X)

# Numero di cluster (escludendo noise -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Numero cluster stimato: {n_clusters}")
print(f"Numero punti noise: {n_noise}")

# Tuning parametri
from sklearn.neighbors import NearestNeighbors

# Metodo per trovare eps ottimale
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.ylabel('4th Nearest Neighbor Distance')
plt.show()
```

#### Clustering Gerarchico
**Usare quando:**
- Serve la gerarchia dei cluster
- Dataset piccoli-medi
- Si vogliono esplorare diversi numeri di cluster
- Importante l'interpretabilità

**Vantaggi:**
- Non serve specificare il numero di cluster
- Fornisce gerarchia
- Risultati deterministici

**Svantaggi:**
- Computazionalmente costoso O(n³)
- Sensibile agli outlier
- Difficile con dataset grandi

**Codice:**
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

# Clustering agglomerativo
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clustering.fit_predict(X)

# Dendrogramma
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.title('Dendrogramma')
plt.show()

# Taglio del dendrogramma
labels = fcluster(linkage_matrix, t=3, criterion='maxclust')

# Diversi metodi di linkage: 'ward', 'complete', 'average', 'single'
```

---

## Regressione

### Scopo
Predire valori numerici continui basati su feature di input.

### Quando Usare la Regressione
- Predire prezzi case
- Previsioni di vendite
- Valutazione del rischio
- Modellazione scientifica

### Algoritmi

#### Regressione Lineare
**Usare quando:**
- Relazione lineare tra feature e target
- Serve un modello semplice e interpretabile
- Dataset piccoli-medi
- Le feature non sono altamente correlate

**Vantaggi:**
- Altamente interpretabile
- Training e predizione veloci
- Nessun tuning di iperparametri
- Buon modello baseline

**Svantaggi:**
- Assume relazioni lineari
- Sensibile agli outlier
- Problemi con multicollinearità
- Può sottoadattare dati complessi

**Codice:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inizializzazione e training
model = LinearRegression()
model.fit(X_train, y_train)

# Predizioni
y_pred = model.predict(X_test)

# Coefficienti e intercetta
print("Coefficienti:", model.coef_)
print("Intercetta:", model.intercept_)

# Valutazione
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}")
print(f"R²: {r2}")
```

#### Ridge Regression
**Usare quando:**
- Presente multicollinearità
- Molte feature
- Si vuole prevenire overfitting
- Serve riduzione delle feature

**Vantaggi:**
- Gestisce multicollinearità
- Previene overfitting
- Mantiene tutte le feature
- Soluzioni stabili

**Svantaggi:**
- Meno interpretabile della lineare
- Non fa selezione feature
- Richiede tuning iperparametri

**Codice:**
```python
from sklearn.linear_model import Ridge, RidgeCV

# Con alpha fisso
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

# Cross-validation per alpha ottimale
alphas = [0.1, 1.0, 10.0, 100.0]
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Alpha ottimale: {ridge_cv.alpha_}")

# Predizioni
y_pred = ridge_cv.predict(X_test)
```

#### Lasso Regression
**Usare quando:**
- Necessaria selezione feature
- Si desiderano soluzioni sparse
- Molte feature irrilevanti
- Importante l'interpretabilità

**Vantaggi:**
- Selezione automatica feature
- Soluzioni sparse
- Buona per dati ad alta dimensionalità
- Interpretabile

**Svantaggi:**
- Può rimuovere arbitrariamente feature correlate
- Instabile con feature altamente correlate
- Richiede tuning iperparametri

**Codice:**
```python
from sklearn.linear_model import Lasso, LassoCV

# Con alpha fisso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

# Cross-validation per alpha ottimale
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)

print(f"Alpha ottimale: {lasso_cv.alpha_}")

# Feature selezionate (coefficienti non zero)
selected_features = X.columns[lasso_cv.coef_ != 0]
print("Feature selezionate:", selected_features.tolist())
```

#### Random Forest Regression
**Usare quando:**
- Relazioni non lineari
- Tipi di dati misti
- Si vuole importanza delle feature
- Serve un modello robusto

**Vantaggi:**
- Gestisce non linearità
- Score di importanza feature
- Robusto agli outlier
- Funziona con tipi di dati misti

**Svantaggi:**
- Meno interpretabile
- Può overfittare con dataset piccoli
- Intensivo in memoria
- Molti iperparametri

**Codice:**
```python
from sklearn.ensemble import RandomForestRegressor

# Inizializzazione
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Training
rf.fit(X_train, y_train)

# Predizioni
y_pred = rf.predict(X_test)

# Importanza delle feature
feature_importance = rf.feature_importances_
feature_names = X.columns

# Visualizzazione importanza
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Importanza')
plt.title('Importanza delle Feature')
plt.show()

# Grid search per iperparametri
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Migliori parametri:", grid_search.best_params_)
```

---

## Classificazione

### Scopo
Predire etichette di classe discrete basate su feature di input.

### Quando Usare la Classificazione
- Rilevazione spam email
- Diagnosi medica
- Riconoscimento immagini
- Predizione churn clienti

### Algoritmi

#### Regressione Logistica
**Usare quando:**
- Problemi binari o multiclasse
- Confini di decisione lineari
- Necessarie stime di probabilità
- Importante l'interpretabilità

**Vantaggi:**
- Fornisce probabilità
- Veloce e semplice
- Nessun tuning iperparametri
- Buon modello baseline

**Svantaggi:**
- Assume confini lineari
- Sensibile agli outlier
- Può sottoadattare dati complessi

**Codice:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Inizializzazione
log_reg = LogisticRegression(random_state=42)

# Training
log_reg.fit(X_train, y_train)

# Predizioni
y_pred = log_reg.predict(X_test)

# Probabilità di predizione
y_prob = log_reg.predict_proba(X_test)

# Valutazione
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Coefficienti (per interpretabilità)
print("Coefficienti:", log_reg.coef_)
```

#### Alberi di Decisione
**Usare quando:**
- Interpretabilità cruciale
- Tipi di dati misti
- Relazioni non lineari
- Importanti interazioni tra feature

**Vantaggi:**
- Altamente interpretabile
- Gestisce tipi di dati misti
- Cattura interazioni
- Nessun preprocessing necessario

**Svantaggi:**
- Soggetto a overfitting
- Instabile (alta varianza)
- Bias verso feature con più livelli

**Codice:**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Inizializzazione
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Training
dt.fit(X_train, y_train)

# Predizioni
y_pred = dt.predict(X_test)

# Visualizzazione albero
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['Classe_0', 'Classe_1'], filled=True)
plt.show()

# Importanza feature
feature_importance = dt.feature_importances_
print("Importanza feature:", dict(zip(X.columns, feature_importance)))

# Pruning per evitare overfitting
from sklearn.tree import DecisionTreeClassifier

# Cost complexity pruning
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Training con diversi alpha
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
```

#### Random Forest Classification
**Usare quando:**
- Necessaria alta accuratezza
- Tipi di dati misti
- Si vuole importanza feature
- Richiesto modello robusto

**Vantaggi:**
- Alta accuratezza
- Robusto a overfitting
- Importanza feature
- Gestisce valori mancanti

**Svantaggi:**
- Meno interpretabile
- Intensivo in memoria
- Può overfittare dati rumorosi

**Codice:**
```python
from sklearn.ensemble import RandomForestClassifier

# Inizializzazione
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42
)

# Training
rf_clf.fit(X_train, y_train)

# Predizioni
y_pred = rf_clf.predict(X_test)
y_prob = rf_clf.predict_proba(X_test)

# Importanza feature
feature_importance = rf_clf.feature_importances_

# Out-of-bag score (se oob_score=True)
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)
print(f"OOB Score: {rf_oob.oob_score_}")
```

#### Support Vector Machine (SVM)
**Usare quando:**
- Dati ad alta dimensionalità
- Margine chiaro tra classi
- Confini non lineari (con kernel)
- Dataset piccoli-medi

**Vantaggi:**
- Efficace in alte dimensioni
- Efficiente in memoria
- Versatile (diversi kernel)
- Funziona bene con margini chiari

**Svantaggi:**
- Lento su dataset grandi
- Sensibile alla scala delle feature
- Nessuna stima di probabilità
- Molti iperparametri

**Codice:**
```python
from sklearn.svm import SVC

# SVM lineare
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

# SVM con kernel RBF
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

# SVM con probabilità
svm_prob = SVC(kernel='rbf', probability=True, random_state=42)
svm_prob.fit(X_train, y_train)
y_prob = svm_prob.predict_proba(X_test)

# Grid search per iperparametri
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Migliori parametri:", grid_search.best_params_)
```

---

## Valutazione del Modello

### Metriche per Regressione

**Codice:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Mean Squared Error
mse = mean_squared_error(y_true, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)

# R² Score
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
```

### Metriche per Classificazione

**Codice:**
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve)

# Metriche base
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Classification report completo
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ROC AUC (per classificazione binaria)
if len(np.unique(y_true)) == 2:
    auc = roc_auc_score(y_true, y_prob[:, 1])
    print(f"ROC AUC: {auc}")
    
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
```

### Cross-Validation

**Codice:**
```python
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold

# Cross-validation semplice
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Cross-validation con multiple metriche
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"CV {metric}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Stratified K-Fold per dataset sbilanciati
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
```

---

## Matrice di Decisione Riassuntiva

| Tipo Problema | Dimensione Dati | Necessità Interpretabilità | Necessità Accuratezza | Approccio Raccomandato |
|---------------|-----------------|----------------------------|----------------------|------------------------|
| **Regressione** | Piccoli | Alta | Media | Linear/Ridge/Lasso |
| **Regressione** | Grandi | Bassa | Alta | Random Forest/XGBoost |
| **Classificazione** | Piccoli | Alta | Media | Logistic/Decision Tree |
| **Classificazione** | Grandi | Bassa | Alta | Random Forest/SVM |
| **Clustering** | Qualsiasi | Alta | N/A | Gerarchico |
| **Clustering** | Grandi | Bassa | N/A | K-Means |
| **Esplorazione** | Qualsiasi | Alta | N/A | Visualizzazione + EDA |

### Pipeline Completa

**Codice:**
```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Pipeline con preprocessing e modello
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Grid search sulla pipeline
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10],
    'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Migliore pipeline
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
```

### Linee Guida Generali

1. **Inizia Semplice**: Comincia con algoritmi base (Regressione Lineare/Logistica)
2. **Preprocessing Sempre**: Scala le feature, gestisci valori mancanti
3. **Visualizza Prima**: Comprendi i dati prima di modellare
4. **Cross-Valida**: Valida sempre le performance del modello
5. **Considera i Trade-off**: Bilancia accuratezza, interpretabilità e costo computazionale
6. **Itera**: Il machine learning è un processo iterativo

### Errori Comuni da Evitare

- Saltare il preprocessing dei dati
- Non validare le performance del modello
- Overfitting sui dati di training
- Ignorare la scala delle feature per algoritmi basati su distanza
- Usare modelli complessi quando bastano quelli semplici
- Non considerare il contesto business e le necessità di interpretabilità
- Data leakage nel preprocessing (fare fit su tutto il dataset)
- Non gestire il dataset sbilanciato nelle classificazioni