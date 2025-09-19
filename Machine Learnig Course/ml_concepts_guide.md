# Guida ai Concetti di Machine Learning: Quando Usare Cosa

## Indice
1. [Preprocessing dei Dati](#preprocessing-dei-dati)
2. [Visualizzazione dei Dati](#visualizzazione-dei-dati)
3. [Clustering](#clustering)
4. [Regressione](#regressione)
5. [Classificazione](#classificazione)
6. [Valutazione del Modello](#valutazione-del-modello)
7. [Gradient Boosting](#gradient-boosting)
8. [Time Series e Anomaly Detection](#time-series-e-anomaly-detection)
9. [Regolarizzazione](#regolarizzazione)
10. [Matrice di Decisione Riassuntiva](#matrice-di-decisione-riassuntiva)

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

#### Gestione Valori Mancanti

**Quando usare:**
- Sempre come primo step nell'analisi dei dati
- Prima di applicare qualsiasi algoritmo di ML

**Codice:**
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Controllo valori mancanti
print("Valori mancanti per colonna:")
print(df.isnull().sum())

# Visualizzazione pattern dei valori mancanti
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
plt.title('Pattern dei Valori Mancanti')
plt.show()

# Strategia 1: Rimozione
# Rimuovi righe con valori mancanti
df_clean = df.dropna()

# Rimuovi colonne con troppi valori mancanti (>50%)
threshold = len(df) * 0.5
df_clean = df.dropna(thresh=threshold, axis=1)

# Strategia 2: Imputazione semplice
# Per variabili numeriche
imputer_num = SimpleImputer(strategy='mean')  # o 'median', 'most_frequent'
df[['col1', 'col2']] = imputer_num.fit_transform(df[['col1', 'col2']])

# Per variabili categoriche
imputer_cat = SimpleImputer(strategy='most_frequent')
df[['cat_col']] = imputer_cat.fit_transform(df[['cat_col']])

# Strategia 3: Imputazione avanzata
# KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

# Iterative Imputer (simile a MICE)
iterative_imputer = IterativeImputer(random_state=42)
df_iterative = pd.DataFrame(iterative_imputer.fit_transform(df), columns=df.columns)

# Strategia 4: Imputazione con informazione aggiuntiva
# Crea indicatore di valore mancante
df['col1_was_missing'] = df['col1'].isnull().astype(int)
df['col1'].fillna(df['col1'].mean(), inplace=True)
```

#### Gestione Outliers

**Quando usare:**
- Quando sospetti presenza di valori anomali
- Prima dell'addestramento di modelli sensibili agli outlier
- Come parte dell'analisi esplorativa

**Codice:**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

# Metodo 1: Z-Score
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

outliers_z = detect_outliers_zscore(df['numeric_column'])
print(f"Outliers rilevati con Z-score: {outliers_z.sum()}")

# Metodo 2: IQR (Interquartile Range)
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

outliers_iqr = detect_outliers_iqr(df['numeric_column'])
print(f"Outliers rilevati con IQR: {outliers_iqr.sum()}")

# Metodo 3: Isolation Forest (per dataset multivariati)
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers_iso = iso_forest.fit_predict(df[numeric_features])
outliers_iso = outliers_iso == -1

# Visualizzazione outliers
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Boxplot
axes[0].boxplot(df['numeric_column'])
axes[0].set_title('Boxplot - Rilevamento Outliers')

# Scatterplot con outliers evidenziati
axes[1].scatter(df.index[~outliers_iqr], df['numeric_column'][~outliers_iqr], 
                c='blue', alpha=0.6, label='Normal')
axes[1].scatter(df.index[outliers_iqr], df['numeric_column'][outliers_iqr], 
                c='red', alpha=0.8, label='Outliers')
axes[1].set_title('Outliers con IQR')
axes[1].legend()

# Distribuzione
axes[2].hist(df['numeric_column'], bins=30, alpha=0.7)
axes[2].axvline(df['numeric_column'].mean(), color='red', linestyle='--', label='Media')
axes[2].axvline(df['numeric_column'].median(), color='green', linestyle='--', label='Mediana')
axes[2].set_title('Distribuzione')
axes[2].legend()

plt.tight_layout()
plt.show()

# Gestione outliers
# Opzione 1: Rimozione
df_no_outliers = df[~outliers_iqr]

# Opzione 2: Capping (limitazione ai percentili)
lower_percentile = df['numeric_column'].quantile(0.05)
upper_percentile = df['numeric_column'].quantile(0.95)
df['numeric_column_capped'] = df['numeric_column'].clip(lower_percentile, upper_percentile)

# Opzione 3: Trasformazione logaritmica
df['numeric_column_log'] = np.log1p(df['numeric_column'])  # log1p per gestire valori zero
```

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

# Encoding delle variabili categoriche più avanzato
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Label Encoding (per variabili target o ordinali)
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Ordinal Encoding (per variabili ordinali)
ordinal_encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['ordinal_encoded'] = ordinal_encoder.fit_transform(df[['ordinal_feature']])

# Preprocessing automatico con ColumnTransformer
numeric_features = ['age', 'income']
categorical_features = ['gender', 'education']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'  # mantieni altre colonne
)

X_processed = preprocessor.fit_transform(X)
```

#### Feature Engineering

**Quando usare:**
- Per creare nuove feature informative dai dati esistenti
- Per migliorare le performance del modello
- Per catturare relazioni non lineari

**Codice:**
```python
# Feature da date/time
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter

# Feature binarie
df['is_high_value'] = (df['value'] > df['value'].median()).astype(int)

# Feature di interazione
df['feature1_x_feature2'] = df['feature1'] * df['feature2']
df['feature1_div_feature2'] = df['feature1'] / (df['feature2'] + 1e-8)  # evita divisione per zero

# Feature aggregate (groupby)
df['category_mean'] = df.groupby('category')['target'].transform('mean')
df['category_std'] = df.groupby('category')['target'].transform('std')
df['category_count'] = df.groupby('category')['target'].transform('count')

# Feature polinomiali
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['feature1', 'feature2']])
poly_feature_names = poly.get_feature_names_out(['feature1', 'feature2'])

# Feature di testo (se hai colonne di testo)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Lunghezza del testo
df['text_length'] = df['text_column'].str.len()
df['word_count'] = df['text_column'].str.split().str.len()

# TF-IDF
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_features = tfidf.fit_transform(df['text_column'])
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

## Gradient Boosting

### Scopo
Il Gradient Boosting è una tecnica di ensemble learning che combina sequenzialmente modelli "deboli" (tipicamente alberi di decisione) per creare un modello "forte". Ogni nuovo modello viene addestrato per correggere gli errori dei modelli precedenti.

### Come Funziona
1. Si inizia con un modello semplice (spesso la media per regressione)
2. Si calcola l'errore residuo (differenza tra predizione e valore reale)
3. Si addestra un nuovo modello per predire questi residui
4. Si aggiunge il nuovo modello al modello esistente
5. Si ripete fino a convergenza o numero massimo di iterazioni

### Quando Usare Gradient Boosting
- Necessaria alta accuratezza predittiva
- Dati strutturati/tabulari
- Dataset di medie-grandi dimensioni
- Competizioni di machine learning
- Quando Random Forest non è sufficiente
- Problemi di classificazione e regressione complessi

### Vantaggi Generali
- Eccellenti performance predittive
- Gestisce bene overfitting con regolarizzazione
- Funziona con feature miste (numeriche e categoriche)
- Robusto agli outlier
- Fornisce importanza delle feature

### Svantaggi Generali
- Computazionalmente intensivo
- Molti iperparametri da tuning
- Meno interpretabile di modelli semplici
- Può overfittare se mal configurato
- Sensibile al rumore nei dati

---

### Algoritmi di Gradient Boosting

#### GradientBoostingClassifier/Regressor (Scikit-Learn)

**Quando usare:**
- Prototipazione e apprendimento
- Dataset piccoli-medi
- Quando serve interpretabilità del processo
- Baseline per confronti

**Vantaggi:**
- Incluso in scikit-learn (nessuna installazione aggiuntiva)
- Buona documentazione e integrazione
- Perfetto per apprendimento

**Svantaggi:**
- Più lento rispetto ad altre implementazioni
- Meno ottimizzato per performance
- Limitato in termini di funzionalità avanzate

**Codice Classificazione:**
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Dataset sintetico
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)

# Modello
sgb = GradientBoostingClassifier()

# Cross-validation
scores = cross_val_score(sgb, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print(f'Accuracy media: {np.mean(scores):.3f}, std: {np.std(scores):.3f}')

# Fit e predizione
sgb.fit(X, y)
row = [[2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, 
        -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]]
yhat = sgb.predict(row)
print(f'Predizione: {yhat[0]}')
```

**Codice Regressione:**
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression

# Dataset sintetico
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)

# Modello
sgb = GradientBoostingRegressor()

# Cross-validation
scores = cross_val_score(sgb, X, y, scoring='neg_mean_absolute_error', cv=10, n_jobs=-1)
print(f'Neg MAE media: {np.mean(scores):.3f}, std: {np.std(scores):.3f}')

# Fit e predizione
sgb.fit(X, y)
row = [[2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, 
        -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = sgb.predict(row)
print(f'Predizione: {yhat[0]:.3f}')
```

#### HistGradientBoostingClassifier/Regressor (Scikit-Learn)

**Quando usare:**
- Alternative più veloce ai GradientBoosting standard
- Dataset grandi
- Quando serve integrazione con scikit-learn ma con migliori performance

**Vantaggi:**
- Più veloce del GradientBoosting standard
- Discretizzazione automatica in istogrammi
- Gestisce meglio feature categoriche
- Integrato in scikit-learn

**Svantaggi:**
- Meno controllo fine rispetto a XGBoost/LightGBM
- Relativamente nuovo
- Meno opzioni di personalizzazione

**Come funziona:**
Discretizza le feature continue in istogrammi (es. 255 bucket), rendendo la ricerca degli split più efficiente.

**Codice:**
```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Classificazione
model = HistGradientBoostingClassifier()
scores = cross_val_score(model, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print(f'Accuracy media: {np.mean(scores):.3f}, std: {np.std(scores):.3f}')

# Regressione
model_reg = HistGradientBoostingRegressor()
scores_reg = cross_val_score(model_reg, X, y, scoring='neg_mean_absolute_error', cv=10, n_jobs=-1)
print(f'Neg MAE media: {np.mean(scores_reg):.3f}, std: {np.std(scores_reg):.3f}')
```

#### XGBoost (eXtreme Gradient Boosting)

**Quando usare:**
- Massima performance su dati strutturati
- Competizioni di machine learning
- Produzione con dataset grandi
- Quando serve il meglio in termini di accuratezza

**Vantaggi:**
- Eccellenti performance
- Altamente ottimizzato
- Supporto GPU
- Regolarizzazione avanzata
- Gestione automatica valori mancanti
- Parallelizzazione efficiente

**Svantaggi:**
- Molti iperparametri
- Richiede tuning estensivo
- Può overfittare facilmente
- Installazione aggiuntiva necessaria

**Caratteristiche Uniche:**
- Regolarizzazione L1 e L2 integrate
- Tree pruning intelligente
- Supporto per objective functions personalizzate
- Excellent per problemi di ranking

**Codice Classificazione:**
```python
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Griglia parametri
parameter_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2], 
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5, 1],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 1, 10]
}

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
model = XGBClassifier()

# Randomized search per ottimizzazione
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameter_grid,
    cv=5,
    n_iter=30,
    scoring='f1', 
    n_jobs=-1
)

random_search.fit(X, y)
scores = cross_val_score(random_search.best_estimator_, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print(f'Accuracy media: {np.mean(scores):.3f}, std: {np.std(scores):.3f}')

# Predizione
row = [[2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, 
        -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]]
yhat = random_search.best_estimator_.predict(row)
print(f'Predizione: {yhat[0]}')
```

**Codice Regressione:**
```python
from xgboost import XGBRegressor

parameter_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2], 
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 1, 10]
}

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
model = XGBRegressor(objective='reg:squarederror')

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameter_grid, 
    cv=5, 
    n_iter=30, 
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

random_search.fit(X, y)
scores = cross_val_score(random_search.best_estimator_, X, y, scoring='neg_mean_absolute_error', cv=10, n_jobs=-1)
print(f'Neg MAE: {np.mean(scores):.3f}, std: {np.std(scores):.3f}')

# Predizione
row = [2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, 
       -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]
row = np.asarray(row).reshape((1, len(row)))
yhat = random_search.best_estimator_.predict(row)
print(f'Predizione: {yhat[0]}')
```

#### LightGBM (Light Gradient Boosting Machine)

**Quando usare:**
- Dataset molto grandi
- Memoria limitata
- Quando serve velocità di training
- Problemi con molte feature

**Vantaggi:**
- Molto veloce su dataset grandi
- Uso efficiente della memoria
- Leaf-wise tree growth (più efficiente)
- Ottima gestione feature categoriche
- Supporto GPU eccellente

**Svantaggi:**
- Può overfittare su dataset piccoli
- Leaf-wise growth può essere instabile
- Meno stabile di XGBoost su alcuni problemi

**Caratteristiche Uniche:**
- Crescita leaf-wise invece di level-wise
- Gestione nativa di feature categoriche
- Ottimizzazione per velocità estrema
- Network communication per distributed training

**Codice Classificazione:**
```python
from lightgbm import LGBMClassifier
import pandas as pd

parameter_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2], 
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 1, 10]
}

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
model = LGBMClassifier()

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameter_grid,
    cv=5,
    n_iter=30,
    scoring='f1', 
    n_jobs=-1
)

random_search.fit(X, y)
scores = cross_val_score(random_search.best_estimator_, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print(f'Accuracy media: {np.mean(scores):.3f}, std: {np.std(scores):.3f}')

# Predizione con DataFrame (importante per LightGBM)
row = [2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, 
       -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]
rows = pd.DataFrame([row], columns=[f'feature{i}' for i in range(len(row))])
yhat = random_search.best_estimator_.predict(rows)
print(f'Predizione: {yhat[0]}')
```

**Codice Regressione:**
```python
from lightgbm import LGBMRegressor

parameter_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 1, 10]
}

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
model = LGBMRegressor()

random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=parameter_grid, 
    n_jobs=-1, 
    n_iter=30, 
    cv=5, 
    scoring='neg_mean_absolute_error'
)

random_search.fit(X, y)
scores = cross_val_score(random_search.best_estimator_, X, y, scoring='neg_mean_absolute_error', cv=10, n_jobs=-1)
print(f'Neg MAE: {np.mean(scores):.3f}, std: {np.std(scores):.3f}')

# Predizione
row = [2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, 
       -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]
rows = pd.DataFrame([row], columns=[f'feature {i}' for i in range(len(row))])
yhat = random_search.best_estimator_.predict(rows)
print(f'Predizione: {yhat[0]}')
```

#### CatBoost

**Quando usare:**
- Molte feature categoriche
- Vuoi ridurre feature engineering
- Dataset con categorical features ad alta cardinalità
- Quando serve un modello robusto out-of-the-box

**Vantaggi:**
- Gestione eccellente di feature categoriche
- Meno preprocessing necessario
- Robusto contro overfitting
- Configurazione automatica intelligente
- Ottima gestione missing values

**Svantaggi:**
- Più lento di LightGBM
- Meno controllo fine rispetto a XGBoost
- Installazione aggiuntiva necessaria
- Documentazione meno estesa

**Caratteristiche Uniche:**
- Ordered boosting per evitare target leakage
- Gestione automatica feature categoriche
- Built-in regularization techniques
- Symmetric trees per performance

**Codice Classificazione:**
```python
from catboost import CatBoostClassifier

parameter_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2], 
    'subsample': [0.6, 0.8, 1.0],
    'reg_lambda': [0.1, 1, 10]
}

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
model = CatBoostClassifier(verbose=False)  # verbose=False per silenziare output

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameter_grid,
    cv=5,
    n_iter=30,
    scoring='f1', 
    n_jobs=-1
)

random_search.fit(X, y)
scores = cross_val_score(random_search.best_estimator_, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print(f'Accuracy media: {np.mean(scores):.3f}, std: {np.std(scores):.3f}')

# Predizione
row = [2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, 
       -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]
rows = pd.DataFrame([row], columns=[f'feature{i}' for i in range(len(row))])
yhat = random_search.best_estimator_.predict(rows)
print(f'Predizione: {yhat[0]}')
```

**Codice Regressione:**
```python
from catboost import CatBoostRegressor

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
model = CatBoostRegressor(verbose=False)

parameter_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2], 
    'subsample': [0.6, 0.8, 1.0],
    'reg_lambda': [0.1, 1, 10]
}

random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=parameter_grid, 
    n_iter=30, 
    n_jobs=-1, 
    cv=5, 
    scoring='neg_mean_absolute_error'
)

random_search.fit(X, y)
scores = cross_val_score(random_search.best_estimator_, X, y, scoring='neg_mean_absolute_error', n_jobs=-1, cv=5)
print(f'Neg MAE: {np.mean(scores):.3f}, std: {np.std(scores):.3f}')

# Predizione
row = [2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, 
       -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]
rows = pd.DataFrame([row], columns=[f'feature {i}' for i in range(len(row))])
yhat = random_search.best_estimator_.predict(rows)
print(f'Predizione: {yhat[0]}')
```

### Comparazione degli Algoritmi di Gradient Boosting

| Algoritmo | Velocità | Memoria | Feature Categoriche | Overfitting | Tuning Necessario |
|-----------|----------|---------|-------------------|-------------|------------------|
| **Scikit-Learn GB** | Lenta | Media | Manuale | Medio | Basso |
| **HistGB** | Media | Bassa | Limitato | Medio | Basso |
| **XGBoost** | Media | Media | Manuale | Alto | Alto |
| **LightGBM** | Veloce | Bassa | Buono | Alto | Medio |
| **CatBoost** | Medio | Media | Eccellente | Basso | Basso |

### Consigli per l'Uso

1. **Per iniziare**: Scikit-Learn GradientBoosting o HistGradientBoosting
2. **Per produzione con dati strutturati**: XGBoost
3. **Per dataset enormi**: LightGBM
4. **Per molte feature categoriche**: CatBoost
5. **Per competizioni**: XGBoost o LightGBM con tuning estensivo

### Iperparametri Comuni

**Parametri principali da tuning:**
- `n_estimators`: Numero di alberi (100-1000+)
- `learning_rate`: Tasso di apprendimento (0.01-0.3)
- `max_depth`: Profondità massima alberi (3-10)
- `subsample`: Frazione di campioni per albero (0.6-1.0)
- `reg_alpha`, `reg_lambda`: Regolarizzazione L1 e L2

**Strategia di tuning:**
1. Inizia con parametri default
2. Usa RandomizedSearchCV per esplorazione iniziale
3. Affina con GridSearchCV su range ristretti
4. Monitora overfitting con validation curves
5. Usa early stopping quando disponibile

## Regolarizzazione

### Cos'è la Regolarizzazione
La regolarizzazione è una tecnica per prevenire l'overfitting aggiungendo un termine di penalità alla funzione di costo del modello. Questo termine penalizza la complessità del modello, favorendo soluzioni più semplici e generalizzabili.

### Tipi di Regolarizzazione

#### Ridge Regression (Regolarizzazione L2)
Aggiunge una penalità proporzionale alla somma dei quadrati dei coefficienti.

**Quando usare:**
- Quando hai multicollinearità tra le feature
- Quando vuoi ridurre tutti i coefficienti ma non portarli a zero
- Come primo approccio alla regolarizzazione

**Codice:**
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import RidgeCV
import numpy as np
import matplotlib.pyplot as plt

# Ridge Regression con cross-validation per scegliere alpha
alphas = np.logspace(-6, 6, 200)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Miglior alpha: {ridge_cv.alpha_}")

# Predizione
y_pred_ridge = ridge_cv.predict(X_test)

# Visualizzazione dei coefficienti per diversi valori di alpha
alphas_plot = np.logspace(-6, 6, 50)
coefficients = []

for alpha in alphas_plot:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefficients.append(ridge.coef_)

coefficients = np.array(coefficients)

plt.figure(figsize=(10, 6))
for i in range(coefficients.shape[1]):
    plt.plot(alphas_plot, coefficients[:, i], label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficienti')
plt.title('Ridge Path: Evoluzione dei Coefficienti')
plt.legend()
plt.grid(True)
plt.show()
```

#### Lasso Regression (Regolarizzazione L1)
Aggiunge una penalità proporzionale alla somma dei valori assoluti dei coefficienti.

**Quando usare:**
- Quando vuoi feature selection automatica
- Quando sospetti che molte feature siano irrilevanti
- Quando vuoi un modello sparso (molti coefficienti = 0)

**Codice:**
```python
from sklearn.linear_model import Lasso, LassoCV

# Lasso con cross-validation
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

print(f"Miglior alpha: {lasso_cv.alpha_}")
print(f"Feature selezionate: {np.sum(lasso_cv.coef_ != 0)}")

# Coefficienti selezionati
feature_names = [f'Feature_{i}' for i in range(len(lasso_cv.coef_))]
selected_features = [(name, coef) for name, coef in zip(feature_names, lasso_cv.coef_) if coef != 0]

print("Feature selezionate e loro coefficienti:")
for name, coef in selected_features:
    print(f"{name}: {coef:.4f}")

# Lasso Path
from sklearn.linear_model import lasso_path
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=alphas_plot)

plt.figure(figsize=(10, 6))
for i in range(coefs_lasso.shape[0]):
    plt.plot(alphas_lasso, coefs_lasso[i], label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficienti')
plt.title('Lasso Path: Feature Selection')
plt.legend()
plt.grid(True)
plt.show()
```

#### Elastic Net (Combinazione L1 + L2)
Combina Ridge e Lasso, usando entrambe le penalità.

**Quando usare:**
- Quando hai gruppi di feature correlate
- Quando vuoi un compromesso tra Ridge e Lasso
- Dataset con più feature che osservazioni

**Codice:**
```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Elastic Net con cross-validation
l1_ratios = np.linspace(0.1, 0.9, 9)  # Mixing parameter tra L1 e L2
elastic_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5, random_state=42)
elastic_cv.fit(X_train, y_train)

print(f"Miglior alpha: {elastic_cv.alpha_}")
print(f"Miglior l1_ratio: {elastic_cv.l1_ratio_}")
print(f"Feature selezionate: {np.sum(elastic_cv.coef_ != 0)}")

# Confronto tra Ridge, Lasso e Elastic Net
models = {
    'Ridge': ridge_cv,
    'Lasso': lasso_cv,
    'Elastic Net': elastic_cv
}

from sklearn.metrics import mean_squared_error, r2_score

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n_features = np.sum(model.coef_ != 0)
    
    results[name] = {
        'MSE': mse,
        'R²': r2,
        'Features Used': n_features,
        'Alpha': model.alpha_
    }

# Tabella comparativa
import pandas as pd
comparison_df = pd.DataFrame(results).T
print("\nConfronto Modelli Regolarizzati:")
print(comparison_df)
```

### Regolarizzazione in Altri Algoritmi

#### Support Vector Machines
```python
from sklearn.svm import SVC, SVR

# SVM con regolarizzazione C
# C basso = più regolarizzazione
svm_model = SVC(C=1.0, kernel='rbf')
svm_model.fit(X_train, y_train)

# Grid search per trovare miglior C
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000]}
svm_grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
svm_grid.fit(X_train, y_train)
print(f"Miglior C: {svm_grid.best_params_['C']}")
```

#### Random Forest e Decision Trees
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Random Forest con regolarizzazione attraverso parametri strutturali
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,           # Limita profondità
    min_samples_split=10,   # Minimo per split
    min_samples_leaf=5,     # Minimo per foglia
    max_features='sqrt',    # Subset di features
    random_state=42
)

rf_model.fit(X_train, y_train)

# Decision Tree con pruning
dt_model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    ccp_alpha=0.01,  # Cost complexity pruning
    random_state=42
)

dt_model.fit(X_train, y_train)
```

#### Neural Networks
```python
from sklearn.neural_network import MLPRegressor

# Neural Network con regolarizzazione
nn_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    alpha=0.001,        # Regolarizzazione L2
    early_stopping=True, # Stop quando validation loss non migliora
    validation_fraction=0.2,
    random_state=42
)

nn_model.fit(X_train, y_train)
```

### Regolarizzazione in Gradient Boosting

```python
# XGBoost con regolarizzazione
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    reg_alpha=0.1,      # Regolarizzazione L1
    reg_lambda=1.0,     # Regolarizzazione L2
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,      # Regolarizzazione tramite sampling
    colsample_bytree=0.8, # Regolarizzazione feature sampling
    early_stopping_rounds=10,
    random_state=42
)

xgb_model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)

# LightGBM con regolarizzazione
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    reg_alpha=0.1,
    reg_lambda=1.0,
    max_depth=6,
    learning_rate=0.1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    early_stopping_rounds=10,
    random_state=42
)

lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='rmse',
              verbose=False)
```

### Tecniche di Regolarizzazione Avanzate

#### Dropout (per Neural Networks)
```python
from sklearn.neural_network import MLPRegressor

# Simulazione di dropout tramite regolarizzazione forte
nn_dropout = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),
    alpha=0.01,  # Regolarizzazione più forte simula dropout
    learning_rate_init=0.001,
    early_stopping=True,
    random_state=42
)
```

#### Cross-Validation per Selezione Regolarizzazione
```python
def compare_regularization_methods(X_train, X_test, y_train, y_test):
    """
    Confronta diversi metodi di regolarizzazione
    """
    results = {}
    
    # Ridge
    ridge = RidgeCV(alphas=np.logspace(-6, 6, 50), cv=5)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    results['Ridge'] = {
        'MSE': mean_squared_error(y_test, ridge_pred),
        'R²': r2_score(y_test, ridge_pred),
        'Alpha': ridge.alpha_,
        'Features': len(ridge.coef_)
    }
    
    # Lasso
    lasso = LassoCV(alphas=np.logspace(-6, 6, 50), cv=5)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    results['Lasso'] = {
        'MSE': mean_squared_error(y_test, lasso_pred),
        'R²': r2_score(y_test, lasso_pred),
        'Alpha': lasso.alpha_,
        'Features': np.sum(lasso.coef_ != 0)
    }
    
    # Elastic Net
    elastic = ElasticNetCV(alphas=np.logspace(-6, 6, 50), 
                          l1_ratio=np.linspace(0.1, 0.9, 9), cv=5)
    elastic.fit(X_train, y_train)
    elastic_pred = elastic.predict(X_test)
    results['ElasticNet'] = {
        'MSE': mean_squared_error(y_test, elastic_pred),
        'R²': r2_score(y_test, elastic_pred),
        'Alpha': elastic.alpha_,
        'Features': np.sum(elastic.coef_ != 0)
    }
    
    return pd.DataFrame(results).T

# Esempio di utilizzo
comparison = compare_regularization_methods(X_train, X_test, y_train, y_test)
print(comparison)
```

### Quando Usare Ogni Tipo di Regolarizzazione

| Situazione | Metodo Raccomandato | Motivo |
|-----------|-------------------|--------|
| **Multicollinearità** | Ridge (L2) | Riduce tutti i coefficienti proporzionalmente |
| **Feature Selection** | Lasso (L1) | Porta coefficienti irrilevanti a zero |
| **Molte feature correlate** | Elastic Net | Combina benefici di L1 e L2 |
| **Dataset piccolo** | Ridge | Meno aggressivo, mantiene tutte le feature |
| **Molte feature irrilevanti** | Lasso | Selezione automatica delle feature |
| **Neural Networks** | Dropout + L2 | Previene overfitting in reti profonde |
| **Tree-based** | Strutturale | max_depth, min_samples_split, etc. |
| **Gradient Boosting** | L1 + L2 + Early Stopping | Controllo completo dell'overfitting |

---

## Time Series e Anomaly Detection

### Introduzione alle Serie Temporali
Le serie temporali sono sequenze di dati raccolti in ordine cronologico. Nel contesto industriale e dell'IoT, l'analisi delle serie temporali è fondamentale per il monitoraggio, la manutenzione predittiva e il rilevamento di anomalie.

### Feature Extraction da Serie Temporali

#### Libreria TSFEL (Time Series Feature Extraction Library)
TSFEL è una libreria Python specializzata nell'estrazione di caratteristiche da serie temporali.

```python
import pandas as pd
import numpy as np
import tsfel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione di TSFEL
cfg = tsfel.get_features_by_domain()

# Esempio di estrazione features da dati di sensori
def extract_features_from_timeseries(data, window_size=100, overlap=0.5):
    """
    Estrae features da serie temporali usando TSFEL
    
    Parameters:
    - data: DataFrame con colonne temporali
    - window_size: dimensione della finestra per l'estrazione
    - overlap: sovrapposizione tra finestre consecutive
    """
    
    # Estrazione features con finestre scorrevoli
    features = tsfel.time_series_features_extractor(
        cfg, 
        data, 
        window_size=window_size,
        overlap=overlap,
        verbose=0
    )
    
    return features

# Esempio di utilizzo con dati di robot industriale
# Simulazione dati multi-frequenza
def simulate_robot_data(n_samples=1000, freq_hz=100):
    """Simula dati di sensori di robot industriale"""
    time = np.linspace(0, n_samples/freq_hz, n_samples)
    
    # Sensori di accelerometro (3 assi)
    accel_x = np.sin(2*np.pi*5*time) + 0.1*np.random.normal(size=n_samples)
    accel_y = np.cos(2*np.pi*3*time) + 0.1*np.random.normal(size=n_samples)
    accel_z = 0.5*np.sin(2*np.pi*8*time) + 0.05*np.random.normal(size=n_samples)
    
    # Sensori di giroscopio (3 assi)
    gyro_x = 0.3*np.sin(2*np.pi*2*time) + 0.05*np.random.normal(size=n_samples)
    gyro_y = 0.4*np.cos(2*np.pi*4*time) + 0.05*np.random.normal(size=n_samples)
    gyro_z = 0.2*np.sin(2*np.pi*6*time) + 0.03*np.random.normal(size=n_samples)
    
    df = pd.DataFrame({
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'timestamp': time
    })
    
    return df

# Generazione dati di esempio per diverse frequenze
data_1hz = simulate_robot_data(100, 1)
data_10hz = simulate_robot_data(1000, 10)
data_100hz = simulate_robot_data(10000, 100)
data_200hz = simulate_robot_data(20000, 200)

print("Dati generati per frequenze: 1Hz, 10Hz, 100Hz, 200Hz")
```

### Pipeline per Anomaly Detection

#### Implementazione completa con PCA e Isolation Forest

```python
class TimeSeriesAnomalyDetector:
    """
    Detector di anomalie per serie temporali industriali
    """
    
    def __init__(self, n_components=0.95, contamination=0.1):
        self.n_components = n_components
        self.contamination = contamination
        self.pipeline = None
        self.feature_names = None
        
    def build_pipeline(self):
        """Costruisce la pipeline di preprocessing e detection"""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.n_components)),
            ('detector', IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            ))
        ])
        
    def extract_and_fit(self, time_series_data, window_size=100):
        """
        Estrae features e addestra il modello
        """
        # Estrazione features
        features = extract_features_from_timeseries(
            time_series_data.drop('timestamp', axis=1),
            window_size=window_size
        )
        
        # Rimozione di NaN e infiniti
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        
        self.feature_names = features.columns.tolist()
        
        # Costruzione e training della pipeline
        if self.pipeline is None:
            self.build_pipeline()
            
        self.pipeline.fit(features)
        
        return features
    
    def predict_anomalies(self, features):
        """Predice anomalie (-1: anomalia, 1: normale)"""
        return self.pipeline.predict(features)
    
    def get_anomaly_scores(self, features):
        """Ottiene i punteggi di anomalia"""
        return self.pipeline.decision_function(features)
    
    def plot_pca_components(self, features, anomalies=None):
        """Visualizza i dati nello spazio PCA"""
        # Transform dei dati
        scaler = self.pipeline.named_steps['scaler']
        pca = self.pipeline.named_steps['pca']
        
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        plt.figure(figsize=(12, 5))
        
        # Plot delle prime due componenti
        plt.subplot(1, 2, 1)
        if anomalies is not None:
            normal_mask = anomalies == 1
            anomaly_mask = anomalies == -1
            
            plt.scatter(features_pca[normal_mask, 0], features_pca[normal_mask, 1], 
                       c='blue', alpha=0.6, label='Normale')
            plt.scatter(features_pca[anomaly_mask, 0], features_pca[anomaly_mask, 1], 
                       c='red', alpha=0.8, label='Anomalia')
            plt.legend()
        else:
            plt.scatter(features_pca[:, 0], features_pca[:, 1], alpha=0.6)
        
        plt.xlabel('Prima Componente Principale')
        plt.ylabel('Seconda Componente Principale')
        plt.title('Dati nello Spazio PCA')
        
        # Plot della varianza spiegata
        plt.subplot(1, 2, 2)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        plt.xlabel('Numero di Componenti')
        plt.ylabel('Varianza Cumulativa Spiegata')
        plt.title('Varianza Spiegata dalle Componenti PCA')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Esempio di utilizzo completo
detector = TimeSeriesAnomalyDetector(n_components=0.95, contamination=0.05)

# Estrazione features e training
features = detector.extract_and_fit(data_100hz, window_size=200)
print(f"Features estratte: {features.shape}")
print(f"Componenti PCA mantenute: {detector.pipeline.named_steps['pca'].n_components_}")

# Predizione anomalie
anomalies = detector.predict_anomalies(features)
anomaly_scores = detector.get_anomaly_scores(features)

print(f"Anomalie rilevate: {np.sum(anomalies == -1)} su {len(anomalies)} campioni")

# Visualizzazione risultati
detector.plot_pca_components(features, anomalies)
```

### Gestione di Diverse Frequenze di Campionamento

```python
def analyze_multi_frequency_data(data_dict):
    """
    Analizza dati a diverse frequenze di campionamento
    
    Parameters:
    - data_dict: dizionario {frequenza: dataframe}
    """
    results = {}
    
    for freq, data in data_dict.items():
        print(f"\n=== Analisi dati a {freq} ===")
        
        # Calcolo window size appropriata basata sulla frequenza
        if '1hz' in freq.lower():
            window_size = 10  # 10 secondi di dati
        elif '10hz' in freq.lower():
            window_size = 50  # 5 secondi di dati
        elif '100hz' in freq.lower():
            window_size = 200  # 2 secondi di dati
        elif '200hz' in freq.lower():
            window_size = 400  # 2 secondi di dati
        else:
            window_size = 100  # default
        
        # Setup detector
        detector = TimeSeriesAnomalyDetector(contamination=0.05)
        
        try:
            # Estrazione features
            features = detector.extract_and_fit(data, window_size=window_size)
            
            # Predizione
            anomalies = detector.predict_anomalies(features)
            scores = detector.get_anomaly_scores(features)
            
            results[freq] = {
                'features_shape': features.shape,
                'n_anomalies': np.sum(anomalies == -1),
                'anomaly_percentage': np.mean(anomalies == -1) * 100,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
            
            print(f"Features: {features.shape}")
            print(f"Anomalie: {results[freq]['n_anomalies']} ({results[freq]['anomaly_percentage']:.2f}%)")
            print(f"Score medio: {results[freq]['mean_score']:.4f} ± {results[freq]['std_score']:.4f}")
            
        except Exception as e:
            print(f"Errore nell'analisi di {freq}: {e}")
            results[freq] = {'error': str(e)}
    
    return results

# Analisi multi-frequenza
data_dict = {
    '1Hz': data_1hz,
    '10Hz': data_10hz,
    '100Hz': data_100hz,
    '200Hz': data_200hz
}

multi_freq_results = analyze_multi_frequency_data(data_dict)
```

### Metriche di Valutazione per Anomaly Detection

```python
def evaluate_anomaly_detection(y_true, y_pred, scores=None, plot=True):
    """
    Valuta le performance di un sistema di anomaly detection
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
    
    # Conversione predizioni (-1,1) in (0,1) per le metriche
    y_pred_binary = (y_pred == -1).astype(int)
    y_true_binary = y_true.astype(int)
    
    # Metriche base
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    print("=== Metriche di Valutazione Anomaly Detection ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    if scores is not None:
        # ROC AUC (per anomaly detection, score più negativi = più anomali)
        roc_auc = roc_auc_score(y_true_binary, -scores)
        print(f"ROC AUC: {roc_auc:.4f}")
        
        if plot:
            plt.figure(figsize=(15, 5))
            
            # Confusion Matrix
            plt.subplot(1, 3, 1)
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # ROC Curve
            plt.subplot(1, 3, 2)
            fpr, tpr, _ = roc_curve(y_true_binary, -scores)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            
            # Precision-Recall Curve
            plt.subplot(1, 3, 3)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, -scores)
            plt.plot(recall_curve, precision_curve)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc if scores is not None else None
    }

# Esempio di creazione di ground truth sintetico per valutazione
def create_synthetic_anomalies(data, anomaly_rate=0.05):
    """Crea anomalie sintetiche per testing"""
    n_samples = len(data)
    n_anomalies = int(n_samples * anomaly_rate)
    
    # Labels (0: normale, 1: anomalia)
    y_true = np.zeros(n_samples)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    y_true[anomaly_indices] = 1
    
    return y_true

# Esempio di valutazione completa
y_true_synthetic = create_synthetic_anomalies(features, anomaly_rate=0.05)
metrics = evaluate_anomaly_detection(y_true_synthetic, anomalies, anomaly_scores)
```

### Feature Engineering Avanzato per Time Series

```python
def advanced_time_series_features(data, sensor_columns, window_sizes=[50, 100, 200]):
    """
    Estrae features avanzate da serie temporali multi-sensore
    """
    features_list = []
    
    for window_size in window_sizes:
        print(f"Elaborazione finestre di dimensione {window_size}...")
        
        # Features statistiche per ogni sensore
        for col in sensor_columns:
            series = data[col].values
            
            # Rolling statistics
            rolling_data = pd.Series(series).rolling(window=window_size, min_periods=1)
            
            features_dict = {
                f'{col}_mean_w{window_size}': rolling_data.mean().iloc[-1],
                f'{col}_std_w{window_size}': rolling_data.std().iloc[-1],
                f'{col}_min_w{window_size}': rolling_data.min().iloc[-1],
                f'{col}_max_w{window_size}': rolling_data.max().iloc[-1],
                f'{col}_range_w{window_size}': rolling_data.max().iloc[-1] - rolling_data.min().iloc[-1],
                f'{col}_skew_w{window_size}': rolling_data.skew().iloc[-1],
                f'{col}_kurt_w{window_size}': rolling_data.kurt().iloc[-1]
            }
            
            # FFT features (frequenza dominante)
            if len(series) >= window_size:
                fft_vals = np.fft.fft(series[-window_size:])
                fft_freqs = np.fft.fftfreq(window_size)
                dominant_freq_idx = np.argmax(np.abs(fft_vals[1:window_size//2])) + 1
                features_dict[f'{col}_dominant_freq_w{window_size}'] = abs(fft_freqs[dominant_freq_idx])
                features_dict[f'{col}_dominant_power_w{window_size}'] = abs(fft_vals[dominant_freq_idx])
            
            features_list.append(features_dict)
    
    # Cross-correlation features tra sensori
    for i, col1 in enumerate(sensor_columns):
        for j, col2 in enumerate(sensor_columns[i+1:], i+1):
            corr = np.corrcoef(data[col1], data[col2])[0, 1]
            features_list.append({f'corr_{col1}_{col2}': corr})
    
    # Combinazione di tutte le features
    combined_features = {}
    for feature_dict in features_list:
        combined_features.update(feature_dict)
    
    return pd.DataFrame([combined_features])

# Esempio di utilizzo
sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
advanced_features = advanced_time_series_features(data_100hz, sensor_cols)
print(f"Features avanzate estratte: {advanced_features.shape[1]} features")
print("\nPrime 10 features:")
print(advanced_features.columns[:10].tolist())
```

### Quando Usare Time Series Analysis e Anomaly Detection

**Utilizzare Time Series Analysis quando:**
- I dati hanno una componente temporale significativa
- Serve monitoraggio in tempo reale di sistemi
- Si vuole rilevare comportamenti anomali in processi industriali
- I dati provengono da sensori IoT o dispositivi di monitoraggio

**Feature Extraction con TSFEL quando:**
- Hai serie temporali complesse con molti sensori
- Serve automazione nell'estrazione di caratteristiche
- Vuoi features del dominio della frequenza e del tempo
- Lavori con dati di accelerometri, giroscopi, o sensori industriali

**PCA per Time Series quando:**
- Le features estratte sono ad alta dimensionalità
- Vuoi ridurre il rumore nei dati
- Serve visualizzazione dei pattern principali
- Hai correlazioni elevate tra le features

**Isolation Forest quando:**
- Cerchi anomalie senza labels (unsupervised)
- I dati hanno alta dimensionalità
- Le anomalie sono rare (< 10% dei dati)
- Vuoi un metodo robusto e veloce

---

## Matrice di Decisione Riassuntiva

| Tipo Problema | Dimensione Dati | Necessità Interpretabilità | Necessità Accuratezza | Approccio Raccomandato |
|---------------|-----------------|----------------------------|----------------------|------------------------|
| **Regressione** | Piccoli | Alta | Media | Linear/Ridge/Lasso |
| **Regressione** | Grandi | Bassa | Alta | XGBoost/LightGBM |
| **Classificazione** | Piccoli | Alta | Media | Logistic/Decision Tree |
| **Classificazione** | Grandi | Bassa | Alta | XGBoost/LightGBM |
| **Con Feature Categoriche** | Qualsiasi | Bassa | Alta | CatBoost |
| **Clustering** | Qualsiasi | Alta | N/A | Gerarchico |
| **Clustering** | Grandi | Bassa | N/A | K-Means |
| **Time Series Anomaly** | Qualsiasi | Media | Alta | Isolation Forest + PCA |
| **Dati con Outliers** | Qualsiasi | Alta | Media | RobustScaler + Ridge |
| **Feature Selection** | Alta Dimensionalità | Bassa | Alta | Lasso + Feature Engineering |
| **Esplorazione** | Qualsiasi | Alta | N/A | Visualizzazione + EDA |

### Linee Guida per Scelta Algoritmo

#### Basate sul Tipo di Dati

| Caratteristica Dati | Algoritmo Suggerito | Preprocessing Necessario |
|-------------------|-------------------|-------------------------|
| **Dati tabulari strutturati** | XGBoost/LightGBM/CatBoost | Minimal |
| **Dati con molti missing** | CatBoost/Random Forest | Imputazione avanzata |
| **Serie temporali** | LSTM/ARIMA/Prophet | Feature extraction TSFEL |
| **Testo** | TF-IDF + Logistic/BERT | Preprocessing testo |
| **Immagini** | CNN/Vision Transformer | Augmentation/Normalizzazione |
| **Dati misti** | Pipeline composita | ColumnTransformer |

#### Basate sul Business Context

| Contesto Business | Priorità | Algoritmo Raccomandato |
|------------------|----------|------------------------|
| **Medicina/Finanza** | Interpretabilità | Logistic Regression/Decision Trees |
| **Ricerca/Sperimentazione** | Accuratezza | Ensemble Methods |
| **Produzione Real-time** | Velocità | Linear Models/Fast Trees |
| **Prototipazione** | Semplicità | Scikit-learn Standard |
| **Competizioni** | Performance | Heavy Tuning + Ensemble |

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
7. **Gradient Boosting per Accuratezza**: Quando serve il massimo in termini di performance

### Errori Comuni da Evitare

**Preprocessing:**
- Saltare il preprocessing dei dati
- Fare fit dello scaler su tutto il dataset (data leakage)
- Non gestire i valori mancanti appropriatamente
- Ignorare gli outliers quando sono importanti
- Non creare indicatori per valori mancanti quando informativo

**Modellazione:**
- Non validare le performance del modello
- Overfitting sui dati di training
- Ignorare la scala delle feature per algoritmi basati su distanza
- Usare modelli complessi quando bastano quelli semplici
- Non considerare il contesto business e le necessità di interpretabilità

**Gradient Boosting Specifici:**
- Overtuning degli iperparametri
- Non usare early stopping
- Non monitorare overfitting durante il training
- Ignorare l'importanza delle feature per debugging

**Time Series:**
- Non rispettare l'ordine temporale nel train/test split
- Usare features future per predire il passato (data leakage temporale)
- Non considerare la stagionalità e i trend
- Window size inappropriata per l'estrazione features

**Regolarizzazione:**
- Non bilanciare il trade-off bias-varianza
- Usare regolarizzazione troppo forte (underfitting)
- Non cross-validare il parametro di regolarizzazione
- Applicare regolarizzazione a modelli già regolarizzati internamente

**Generale:**
- Data leakage nel preprocessing (fare fit su tutto il dataset)
- Non gestire il dataset sbilanciato nelle classificazioni
- Non documentare le scelte e i risultati degli esperimenti
- Non testare su dati realmente nuovi (out-of-time validation)