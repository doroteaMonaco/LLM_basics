import pandas as pd
import matplotlib.pyplot as ptl

a = [1, 2, 7]

b = pd.Series(a)
print(b)
print(b[0]) #primo elemento
print(b[1]) #secondo elemento
print(b[2]) #terzo elemento

c = pd.Series(a, index=["x", "y", "z"]) #posso specificare gli indici
print(c)
print(c["x"]) #primo elemento
print(c["y"]) #secondo elemento
print(c["z"]) #terzo elemento

calories = {
    "day1": 200,
    "day2": 250,
    "day3": 300
}

result = pd.Series(calories, index=["day1", "day2", "day3"])
print(result)

data = {
    "calories": [200, 250, 300],
    "duration": [30, 45, 60]
}

df = pd.DataFrame(data)
print(df)
print(df.loc[0]) #prima riga
print(df.loc[1]) #seconda riga
print(df.loc[2]) #terza riga

#Read CSV

pd.options.display.max_rows = 100
df = pd.read_csv('test.csv')
print(df) #per stampare tutto il dataframe
print(df.head(3)) #per stampare le prime 3 righe
print(df.tail(3)) #per stampare le ultime 3 righe
print(df.info()) #per avere informazioni sul dataframe

newf = df.dropna() #elimina le righe con valori nulli
print(newf)

df.fillna(130, inplace=True) #sostituisce i valori nulli con 130
#df['Date'] = pd.to_datetime(df['Date'], format='mixed')
#print(df) per stampare tutto il dataframe

print(df.duplicated())
df.drop_duplicates(inplace=True)
print(df)
print("Correlation matrix:")
#print(df.corr())

df.plot()
df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')
df["Duration"].plot(kind = 'hist')
ptl.show()