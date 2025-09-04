import numpy as np

#Questo viene chimato virtualization

x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
z = []

for i, j in zip(x, y):
    z.append(i + j)

print("Prima soluzione:", z)

z = np.add(x, y) #somma elemento per elemento
print("Seconda soluzione:", z)

#per poter creare una ufunc bisogna prima definirla e poi aggiungerla alla libreria

def myfunc(x, y):
    return x + y

myfunc = np.frompyfunc(myfunc, 2, 1) #2 input e 1 output
print("Terza soluzione:", myfunc(x, y))

#per controllare che sia un ufunc
print("Tipo di myfunc:", type(myfunc))

z = np.subtract(x, y) #sottrazione elemento per elemento
print("Sottrazione:", z)

z = np.multiply(x, y) #moltiplicazione elemento per elemento
print("Moltiplicazione:", z)

z = np.divide(x, y) #divisione elemento per elemento
print("Divisione:", z)

z = np.power(x, y)
print("Potenza:", z)

z = np.mod(x, y) #resto della divisione elemento per elemento
print("Resto della divisione:", z)

z = np.remainder(x, y) #resto della divisione elemento per elemento
print("Resto della divisione:", z)

z = np.divmod(x, y) #resto e quoziente della divisione elemento per elemento
print("Quoziente e resto della divisione:", z)

arrAbs = [-1, 5, 0, -8, 4]
z = np.abs(arrAbs)
print("Valori assoluti:", z)

#ROUNDING


outArr = np.trunc([-3.166, 3.6985])
print("Troncamento:", outArr)

outArr = np.fix([-3.166, 3.6985])
print("Arrotondamento per difetto:", outArr)

val = np.around(3.1666, 3) #arrotonda a 3 cifre decimali
print("Arrotondamento:", val)

out = np.floor([-3.166, 3.6985]) #arrotonda per difetto al numero intero più vicino
print("Arrotondamento per difetto:", out)

out = np.ceil([-3.166, 3.6985]) #arrotonda per eccesso al numero intero più vicino
print("Arrotondamento per eccesso:", out)

#LOGS
arr = np.arange(1, 10)
print("Log2: ", np.log2(arr))
print("Log10: ", np.log10(arr))
print("Log:", np.log(arr)) #log naturale

from math import log

mylog = np.frompyfunc(log, 2, 1) #2 input e 1 output
print(mylog(100, 15))

#se uso np.add ottengo un array con la somma degli elementi dei due array
#se uso sum sommo tutti gli elementi dei due array ottenendo un unico valore

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
newval = np.sum([arr1, arr2])
print("Somma totale:", newval)
newval = np.sum([arr1, arr2], axis=1)
print("Somma totale dei due:", newval)

print("Somma cumulativa:", np.cumsum(arr1)) #somma cumulativa degli elementi dell'array

print("Prodotto:", np.prod(arr1))
print("Prodotto tra due vettori:", np.prod([arr1, arr2])) #posso aggiungere axis = 1
print("Prodotto cumulativo:", np.cumprod(arr1))

print("Differenza:", np.diff(arr1))
print("Differenza:", np.diff(arr2, n=2))

print("Minimo comune multiplo:", np.lcm(4, 6))
print("Minimo comune multiplo:", np.lcm.reduce(arr1))

print("Massimo comune divisore:", np.gcd(4, 6))
print("Massimo comune divisore:", np.gcd.reduce(arr1))

#funzioni trigonometriche
x = np.sin(np.pi/2)
print("Seno di π/2:", x)

arrSin = [np.pi/2, np.pi/3, np.pi/4]
z = np.sin(arrSin)
print("Seno di π/2, π/3, π/4:", z)

print("Convertire gradi in radianti:", np.deg2rad([0, 30, 45, 60, 90]))
print("Convertire radianti in gradi:", np.rad2deg([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]))

#arcsin, arccos, arctan

print("Ipotenusa:", np.hypot(3, 4))

#sinh, cosh, tanh, arctanh, arccosh, arcsinh

#set di operazioni
arrSet = np.array([-1, 0, 1, 1, 5, 2])
x = np.unique(arrSet)
print("Elementi unici:", x)
print("Union1d:", np.union1d(arrSet, arrSet))
print("Intersect1d:", np.intersect1d(arrSet, arrSet))
print("Setdiff1d:", np.setdiff1d(arrSet, [1, 2, 3]))
print("Setxor1d:", np.setxor1d(arrSet, [1, 2, 3]))