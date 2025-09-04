import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr) #ndarray
#dentro possiamo passare tuple, liste o array

arr1 = np.array((1, 2, 3, 4, 5))
print(arr1) #ndarray

#possiamo avere più dimensioni
zeroDim = np.array(42)
primaDim = np.array([1, 2, 3])
secondaDim = np.array([[1, 2, 3], [4, 5, 6]])
terzaDim = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(zeroDim.ndim) #0
print(primaDim.ndim) #1
print(secondaDim.ndim) #2
print(terzaDim.ndim) #3

arr2 = np.array([1, 2, 3, 4], ndmin=5)
print(arr2) #ndarray

x = arr2[0, 0, 0, 0, 0]
print(x)

#slicing
y = arr2[0, 0, 0, 0, 1:4]
print(y)

print(arr2.dtype) #int64
#posso creare degli array con specificato un dtype

arr3 = np.array([1, 2, 3, 4], dtype=float)
print(arr3)

#per convertire
newArr = arr3.astype('i')
print(newArr)

copyArray = newArr.copy()
newArr[0] = 6
print(copyArray)

viewArr = newArr.view()
print(viewArr)

print(arr2.shape) #la dimensione di ciascuna dimensione

#reshape per riorganizzare l'array da una dimensione a più dimensioni

arr4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
reshaped = arr4.reshape((3, 4)) #3d array da 4 elementi
print(reshaped)

reshaped1 = arr4.reshape((2, 2, 3)) #2 array che contengono 2 array da 3 elementi
print(reshaped1)

#convertire in una dimensione
reshaped2 = arr2.reshape(-1)
print(reshaped2)

#Iterazione
print("Iterazione prima dim:")
for x in primaDim:
    print(x)

print("Iterazione seconda dim:")
for y in secondaDim:
    print("Array:", y)
    for el in y:
        print(el)

print("Istruzione specifica per itarare su più dim")
for z in np.nditer(reshaped):
    print(z)

for id, i in np.ndenumerate(reshaped):
    print(id, i) #numero array e posizione con il rispettivo valore 
    #output -> (0, 0) 1

#concatenazione
uno = np.array([1, 2, 3])
due = np.array([4, 5, 6])
tre = np.array([7, 8, 9])
print("Concatenzaione di array")
concArr = np.concatenate((uno, due, tre))
print(concArr)
quattro = np.array([[1, 2], [3, 4]])
cinque = np.array([[5, 6], [7, 8]])
print("Concatenzaione di array 2D")
concArr1 = np.concatenate((quattro, cinque), axis=1)

#per concatenare array di 1D posso anche usare stack con axis = 1
concStack = np.stack((uno, due, tre), axis=1)
print("Concatenzaione con Stack", concStack)

concStack1 = np.hstack((uno, due, tre))
print("Concatenzaione con hstack", concStack1)
concStack2 = np.vstack((uno, due, tre))
print("Concatenzaione con vstack", concStack2)
concStack3 = np.dstack((uno, due, tre))
print("Concatenzaione con dstack", concStack3)

#si usa split array per splittare un array in un unico array con tre array dentro
spliArr = np.array_split(arr4, 3)
print("Split array", spliArr) #3 array
print("Primo", spliArr[0])
print("Secondo", spliArr[1])
print("Terzo", spliArr[2])

#search
arrSearch = np.array([1, 2, 3, 4, 5, 4, 6, 7, 8, 9])
val = np.where(arrSearch == 4) #dentro posso anche mettere una condizione
print("Valori trovati:", arrSearch[val])
valSorted = np.searchsorted(arrSearch, 5, side='right') #posizione dove inserire il valore per mantenere l'ordine
print("Posizione ordinata:", valSorted)
searchMultipleValues = np.searchsorted(arrSearch, [2, 4, 6], side='right')
print("Posizioni ordinate multiple:", searchMultipleValues)

#per ordinare un array esiste semplicemente il metodo np.sort()

#creare un array filter

newArrFilt = np.array([41, 42, 43, 44])

filArr = []

for el in newArrFilt:
    if el > 42:
        filArr.append(True)
    else:
        filArr.append(False)
newA = newArrFilt[filArr]
print("Array filtrato:", newA)
print("Filter:", filArr)