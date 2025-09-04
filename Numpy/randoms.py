from numpy import random

randNumb = random.randint(100) #intero
randNumb1 = random.rand() #float
print(randNumb)
print(randNumb1)

#posso generare anche array
randArray = random.randint(100, size=(3, 4)) #array 3x4
print(randArray)
randArray1 = random.randint(100, size = 5)
print(randArray1)
randArray2 = random.rand(5) #array di float di 5 elementi
print(randArray2)

#choice permette di generare un valore random tra una serie di valori proposti
randomChoice = random.choice([3, 5, 7, 9])
print(randomChoice)
randomChoice1 = random.choice([3, 5, 7, 9], size=(3, 4))
print(randomChoice1)

#a questp posso aggiungere un vettore p che rappresenta le probabilità con cui un numero puo essere scelto
randomChoice2 = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.4, 0.2], size=(3, 4))
print(randomChoice2)

#shuffle
print("Array originale:")
print(randArray)
print("Cambia l'array originale modificano ordine:")
random.shuffle(randArray)
print(randArray)

#permutazioni di un array
print("Array originale:")
print(randArray)
print("Array con permutazioni:")
print(random.permutation(randArray))