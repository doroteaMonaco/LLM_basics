a = 15
b = 10

if a > b:
    diff = a-b
    print(diff)
elif a == b:
    print("I numeri sono uguali")
else:
    diff = b-a
    print(diff)

if a > b or a == b:
    print("a è maggiore o uguale a b")
else:
    print("a è minore di b")

comando = "prodotto"
match comando:
    case "somma":
        somma = a+b
        print(somma)
    case "differenza":
        differenza = a-b
        print(differenza)
    case "prodotto":
        prodotto = a*b
        print(prodotto)
    case _:
        print("Comando non riconosciuto")

#per quanto riguarda i cicli while possono essere inseriti anche break continue

i = 0
while i < 5:
    print(i)
    i +=1
    if i == 4:
        break
    if i == 1:
        continue

for it in range(3, 10, 2): #significa 3, 5, 7, 9
    print(it)
#si può anche inserire un else alla fine del loop

for letter in "Hello":
    print(letter)
else:
    print("Hello")