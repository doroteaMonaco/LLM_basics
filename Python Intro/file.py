#ci sono varie modalità per aprire un file: r, lettura; w, scrittura; a, append;

f = open("todo.txt", "r")
print(f.read())
f.close()

f1 = open("todo.txt", "w")
f1.write("I would like to eat a pizza")
f1.close()

f2 = open("todo.txt", "a")
f2.write("I would like to eat a burger")
f2.close()

f3 = open("new.txt", "x") #crea un file nuovo