x = 5
name = "Dorotea"
print("Number is", x)
print("Name is", name)

numberStr = str(x)
print("Number as string is", numberStr)

print(type(x))
print(type(name))
print(type(numberStr))

#many values to multiple variables

w, y, z = "Stupida", "Cogliona", "Ritardata"
print(w)
print(y)
print(z)

#assegnazione

w = y = z = "Ciao"
print(w)
print(y)
print(z)

#unpacking

complimenti = ["Stupida", "Cogliona", "Ritardata"]
w, y, z = complimenti
print(w)
print(y)
print(z)

#concatenazione

a = "Ciao"
b = "Mondo"
c = a + " " + b
print(c)

global dory
dory = "Dory"

def myfunc():
    dory = "Nemo"
    print(dory)

myfunc()
print(dory)

#data types
x = "Hello World"		
x = 20		
x = 20.5		
x = 1j		
x = ["apple", "banana", "cherry"]	#list	
x = ("apple", "banana", "cherry")	#tuple	
x = range(6)		
x = {"name" : "John", "age" : 36}	#dict	
x = {"apple", "banana", "cherry"}	#set	
x = True		
x = b"Hello"	#bytes	
x = None
