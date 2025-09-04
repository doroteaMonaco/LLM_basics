class firstClass:
    x = 5

p1 = firstClass()
print(p1.x)


#constructor + init method

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __str__(self):
        return f"{self.name}({self.age})"
    def modify(self, name2):
        self.name = name2

p2 = Person("Dory", 23)
print(p2.name)
print(p2.age)
print(p2)
p2.modify("Luigi")
print(p2)


#child class
class Student(Person): 
    def __init__(self, name, age, coloreyes):
        super().__init__(name, age)
        #Person.__init__(self, name, age), è equivalente
        self.coloreyes = coloreyes
    def __str__(self):
        return f"{self.name}({self.age},{self.coloreyes})"

s1 = Student("Dorotea", "23", "green")
print(s1)

#grazie al polimorfismo posso eseguire la stessa funzione per tutte le classi

class Car:
    def __init__(self, name, brand):
        self.name = name
        self.brand = brand

    def move(self):
        print("The car is moving")

class Boat:
    def __init__(self, name, brand):
        self.name = name
        self.brand = brand

    def move(self):
        print("The boat is moving")

class Animal:
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def move(self):
        print("The animal is moving")

car1 = Car("Hybrid", "Ford")
boat1 = Boat("Sailboat", "Yamaha")
animal1 = Animal("Lion", "Wild")

for obj in (car1, boat1, animal1):
    obj.move()
#global -> varibaili globale
#nonlocal -> appartiene alla funzione più esterna

def outer_function():
    x = "Dory"
    def inner_function():
        nonlocal x
        x = "Nemo"
    inner_function()
    return x

print(outer_function())

import mymodul

print(mymodul.sum(5, 10))
