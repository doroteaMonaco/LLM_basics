tuplaIt = ("element1", "element2", "element3")
iterator = iter(tuplaIt)

print(next(iterator))
print(next(iterator))
print(next(iterator))

mystr = "Hello"
iterator2 = iter(mystr)

print(next(iterator2))
print(next(iterator2))
print(next(iterator2))
print(next(iterator2))
print(next(iterator2))

#creare un iteratore da capo
class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self
    def __next__(self):
        x = self.a
        self.a += 1
        return x
    
myclass = MyNumbers()
iter5 = iter(myclass)

for i in range(10):
    print(next(iter5))