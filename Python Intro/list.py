mylist = ["apple", "banana", "cherry"]
print(mylist)
print(mylist[0])
print(mylist[1])
print(mylist[2])
print(mylist[-1])
print(mylist[1:3])
print(len(mylist))
mylist.append("orange")
print(mylist)
mylist.remove("banana")
print(mylist)

list2 = ["Dory", "ha", 0, "cervello"]
print(list2)

#constructor

list3 = list(("ciao", "amore"))
print(list3)

if "hello" in list3:
    print("ciao è nella lista")
else:
    print("ciao non è nella lista")

list3[0] = "Hello"
print(list3)
list3.insert(1, "my")
print(list3)
list3.pop(0)
print(list3)
del mylist

for x in list3:
    print(x)

for val in range(len(list3)):
    print(list3[val])

i = 0
while i < len(list3):
    print(list3[i])
    i += 1

list3.sort(reverse=True)
print(list3)

list4 = list3.copy()
list5 = list3[:]

list6 = list4 + list5
print(list6)

ages = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]
sortedAges = ages.sort()
print(sortedAges)