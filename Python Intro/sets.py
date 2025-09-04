firstSet = {"Uno", "Due", "Tre"}
print(firstSet)

for s in firstSet:
    print(s)

if "Uno" not in firstSet:
    print("Uno non c'è")
else:
    print("Uno c'è")

firstSet.add("Quattro")
print(firstSet)

secondSet = {"Cinque", "Sei", "Sette"}
print(secondSet)

firstSet.update(secondSet) #si può fare anche con liste, non deve essere un set
print(firstSet)

secondSet.remove("Cinque")

set1 = {"Uno", "Due", "Tre"}
set2 = {1, 2, 3}

set1.union(set2) #unione
print("U", set1)
set1.intersection(set2) #intersezione
print("I", set1)
set1.symmetric_difference(set2) #elementi che non sono in comune
print("S", set1)
set1.difference(set2) #elementi che sono in set1 ma non in set2
print("D", set1)
print(set1)
