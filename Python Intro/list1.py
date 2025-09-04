ages = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]
ages.sort()
print(ages)

minAge = min(ages)
maxAge = max(ages)
print("Età minima:", minAge)
print("Età massima:", maxAge)

ages.append(minAge)
ages.append(maxAge)
ages.sort()
print(ages)

median = (minAge + maxAge) / 2
print("Età mediana:", median)

sumAges = sum(ages)
media = sumAges / len(ages)
print(media)

min = abs(minAge - media)
max = abs(maxAge - media)

if min > max:
    print("L'età minima è più lontana dalla media.")
elif max > min:
    print("L'età massima è più lontana dalla media.")
else:
    print("Entrambe le età sono equidistanti dalla media.")


familyMember = tuple(("Massimo", "Grazia", "Dory", "Aury"))
parents = familyMember[:2]
siblings = familyMember[2:]
print("Genitori:", parents)
print("Fratelli:", siblings)
