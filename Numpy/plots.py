import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random as rn

sns.displot([0, 1, 2, 3, 4])

#plt.show()
#senza istogramma kind = "kde"
sns.displot([0, 1, 2, 3, 4], kind="kde")
#plt.show()

#distribuzione normale
x = rn.normal(size=(2, 3), loc=1, scale=2) #media pari a 1 e deviazione standard pari a 2
print("Normale:", x)

sns.displot(x, kind="kde")
#distribuzioni binomiale

y = rn.binomial(n=10, p=0.5, size=(2, 3)) #n numero di prove, p probabilità di successo
print("Binomiale:", y)
sns.displot(y, kind="kde")

#distribuzione di poisson

z = rn.poisson(lam=3, size=(2, 3))
print("Poisson:", z)
sns.displot(z, kind="kde")

#distribuzione uniforme
a = rn.uniform(size=(2, 3), low=1.0, high=10.0)
print("Uniforme:", a)
sns.displot(a, kind="kde")
#distribuzione logistica

b = rn.logistic(size=(2, 3), loc=1.0, scale=2.0)
print("Logistica:", b)
sns.displot(b, kind="kde")

#distribuzione multinomiale
c = rn.multinomial(n=10, pvals=[0.2, 0.3, 0.5], size=(2, 3))
print("Multinomiale:", c)

#distribuzione esponenziale
d = rn.exponential(size=(2, 3), scale=2.0)
print("Esponenziale:", d)
sns.displot(d, kind="kde")

#distribuzione chi quadra
e = rn.chisquare(df=2, size=(2, 3))
print("Chi quadro:", e)
sns.displot(e, kind="kde")

#distribuzione rayleigh
f = rn.rayleigh(size=(2, 3), scale=2.0)
print("Rayleigh:", f)
sns.displot(f, kind="kde")

#distribuzione pareto
g = rn.pareto(a=2, size=(2, 3))
print("Pareto:", g)
sns.displot(g, kind="kde")

#distribuzione zipf
h = rn.zipf(a=2, size=(2, 3))
print("Zipf:", h)
sns.displot(h, kind="kde")

#differenza tra varie distribuzioni

data = {
    "normal": rn.normal(loc=0, scale=1, size=1000),
    "binomiale": rn.binomial(n=10, p=0.5, size=1000),
    "poisson": rn.poisson(lam=3, size=1000),
    "uniforme": rn.uniform(size=1000, low=1.0, high=10.0),
    "logistica": rn.logistic(size=1000, loc=1.0, scale=2.0),
    "esponenziale": rn.exponential(size=1000, scale=2.0),
    "chi_quadro": rn.chisquare(df=2, size=1000),
    "rayleigh": rn.rayleigh(size=1000, scale=2.0),
    "pareto": rn.pareto(a=2, size=1000),
    "zipf": rn.zipf(a=2, size=1000)
}

sns.displot(data, kind="kde")

plt.show()