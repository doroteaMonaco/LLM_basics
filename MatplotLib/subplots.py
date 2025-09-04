import matplotlib.pyplot as plt
import numpy as np

#plot1
x = np.array([0, 2, 4, 8, 7])
y = np.array([0, 1, 0, 1, 0])
plt.subplot(2, 1, 1) #2 righe, 1 colonna, primo plot
plt.plot(x, y)
plt.title("First Plot")

#plot2
x = np.array([0, 1, 5, 7, 9])
y = np.array([0, 1, 0, 1, 0])
plt.subplot(2, 1, 2)
plt.plot(x, y)
plt.title("Second Plot")


plt.suptitle("My Subplots")

plt.show()

#scatter
x = np.array([0, 1, 5, 7, 9])
y = np.array([0, 1, 0, 1, 0])
plt.scatter(x, y, color='r', cmap='viridis', s=100, alpha=0.5, marker='x') #s dimensione dei punti
plt.title("Scatter Plot")
plt.show()

#bars

x = np.array([0, 1, 5, 7, 9])
y = np.array([0, 1, 0, 1, 0])
plt.bar(x, y, color='b', alpha=0.7, width=0.5)
plt.title("Bar Plot")
plt.show()

plt.barh(x, y, color='g', alpha=0.7, height=0.5) #barre orizzontali
plt.title("Horizontal Bar Plot")
plt.show()

#istogramma

x = np.random.normal(170, 10, 250) #media, deviazione standard, numero di valori
plt.hist(x, bins=20, color='c', alpha=0.7)
plt.title("Histogram")
plt.show()

#pie charts

x = np.array([35, 25, 25, 15])
labels = ['A', 'B', 'C', 'D']
plt.pie(x, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Pie Chart")
plt.show()
