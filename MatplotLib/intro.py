import matplotlib

import matplotlib.pyplot as plt

xaxis =  [0, 100]
yaxis = [0, 200]

plt.plot(xaxis, yaxis, 'o')

plt.plot(yaxis, marker = 'o') #i marker sono il tipo di marchio per il grafico: punti, x, linee ecc..
plt.plot(xaxis, yaxis, linestyle = '--', color = 'r', linewidth = 2) #linea tratteggiata

#multiple lines
y1 = [0, 50]
y2 = [0, 100]

plt.plot(y1)
plt.plot(y2)

font1 = {'family':'serif','color':'blue','weight':'normal','size':12}
font2 = {'family':'serif','color':'darkred','weight':'normal','size':12}

plt.xlabel("X-axis", fontdict=font1)
plt.ylabel("Y-axis", fontdict=font1)
plt.title("My First Plot", fontdict=font2, loc='left')

plt.grid() #aggiunge la griglia, con axis = x o y solo una delle due
#dentro posso specificare color, linestyle, linewidth

plt.show()