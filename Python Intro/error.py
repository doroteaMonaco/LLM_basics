x = 5
try:
    print(x)
except:
    print("An exception occurred")
else:
    print("No exceptions occurred")

y = "Ciao"
try:
    print(y)
except:
    print("An exception occurred")
finally:
    print("Finally block executed")