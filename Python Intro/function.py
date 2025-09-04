numbers = [200, 452, 985, 56, 54, 258, 453, 1, 27]

def find_media(numbers):
    total = sum(numbers)
    count = len(numbers)
    media = total/count
    return media

media = find_media(numbers)
print(media)

def recursion(n):
    if n == 0:
        return 0
    else:
        return n + recursion(n-1)

print(recursion(5))

func = lambda x: x*5
print(func(5))