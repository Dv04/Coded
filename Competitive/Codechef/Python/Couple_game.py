# cook your dish here
t = int(input())

for i in range(t):
    n = str(input())
    n = n.split()
    numbers = [int(i) for i in n]
    print(numbers[1]-numbers[0])
   