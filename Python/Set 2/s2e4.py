import math

n = int(input("Enter a 5 digit number: "))
tot = [(n//(10**i)) % 10 for i in range(math.ceil(math.log(n, 10))-1, -1, -1)]

print(sum(tot))