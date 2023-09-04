#Check if prime or not
a = int(input())
print("YES" if all(a % i != 0 for i in range(2, int(a**0.5)+1)) else "NO")