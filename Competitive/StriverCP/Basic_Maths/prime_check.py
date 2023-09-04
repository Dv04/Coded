a = int(input("Please enter a number: "))
print(f"{a} is a prime number" if all(a%i!=0 for i in range(2,int(a**0.5)+1)) else f"{a} is not a prime number")