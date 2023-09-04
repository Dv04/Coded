a = input("Enter a number: ")
sum=0
for i in a:
    sum+=int(i)**len(a)
    
print(f"{a} is an Armstrong number" if sum == int(a) else f"{a} is not an Armstrong number")