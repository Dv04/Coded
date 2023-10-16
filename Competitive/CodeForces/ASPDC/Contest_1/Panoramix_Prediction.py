def isprime(x):
    for i in range(2,int(x**0.5)+1):
        if x%i == 0:
            return False
    return True

a,b = map(int,input().split())

if isprime(a) and isprime(b) and not any(isprime(x) for x in range(a+1,b)):
    print("YES")
else:
    print("NO")