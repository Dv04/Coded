a, b = input("Enter two numbers: ").split()
a, b = int(a), int(b)


def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


c = gcd(a, b)
print("GCD of {} and {} is {}".format(a, b, c))

if c == 1:
    print("The numbers are co-prime")
else:
    print("The numbers are not co-prime")
