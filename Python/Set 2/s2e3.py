dec = input("Are you going to input in celcius or farenhite: ")
if dec == "C" or dec == "c":
    C = int(input("Enter Temprature: "))
    F = (1.8*C)+32
    print(F)
elif dec == "F" or dec == "f":
    F = int(input("Enter Temprature: "))
    C = (1.8*F)+32
    print(C)
    