def A(id): print("Class A id found","\nNetwork id is: ", id[0:3],"\nHost id is: ", id[4:])
def B(id): print("Class B id found","\nNetwork id is: ", id[0:7],"\nHost id is: ", id[8:])
def C(id): print("Class C id found","\nNetwork id is: ", id,"\nInvalid host ID")
def check(id):
    check = int(id[0:3])
    if check <= 127: A(id)
    elif check <= 191: B(id)
    elif check <= 223: C(id)
    elif check <= 239:
        print("Class D id found. It is reserved for Multicasting","\nInvalid host or network ID")
    elif check <= 255:
        print("Class E id found. It is reserved for Research","\nInvalid host or network ID")
for _ in range(5): check(input("\nEnter your id: "))