# code to find class, host id and network id of an ip address in macOS

def A(id):
    print("Class A id found")
    print("Network id is: ", id[0:3])
    print("Host id is: ", id[4:])

def B(id):
    print("Class B id found")
    print("Network id is: ", id[0:7])
    print("Host id is: ", id[8:])
    
def C(id):
    print("Class C id found")
    print("Network id is: ", id)
    print("Invalid host ID")
    

def check(id):
    check = int(id[0:3])
    if check <= 127:
        A(id)
    elif check <= 191:
        B(id)
    elif check <= 223:
        C(id)
    elif check <= 239:
        print("Class D id found. It is reserved for Multicasting")
        print("Invalid host or network ID")
    elif check <= 255:
        print("Class E id found. It is reserved for Research")
        print("Invalid host or network ID")


for _ in range(4):
    id = input("\nPlease Enter your id: ")
    check(id)