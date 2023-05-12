''' 

ol./This is a calculator program that can calculate:
    1) Compound amount factor
    2) present worth factor
    3) Capital recovery factor
    4) Sinking fund factor
    5) Uniform payment factor

'''
F_cost, n, interest = input("Enter the First Cost, number of years and interest rate: ").split()
F_cost, n, interest = float(F_cost), int(n), float(interest)
change = 0
sum = 0

print("\nIs there any Annual? (Y/N)")
if input() == "Y":
    A_rev = float(input("Enter the Annual: "))
else:
    A_rev = 1

print("\nIs there any salvage value? (Y/N)")
if input() == "Y":
    S_value = float(input("Enter the salvage value: "))
else:
    S_value = 0
    
def single_payment(n_years):
    Compound_amount_factor = (1 + interest/100)**n_years
    Present_worth_factor = 1/Compound_amount_factor
    
    print("Compound amount factor: ", Compound_amount_factor)
    print("Present worth factor: ", Present_worth_factor)
    
def Uniform_payment(n_years,choice, check):
    Present_worth_factor = (((1 + interest/100)**n_years - 1)/(interest*(1 + interest/100)**n_years))*100
    Capital_recovery_factor = 1/Present_worth_factor
    Compound_amount_factor = ((1 + interest/100)**n_years-1)/interest
    Sinking_fund_factor = 1/Compound_amount_factor
    temp = 0
    if choice == 1:
        print("Present worth factor: %.4f and the corresponding Present value of Annual: %.4f" %(Present_worth_factor, Present_worth_factor*A_rev))
        temp = Present_worth_factor*A_rev
    elif choice == 2:
        print("Capital recovery factor: %.4f and the corresponding Present value of Annual: %.4f" %(Capital_recovery_factor, Capital_recovery_factor*A_rev))
        temp = Capital_recovery_factor*A_rev
        
    elif choice == 3:
        print("Compound amount factor: %.4f and the corresponding Present value of Annual: %.4f" %(Compound_amount_factor, Compound_amount_factor*A_rev))
        temp = Compound_amount_factor*A_rev
        
    elif choice == 4:
        print("Sinking fund factor: %.4f and the corresponding Present value of Annual: %.4f" %(Sinking_fund_factor, Sinking_fund_factor*A_rev))
        temp = Sinking_fund_factor*A_rev
        
    print("\n")
    if check:
        summation(temp)
    
def Salvage(value, choice):
    Compound_amount_factor = (1 + interest/100)**n
    Present_worth_factor = 1/Compound_amount_factor
    
    if choice == 1:
        print("Present worth factor: %.4f and The corresponding Present value of Salvage: %.4f" %(Present_worth_factor, Present_worth_factor*S_value))
        temp = Present_worth_factor*S_value
        
    elif choice == 3:
        print("Compound amount factor: %.4f and the The corresponding Present value of Salvage: %.4f" %( Compound_amount_factor, Compound_amount_factor*S_value))
        temp = Compound_amount_factor*S_value
        
    summation(temp)

def summation(value):
    global sum
    sum+=value

while True:
    
        
    if A_rev>0:
        print("\n\nWhat do you want to know?\n1) Present worth factor\n2) Capital recovery factor\n3) Compound amount factor\n4) Sinking fund factor")
        choice = int(input("Enter your choice: "))
        print("\nDo you want a sum of uniform payments? (Y/N)")
        if input() == "Y":
            print("\n")
            Uniform_payment(n,choice, True)
            
            print("\nSum of uniform payments: ", sum)

            if S_value>0:
                Salvage(S_value, choice)
            print("\nSum of uniform payments with salvage: ", sum)
            sum = 0
        else:
            for i in range(n):
                Uniform_payment(i+1,choice, False)
        
    else:
        for i in range(n):
            single_payment(i+1)
    
    
    
    
    print("Do you want to continue? (Y/N)")
    if input() == "Y":
    
        while change != 6:
            print("Do you want to change anything\n1) First Cost\n2) Number of years\n3) Interest rate\n4) Annual\n5) Salvage value\n6) Nothing")
            change = int(input("Enter your choice: "))
            if change == 1:
                F_cost = float(input("Enter the First Cost: "))
            elif change == 2:
                n = int(input("Enter the number of years: "))
            elif change == 3:
                interest = float(input("Enter the interest rate: "))
            elif change == 4:
                A_rev = float(input("Enter the Annual: "))
            elif change == 5:
                S_value = float(input("Enter the salvage value: "))
            elif change == 6:
                break
            else:
                print("Invalid choice")
                continue
            
        continue
    
    else:
        break