# Banker's Algorithm in Python

# Number of processes
P = int(input("Enter the number of processes: "))

# Number of resources
R = int(input("Enter the number of resources: "))
print("\n")

# Available instances of resources
avail = []
for i in range(R):
    avail.append(int(input("Enter the available instance of resource {}: ".format(i+1))))

# Maximum resource matrix
maxm = []
for i in range(P):
    maxm.append([])
    print("\n")
    for j in range(R):
        maxm[i].append(int(input("Enter the maximum instance of resource {} for process {}: ".format(j+1, i+1))))

# Allocated resource matrix
alloc = []
for i in range(P):
    alloc.append([])
    print("\n")
    for j in range(R):
        alloc[i].append(int(input("Enter the allocated instance of resource {} for process {}: ".format(j+1, i+1))))
print("\n")

# Function to find the need of each process
def calculateNeed(need, maxm, alloc):
    for i in range(P):
        need.append([])
        
        for j in range(R):
            # Need of instance = maxm instance - allocated instance
            need[i].append(maxm[i][j] - alloc[i][j])

# Function to find if the system is in safe state or not
def isSafe(processes, avail, maxm, alloc):
    need = []
    # Calculate need matrix
    calculateNeed(need, maxm, alloc)

    # Mark all processes as unfinished
    finish = [0] * P

    # To store safe sequence
    safeSeq = [0] * P

    # Make a copy of available resources
    work = [0] * R
    for i in range(R):
        work[i] = avail[i]

    # While all processes are not finished or system is not in safe state.
    count = 0
    while count < P:
        found = False
        for p in range(P):
            if finish[p] == 0:
                for j in range(R):
                    if need[p][j] > work[j]:
                        break
                        
                if j == R - 1:
                    for k in range(R):
                        work[k] += alloc[p][k]

                    safeSeq[count] = p
                    count += 1

                    finish[p] = 1

                    found = True

        if not found:
            print("System is not in safe state")
            return False

    print("\n\nSystem is in safe state.\nSafe sequence is: ", end=" ")
    print(*safeSeq)
    return True

def main():
    isSafeStatus = isSafe(P, avail, maxm, alloc)
    
    if isSafeStatus:
        print("\nAllocated resources:")
        for i in range(P):
            for j in range(R):
                print(alloc[i][j], end=" ")
            print()

if __name__ == "__main__":
    main()