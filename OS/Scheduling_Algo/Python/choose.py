
from fcfs import fcfs, sjf
from srtf import findavgTime
from rr import rr


print("Welcome To the Scheduler!")

n = int(input("Enter the number of processes: "))
print("\n")
arr = []
burst, arrive = 0, 0


for i in range(n):
    burst = int(input("Please enter the burst time for process %s : "%(i+1)))
    arrive = int(input("Please enter the arrival time for process %s : "%(i+1)))
    print("\n")
    arr.append([i+1,burst,arrive])

arr.sort(key = lambda x: x[2])


argument = int(input("Please enter the number of the algorithm you want to use:\n\t1. First Come First Serve\n\t2. Shortest Job First\n\t3. Shortest Remaining Time First\n\t4. Round Robin\n"))

def switch(argument):
    match argument:
        case 1: fcfs(n, arr),
        case 2: sjf(n, arr),
        case 3: findavgTime(arr, n),
        case 4: rr()
        case _: print("Invalid input")
        
switch(argument)


