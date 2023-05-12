#First come First Serve Scheduling Algorithm in Python

count = 0
def non_preemp_run(arr):
    global count
    if count >= arr[2]:
        count+=arr[1]
    else:
        count+= (arr[2]-count)+arr[1]
    return count

def fcfs(n, arr):
    avgT, avgA = 0, 0
    count = 0
    for i in arr:
        a = non_preemp_run(i)
        print("For process %d:\n\tFinish time: %d ms\n\tThe Turnaround time: %d ms\n\tWaiting time: %d ms\n" % (i[0],a,a-i[2],(a-i[2])-i[1]))
        avgT += a-i[2]
        avgA += (a-i[2])-i[1]
    print("The averages:\n\tTurnaround: %.2f ms\n\tWaiting: %.2f ms\n"%(avgT/n, avgA/n))

def sjf(n, arr):
    avgT, avgA = 0, 0
    count = 0
    for k in range(n):
        li = [j for j in arr if j[2] <= count]
        li.sort(key=lambda x: x[1])
        i = li[0]
        arr.remove(i)
        a = non_preemp_run(li[0])
        print("For process %d:\n\tFinish time: %d ms\n\tThe Turnaround time: %d ms\n\tWaiting time: %d ms\n" % (i[0],a,a-i[2],(a-i[2])-i[1]))
        avgT += a-i[2]
        avgA += (a-i[2])-i[1]
    print("The averages:\n\tTurnaround: %.2f ms\n\tWaiting: %.2f ms\n"%(avgT/n, avgA/n))
