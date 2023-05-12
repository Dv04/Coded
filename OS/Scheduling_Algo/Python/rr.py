
def rr():
	# Python program for implementing Round Robin Algorithm
	def queueUpdation(queue, timer, arrival, n, maxProccessIndex):
		zeroIndex = -1
		for i in range(n):
			if(queue[i] == 0):
				zeroIndex = i
				break

		if(zeroIndex == -1):
			return
		queue[zeroIndex] = maxProccessIndex + 1


	def checkNewArrival(timer, arrival, n, maxProccessIndex, queue):
		if(timer <= arrival[n-1]):
			newArrival = False
			for j in range(maxProccessIndex+1, n):
				if(arrival[j] <= timer):
					if(maxProccessIndex < j):
						maxProccessIndex = j
						newArrival = True

			# adds the index of the arriving process(if any)
			if(newArrival):
				queueUpdation(queue, timer, arrival, n, maxProccessIndex)


	def queueMaintainence(queue, n):
		for i in range(n-1):
			if(queue[i+1] != 0):
				queue[i], queue[i+1] = queue[i+1], queue[i]


	timer, maxProccessIndex = 0, 0
	avgWait, avgTT = 0, 0
	tq = 5
	n = 4
	arrival = [0,1,2,3]
	burst = [0]*n
	wait = [0]*n
	turn = [0]*n
	queue = [0]*n
	temp_burst = [0]*n
	complete = [False]*n


	print("\nEnter the burst time of the processes :", end=" ")
	for i in range(n):
		burst[i] = int(input())
		temp_burst[i] = burst[i]

	for i in range(n):
			# Initializing the queue and complete array
		complete[i] = False
		queue[i] = 0

	while(timer < arrival[0]):
			# Incrementing Timer until the first process arrives
		timer += 1
	queue[0] = 1

	while(True):
		flag = True
		for i in range(n):
			if(temp_burst[i] != 0):
				flag = False
				break

		if(flag):
			break

		for i in range(n and queue[i] != 0):
			ctr = 0
			while((ctr < tq) and (temp_burst[queue[0]-1] > 0)):
				temp_burst[queue[0]-1] -= 1
				timer += 1
				ctr += 1

				# Updating the ready queue until all the processes arrive
				checkNewArrival(timer, arrival, n, maxProccessIndex, queue)

			if((temp_burst[queue[0]-1] == 0) and (complete[queue[0]-1] == False)):
				# turn currently stores exit times
				turn[queue[0]-1] = timer
				complete[queue[0]-1] = True

			# checks whether or not CPU is idle
			idle = True
			if(queue[n-1] == 0):
				for k in range(n):
					if(queue[k] != 0):
						if(complete[queue[k]-1] == False):
							idle = False
			else:
				idle = False

			if(idle):
				timer += 1
				checkNewArrival(timer, arrival, n, maxProccessIndex, queue)

			# Maintaining the entries of processes aftereach premption in the ready Queue
			queueMaintainence(queue, n)

	for i in range(n):
		turn[i] = turn[i] - arrival[i]
		wait[i] = turn[i] - burst[i]

	print("\nProgram No.\tArrival Time\tBurst Time\tWait Time\tTurnAround Time\n")

	for i in range(n):
		print(i+1, "\t\t", arrival[i], "\t\t", burst[i],
			"\t\t", wait[i], "\t\t", turn[i], "\n")

	for i in range(n):
		avgWait += wait[i]
		avgTT += turn[i]

	print("\nAverage wait time : ", (avgWait/n))
	print("\nAverage Turn Around Time : ", (avgTT/n))

rr()