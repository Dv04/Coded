CC=gcc

plots: main.c
	$(CC) -O3 main.c -o main -lm
	./main
	python3 plot_sep.py
