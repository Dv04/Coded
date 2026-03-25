CC=gcc

test: window-bit-count-apx.h test.c
	$(CC) -O3 test.c -o test -lm
	./test

bench: window-bit-count-apx.h bench.c
	$(CC) -O3 bench.c -o bench -lm
	./bench
