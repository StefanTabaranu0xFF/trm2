CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11 -ffast-math
SRC = src/main.c src/gemm.c

all: trm

trm: $(SRC)
	$(CC) $(CFLAGS) -o trm $(SRC) -lm

clean:
	rm -f trm
