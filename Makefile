CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
SRC = src/main.c src/model.c src/attention.c src/gemm.c

all: trm

trm: $(SRC)
	$(CC) $(CFLAGS) -o trm $(SRC) -lm

clean:
	rm -f trm
