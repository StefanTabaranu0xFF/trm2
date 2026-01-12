CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

TARGET = trm

all: $(TARGET)

$(TARGET): trm.c
	$(CC) $(CFLAGS) -o $(TARGET) trm.c $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
