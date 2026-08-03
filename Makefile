# Makefile for the MNIST CNN project (Windows + MinGW-w64 GCC)
#
# Usage:
#   mingw32-make            build main.exe
#   mingw32-make clean      remove build artifacts
#   gcc src/*.c -std=c11 -O2 -Wall -o main.exe   (one-liner, no make needed)

CC      = gcc
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Wpedantic
SRCS    = $(wildcard src/*.c)
OBJS    = $(SRCS:.c=.o)
TARGET  = main.exe

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	-del src\*.o $(TARGET)

.PHONY: clean
