CC=	gcc
CFLAGS=	-g -Wall -Wextra -Wc++-compat -O2
LIBS=	-lm # math.h required by kann

.PHONY: all clean

all: example.o example

example.o: example.c kann/kann.h kann/kautodiff.h
	$(CC) $(CFLAGS) -c $< -o $@

example: example.o kann/kann.o kann/kautodiff.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o ./example
