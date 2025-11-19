#Compiler
CC= gcc

#Flags
CFLAGS= -Wall -Wextra -g -O2 -fopenmp
LDLIBS= -lm

#Sources
PROG= main.c
TARGET= main 

all: $(TARGET)
$(TARGET): $(PROG)
clean:
	rm -f $(TARGET) *.o *.out