#Compiler
CC= mpicc

#Flags
CFLAGS= -Wall -Wextra -g -O2
LDLIBS= -lm

#Sources
PROG= main.c
TARGET= main 

all: $(TARGET)
$(TARGET): $(PROG)
clean:
	rm -f $(TARGET) *.o *.out