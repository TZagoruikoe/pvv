#Compiler
CC= gcc

#Flags
CFLAGS= -Wall -Wextra -g -fopenmp
LDFLAGS= -lm

#Sources
PROG= main.c
TARGET= main 

all: $(TARGET)
$(TARGET): $(PROG)
	$(CC) $(CFLAGS) $(PROG) -o $(TARGET) $(LDFLAGS)
clean:
	rm -f $(TARGET) *.o *.out