PROGRAM     = practica2
#PROGRAMARGS = 9

CC      = g++
CCFLAGS = -O2 -Wall -Wextra -g3

MPICC      = mpic++
MPICCFLAGS = -O2 -Wall -Wextra -g3

OUTPUT = > salida

MPIRUN      = mpirun
MPIRUNFLAGS = -n 6 --verbose

DEBUGGER = gdb
DEBUGGERFLAGS = --args

SRCS = $(wildcard ./*.cpp)


.PHONY: all run3 run6 clean
all : clean $(PROGRAM) run6

sinMejoras :
	$(CC) $(CCFLAGS) $(SRCS) -o $(PROGRAM)

$(PROGRAM) :
	$(MPICC) $(MPICCFLAGS) $(SRCS) -o $(PROGRAM)

run3 : clean $(PROGRAM)
	$(MPIRUN) -n 3 ./$(PROGRAM) #$(PROGRAMARGS) #$(OUTPUT)

run6 : clean $(PROGRAM)
	$(MPIRUN) $(MPIRUNFLAGS) ./$(PROGRAM) #$(PROGRAMARGS)

clean:
	rm -f $(PROGRAM)

debug : clean $(PROGRAM)
	$(DEBUGGER) $(DEBUGGERFLAGS) $(PROGRAM) #$(PROGRAMARGS)
