PROGRAM     = practica2GMD
PROGRAMARGS = 10 # nClientes

MPICC      = mpicc
MPICCFLAGS = -Wall -Wextra -g3 -std=c17

MPIRUN      = mpirun
MPIRUNFLAGS = -n 6

DEBUGGER = gdb
DEBUGGERFLAGS = --args

SRCS = $(wildcard ./*.c)


.PHONY: all run3 run6 clean
all : run6

$(PROGRAM) :
	$(MPICC) $(MPICCFLAGS) $(SRCS) -o $(PROGRAM)

run3 : clean $(PROGRAM)
	$(MPIRUN) -n 3 ./$(PROGRAM) $(PROGRAMARGS)

run6 : clean $(PROGRAM)
	$(MPIRUN) $(MPIRUNFLAGS) ./$(PROGRAM) $(PROGRAMARGS)

clean:
	rm -f $(PROGRAM)

debug : clean $(PROGRAM)
	$(DEBUGGER) $(DEBUGGERFLAGS) $(PROGRAM) $(PROGRAMARGS)
