CC = mpicc
CFLAGS = -O3

all: hello_mpi

hello_mpi: hello_mpi.c
	$(CC) $(CFLAGS) -o hello_mpi hello_mpi.c

clean:
	rm -f hello_mpi
