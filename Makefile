GCC=g++
CFLAGS=-Wall -g -O3
LOADLIBS=-lm

clean:
	rm -rf *.o *.out

test: MCSVM.o SVM.o main.cpp
	$(GCC) -o test MCSVM.o SVM.o main.cpp

MCSVM.o: MCSVM.hpp MCSVM.cpp
	$(GCC) -c MCSVM.cpp 

SVM.o: SVM.hpp SVM.cpp
	$(GCC) -c SVM.cpp

