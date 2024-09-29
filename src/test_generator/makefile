CXX = g++
CXXFLAGS = -g -O2 -std=c++11 -Wall
ARMAFLAGS = -larmadillo
ARMAFLAGSOLD = -DARMA_DONT_USE_WRAPPER -lblas -llapack

%.o: %.cpp
        $(CXX) $(CXXFLAGS) -c $^ -o $@

kuplet_test: kuplet_test.o
        $(CXX) $(CXXFLAGS) kuplet_test.o -o kuplet_test $(ARMAFLAGS)


.PHONY: clean

clean:
        rm -v *.o
