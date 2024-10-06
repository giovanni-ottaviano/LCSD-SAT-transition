CXX = g++
CXXFLAGS = -g -Wall -O2 -std=c++11

# Gurobi
GUROBIINCLUDE = -I$(GUROBI_HOME)/include
GUROBILDPATH  = -L$(GUROBI_HOME)/lib
GUROBILDFLAGS = -lgurobi_c++ -lgurobi91 -lm -lpthread

# Armadillo
ARMAFLAGS = -larmadillo

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(GUROBIINCLUDE) -c $^ -o $@

qcp_solver.cpp: qcp_solver.cpp.o
	$(CXX) $(CXXFLAGS) $(GUROBIINCLUDE) qcp_solver.cpp.o -o qcp_solver.cpp $(GUROBILDPATH) $(GUROBILDFLAGS) $(ARMAFLAGS)


.PHONY: clean

clean:
	rm -v *.o
