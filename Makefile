
CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -O3 -Ofast

all: expt

expt:
	c++ $(CXXFLAGS) expt.cpp -o expt
	time ./expt

clean:
	rm -rf expt

.PHONY: all expt
