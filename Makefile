all:
	g++ -std=c++11 main.cpp gmdh.cpp -larmadillo -lmlpack -fopenmp

clean:
	rm *.o
