#include "matrix.h"
#include "MLP.h"

int main(int argc, char** argv)
{
	double xMin = 0;
	double xMax = 10;
	unsigned M = 20;
	double wMin = -4;
	double wMax = 4;

	knn::matrix w;
	double x;
	ofstream output41e;

	w = knn::matrix(2, M);
	w.fillRandom(wMin, wMax);
	output41e.open("out4.1.e.txt");

	for(unsigned i=0; i<1000; i++) {
		x = knn::randomFromInterval(xMin, xMax);
		MLP mlp = MLP(w, x, M);
		output41e << x << " " << mlp.get() << "\n";
	}
	output41e.close();
}