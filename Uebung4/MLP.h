#ifndef KNN4_MLP_H
#define KNN4_MLP_H

#include "matrix.h"

using namespace std;

class MLP{

public:

	MLP(knn::matrix wE, unsigned ME);
	double get(double xE);

private:

	double y;
	unsigned M;

	knn::matrix w;
	knn::matrix z;

	double fact(double a);
};

MLP::MLP(knn::matrix wE, unsigned ME) {
	w = wE;
	x = xE;
	M = ME;

	z = knn::matrix(1, M);

	//z0 = 1
	y = w(2,1);
}

double MLP::fact(double a) {
	return 1.0/(1.0+exp(-1.0*a));
}

double MLP::get(double xE) {
	for(unsigned m=2; m<=M; m++) {
		z(1,m) = fact(w(1,m)*x);
		y += w(2,m)*z(1,m);
	}
	
	return y;
}

#endif // KNN4_MLP_H