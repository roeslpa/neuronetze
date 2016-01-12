#ifndef KNN4_MLP_H
#define KNN4_MLP_H

#include "matrix.h"

using namespace std;

class MLP{

public:

	MLP(knn::matrix wE, unsigned ME);
	double get(double xE);
	void calcError(double xE, double tE); // Für Aufgabe 4.2 a)

private:

	double y;
	unsigned M;
	double deltaExit; //Für 4.2 a)
	knn::matrix deltaHid; // Für 4.2 a)
	
	knn::matrix w;
	knn::matrix z;

	double fact(double a);
};

MLP::MLP(knn::matrix wE, unsigned ME) {
	w = wE;
	M = ME;
	//x = xE; in die get methode verschoben
	z = knn::matrix(1, M);
	z(1,1)=1; //z0=1
}

double MLP::fact(double a) {
	return 1.0/(1.0+exp(-1.0*a));
}

//Aufgabe e)
double MLP::get(double xE) {
	x = xE; //muss das nicht hier hin?
	//z0 = 1
	y = w(3,1); //durfte ich das ändern?
	
	for(unsigned m=2; m<=M+1; m++) {
		z(1,m) = fact(w(1,m)+w(2,m)*x); //schauen welches w_01 und w_11 ist |||| Durfte ich das ändern?
		y += w(3,m)*z(1,m);
	}
	
	return y;
}
//Aufgabe 4.2 a)
void calcError(double xE, double tE)
{
	deltaExit = get(xE)-tE;
	deltaHid = knn::matrix(1,M);
	for(unsigned m=1; m<=M; m++) {
	deltaHid(1,m)= z(1,m)*(1-z(1,m))*deltaExit*w(3,m);
	}
}
#endif // KNN4_MLP_H
