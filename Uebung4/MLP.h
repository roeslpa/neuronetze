#ifndef KNN4_MLP_H
#define KNN4_MLP_H

#include "matrix.h"

using namespace std;

class MLP{

public:

	MLP(knn::matrix wE, unsigned ME);
	double getY(double xE);
	void calcError(double xE, double tE); // Für Aufgabe 4.2 a)
	void gradientDescent(double x, double t, double lernrate); //Für Aufgabe 4.2 a)

private:

	double y;
	unsigned M;
	double deltaExit; //Für 4.2 a)
	knn::matrix deltaHid; // Für 4.2 a)
	
	knn::matrix w;
	knn::matrix z;
	double x;

	double fact(double a);
};

MLP::MLP(knn::matrix wE, unsigned ME) {
	w = wE;
	M = ME;
	//x = xE; in die get methode verschoben
	
	z = knn::matrix(1, M+1);
	z(1,1)=1; //z0=1
}

double MLP::fact(double a) {
	return 1.0/(1.0+exp(-1.0*a));
}

//Aufgabe e)
double MLP::getY(double xE) {
	x = xE; //muss das nicht hier hin?
	//z0 = 1
	y = w(3,1); 
	
	for(unsigned m=2; m<=M+1; m++) {
		z(1,m) = fact(w(1,m)+w(2,m)*x); //schauen welches w_01 und w_11 ist 
		y += w(3,m)*z(1,m);
	}
	
	return y;
}
//Aufgabe 4.2 a)
void MLP::calcError(double xE, double tE)
{
	deltaExit = getY(xE)-tE;
	deltaHid = knn::matrix(1,M+1); // deltaHid(1,1) wird nicht benötigt. Dies ist nur um den Index and die Indizierung der Gewichte an zupassen.
	for(unsigned m=2; m<=M+1; m++) {
	deltaHid(1,m)= z(1,m)*(1-z(1,m))*deltaExit*w(3,m);
	}
}
void MLP::gradientDescent(double x, double t, double lernrate)
{
	knn::matrix deltaW = knn::matrix(3,M+1);
	calcError(x,t);
	//Berechne deltaW0^(2)
	deltaW(3,1)= -(lernrate*deltaExit);
	//Berechne deltaWm^(2)
	for (unsigned m=2;m<=M+1;m++)
	{
		deltaW(3,m)= -(lernrate*deltaExit*z(1,m));
	}
	//Berechne deltaW0m^(1) und deltaW1m^(1)
	for (unsigned m=2;m<=M+1;m++)
	{
		deltaW(1,m)= -(lernrate*deltaHid(1,m));
		deltaW(2,m)= -(lernrate*deltaHid(1,m)*x);
	}
	//Neue Gewichte berechnen:
	for (unsigned m=1;m<=M+1;m++)
	{
		w(1,m)+= deltaW(1,m);
		w(2,m)+= deltaW(2,m);
		w(3,m)+= deltaW(3,m);
	}
}
#endif // KNN4_MLP_H
