#ifndef KNN5_MIMLP_H
#define KNN5_MIMLP_H

#include "matrix.h"

using namespace std;

class MIMLP{

public:
	MIMLP(knn::matrix w1E, knn::matrix w2E, unsigned ME, unsigned DE);
	double getY(knn::matrix xE);
private:
	unsigned M; //Anzahl der verdeckten Neuronen
	unsigned D; //Anzahl der Eingabeneuronen
	knn::matrix w1; //Gewichtungen der Neuronen Ebene 1
	knn::matrix w2; //Gewichtungen der Neuronen Ebene 2
	knn::matrix z;
	double deltaExit; //
	knn::matrix deltaHid; // 
	double fact(double a); // Berechnung der fact-Funktion
	void calcError(knn::matrix xE, double tE); // Berechnung des Fehlers
	void gradientDescent(knn::matrix xE, double tE, double lernrate); // Berechnung des Gradientenabstiegs

};
//Aufgabe 5.2 a)
MIMLP::MIMLP(knn::matrix w1E, knn::matrix w2E, unsigned ME, unsigned DE) {
	w1 = w1E;  // Gewichte Ebene 1
	w2 = w1E;  // Gewichte Ebene 2
	M = ME; // Anzahl verdeckte Neuronen
	D = DE; // Anzahl Eingabeneuronen
	
	z = knn::matrix(1, M+1);
	z(1,1)=1; //z0=1
}

double MIMLP::fact(double a) {
	return 1.0/(1.0+exp(-1.0*a));
}

double MIMLP::getY(knn::matrix xE) {
	knn::matrix x = xE; //Eingabewerte
	double y = w2(1,1); //Da z0 = 1
	unsigned int sum =0; //Hiflsvariable
	for(unsigned m=2; m<=M+1; m++) {
		for(unsigned d=1; d<=D+1; d++)
		{
			sum += w1(d, m)* x(1,d);
		}
		z(1,m) = fact(sum);		
		y += w2(1,m)*z(1,m);
	}	
	return y;
}
	
void MIMLP::calcError(knn::matrix xE, double tE)
{
	deltaExit = getY(xE)-tE;
	deltaHid = knn::matrix(1,M+1); // deltaHid(1,1) wird nicht benötigt. Dies ist nur um den Index and die Indizierung der Gewichte an zupassen.
	for(unsigned m=2; m<=M+1; m++) {
	deltaHid(1,m)= z(1,m)*(1-z(1,m))*deltaExit*w2(1,m);
	}
}

void MIMLP::gradientDescent(knn::matrix xE, double tE, double lernrate)
{
	knn::matrix deltaW1 = knn::matrix(D+1,M+1);
	knn::matrix deltaW2 = knn::matrix(1, M+1);
	calcError(xE,tE);

	//Berechne deltaWm^(2)
	for (unsigned m=1;m<=M+1;m++)
	{
		deltaW2(1,m)= -(lernrate*deltaExit*z(1,m));
	}
	//Berechne deltaWdm^(1)
	for (unsigned d=1; d<=D+1;d++)
	{
		for (unsigned m=2;m<=M+1;m++)
		{
			deltaW1(d,m)= -(lernrate*deltaHid(1,m)*xE(1,d));
		}
	}
	//Neue Gewichte berechnen:
	for (unsigned m=1;m<=M+1;m++)
	{
		//w(1)00 und w(1)10 werden zwar uninitialisiert berechnet, aber später auch nicht genutzt => ignoriert
		for (unsigned d=1;d<=D+1;d++)
		{
			w1(d,m)+= deltaW1(d,m);
		}
		w2(1,m)+= deltaW2(1,m);
	}
}
#endif // KNN5_MIMLP_H
