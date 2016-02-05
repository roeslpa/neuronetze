#ifndef KNN6_KM_H
#define KNN6_KM_H

#include "matrix.h"
#include <float.h>

using namespace std;

class KM{

public:
	MIMLP(knn::matrix w1E, knn::matrix w2E, unsigned ME, unsigned DE);
	knn::matrix KN::calcBestMatchingNeuron(knn::matrix x);
	double KN::nachbarschaftsfunktion(unsigned r1, unsigned s1, unsigned r2, unsigned s2);
	knn::matrix[40][40] kohonenLernregel(knn::matrix x);
	knn::matrix getRandomX(void);
private:
	unsigned R, S;
	knn::matrix z[40][40];
	double lernrate = 0.1;
};

KM::KM(unsigned RE, unsigned SE) {
	R = RE;
	S = SE;

	//6.1.a) Initialisieren der Neuronen abhängig von r und s, damit es nicht zufällig ist
	for(unsigned r=0; r<R; r++) {
		for(unsigned s=0; s<S; s++) {
			z[r][s] = knn::matrix(1,3);
			z[r][s](1) = 4.0*((double)r/(double)R)-2.0;
			z[r][s](2) = 4.0*((double)s/(double)S)-2.0;
			z[r][s](3) = 0;
		}
	}
}

//6.1.b) Rückgabewert: 1: r, 2: s, 3: Distanz
knn::matrix KN::calcBestMatchingNeuron(knn::matrix x) {
	
	double norm;
	knn::matrix closestNeuron = knn::matrix(1,3);
	closestNeuron(3) = DBL_MAX;

	for(unsigned r=0; r<R; r++) {
		for (unsigned s=0; s<S; s++) {
			norm = pow( sqrt(pow(z[r][s](1)-x(1),2) + pow(z[r][s](2)-x(2),2) + pow(z[r][s](2)-x(2),2)) , 2);
			if(norm<closestNeuron(3)) {
				closestNeuron(1) = r;
				closestNeuron(2) = s;
				closestNeuron(3) = norm;
			}
		}
	}
	return closestNeuron;
}

//6.1.c)
double KN::nachbarschaftsfunktion(unsigned r1, unsigned s1, unsigned r2, unsigned s2) {
	return exp(-0.5*(pow(r1-r2, 2)+pow(s1-s2, 2)));
}

//6.1.d)
knn::matrix[40][40] kohonenLernregel(knn::matrix x) {
	knn::matrix delta[40][40];
	knn::matrix bestMatchingNeuron;
	unsigned bmnR, bmnS;
	
	//Berechne best matching Neuron für gegebenes x und dessen indizes
	bestMatchingNeuron = calcBestMatchingNeuron(x);
	bmnR = (unsigned) bestMatchingNeuron(1);
	bmnS = (unsigned) bestMatchingNeuron(2);
	
	//Berechne deltas
	for(unsigned r=0; r<R; r++) {
		for (unsigned s=0; s<S; s++) {
			delta[r][s] = knn::matrix(1,3);
			delta[r][s](1) = lernrate * nachbarschaftsfunktion(r, s, bmnR, bmnS) * (x(1) - z[r][s](1));
			delta[r][s](2) = lernrate * nachbarschaftsfunktion(r, s, bmnR, bmnS) * (x(2) - z[r][s](2));
			delta[r][s](3) = lernrate * nachbarschaftsfunktion(r, s, bmnR, bmnS) * (x(3) - z[r][s](3));
		}
	}

	return delta;
}

//6.1.e)
knn::matrix getRandomX(void) {
	knn::matrix x;
	double test, eta, rho, tau;

	eta = knn::randomFromInterval(-0.1, 0.1);
	x = knn::matrix(1,3);
	x.fillRandom(-2, 2);
	
	test = sqrt(pow(x(1),2) + pow(x(2),2));
	//Berechne x3
	if(2.0/3.0 < test && test < 2) {
		tau = atan2(x(2),x(1));
		rho = acos( ( x(1)/cos(tau)-(4.0/3.0) ) * 3.0/2.0 );
		x(3) = 2.0/3.0 * sin(rho) + eta;
	} else {
		x(3) = eta;
	}
	
	return x;
}

#endif // KNN6_KM_H