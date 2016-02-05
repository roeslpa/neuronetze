#ifndef KNN6_KM_H
#define KNN6_KM_H

#include "matrix.h"
#include <float.h>

using namespace std;

class KM{

public:
	KM(bool divisorNormalE);
	void kohonenLernregel(knn::matrix x);
	knn::matrix getRandomX(void);
	void ausgabe(char* dateiname);
private:
	knn::matrix calcBestMatchingNeuron(knn::matrix x);
	double nachbarschaftsfunktion(unsigned r1, unsigned s1, unsigned r2, unsigned s2);
	
	static const unsigned R = 40;
	static const unsigned S = 40;
	knn::matrix z[R][S];
	double lernrate = 0.1;
	//Für zweiten Fall Aufgabe 6.1.f
	bool divisorNormal;
};

KM::KM(bool divisorNormalE) {
	divisorNormal = divisorNormalE;

	//6.1.a) Initialisieren der Neuronen abhängig von r und s, damit es nicht zufällig ist
	for(unsigned r=0; r<R; r++) {
		for(unsigned s=0; s<S; s++) {
			z[r][s] = knn::matrix(1,3);
			z[r][s](1,1) = 4.0*((double)r/(double)R)-2.0;
			z[r][s](1,2) = 4.0*((double)s/(double)S)-2.0;
			z[r][s](1,3) = 0;
		}
	}
}

//6.1.b) Rückgabewert: 1: r, 2: s, 3: Distanz
knn::matrix KM::calcBestMatchingNeuron(knn::matrix x) {
	
	double norm;
	knn::matrix closestNeuron = knn::matrix(1,3);
	closestNeuron(1,3) = DBL_MAX;

	for(unsigned r=0; r<R; r++) {
		for (unsigned s=0; s<S; s++) {
			norm = pow( sqrt(pow(z[r][s](1,1)-x(1,1),2) + pow(z[r][s](1,2)-x(1,2),2) + pow(z[r][s](1,2)-x(1,2),2)) , 2);
			if(norm<closestNeuron(1,3)) {
				closestNeuron(1,1) = r;
				closestNeuron(1,2) = s;
				closestNeuron(1,3) = norm;
			}
		}
	}
	return closestNeuron;
}

//6.1.c)
double KM::nachbarschaftsfunktion(unsigned r1, unsigned s1, unsigned r2, unsigned s2) {
	if(divisorNormal = true) {
		return exp(-0.5*(pow(r1-r2, 2)+pow(s1-s2, 2)));
	} else { //Für zweiten Fall in Aufgabe 6.1.f
		return exp(-0.1*(pow(r1-r2, 2)+pow(s1-s2, 2)));
	}
	
}

//6.1.d)
void KM::kohonenLernregel(knn::matrix x) {
	knn::matrix bestMatchingNeuron;
	unsigned bmnR, bmnS;
	
	//Berechne best matching Neuron für gegebenes x und dessen indizes
	bestMatchingNeuron = calcBestMatchingNeuron(x);
	bmnR = (unsigned) bestMatchingNeuron(1,1);
	bmnS = (unsigned) bestMatchingNeuron(1,2);

	//Berechne deltas und neue Positionen
	for(unsigned r=0; r<R; r++) {
		for (unsigned s=0; s<S; s++) {
			z[r][s](1,1) += lernrate * nachbarschaftsfunktion(r, s, bmnR, bmnS) * (x(1,1) - z[r][s](1,1));
			z[r][s](1,2) += lernrate * nachbarschaftsfunktion(r, s, bmnR, bmnS) * (x(1,2) - z[r][s](1,2));
			z[r][s](1,3) += lernrate * nachbarschaftsfunktion(r, s, bmnR, bmnS) * (x(1,3) - z[r][s](1,3));
		}
	}
}

//6.1.e)
knn::matrix KM::getRandomX(void) {
	knn::matrix x;
	double test, eta, rho, tau;

	eta = knn::randomFromInterval(-0.1, 0.1);
	x = knn::matrix(1,3);
	x.fillRandom(-2, 2);
	
	test = sqrt(pow(x(1,1),2) + pow(x(1,2),2));
	//Berechne x3
	if(2.0/3.0 < test && test < 2) {
		tau = atan2(x(1,2),x(1,1));
		rho = acos( ( x(1,1)/cos(tau)-(4.0/3.0) ) * 3.0/2.0 );
		x(1,3) = 2.0/3.0 * sin(rho) + eta;
	} else {
		x(1,3) = eta;
	}
	
	return x;
}
//6.1.f
void KM::ausgabe(char* dateiname) {
	ofstream f;
	f.open(dateiname);
	for(unsigned r=0; r<R; r++){
		for(unsigned s=0;s<S;s++){
			f << z[r][s](1,1) << "\t" << z[r][s](1,2) << "\t" << z[r][s](1,3) << "\n";
		}
		f << "\n";
	} 
	f.close();
}


#endif // KNN6_KM_H
