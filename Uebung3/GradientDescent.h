#ifndef KNN3_GRADIENT_H
#define KNN3_GRADIENT_H

#include "matrix.h"
#include <math.h>
#include <float.h>

using namespace std;

class GradientDescent{

public:
	GradientDescent(double etaA, double alphaA, double betaA, double rangeA); //constructor
	knn::matrix gradient(knn::matrix wA); //b) gradient function
	double error(knn::matrix wA); //b) error function
	knn::matrix gradientDescent(knn::matrix wInitA); //gardient descent algorithm ex. c)
	void executeD(unsigned int noInitVecsA); //d)
	void executeE1(void); //e) 1
	void executeE2(void); //e) 2
	void berechneHessematrix(knn::matrix w); //Berechnung für die Hessematrix
	double calculateChange(knn::matrix wInitOldA, knn::matrix wInitCurrentA); //help-function calculate the change for gradientDescent

private:
	double rangeE; //interval +/- range
	double etaE; //lerningrate
	double alphaE; //alpha
	double betaE; //beta
	knn::matrix minErrorWE; //minimal error w vector pos
	double minErrorE; // minimal error on run
};


GradientDescent::GradientDescent(double etaA, double alphaA, double betaA, double rangeA)
{
	etaE = etaA;
	alphaE = alphaA;
	betaE = betaA;
	rangeE = rangeA;
	minErrorE = DBL_MAX;
	minErrorWE = knn::matrix(3, 1);
	knn::init();
}

//b) error function
double GradientDescent::error(knn::matrix wA)
{
	return 1.0 - exp(-((pow(wA(1,1),2)*pow(wA(2,1),2)*pow(wA(3,1),2))/betaE)) - alphaE*(cos(wA(1,1))*cos(wA(2,1))*cos(wA(3,1))-1.0);
}

//b) gradient function
knn::matrix GradientDescent::gradient(knn::matrix wA)
{
	knn::matrix gradient = knn::matrix(3, 1);
	gradient(1,1) = 2.0*wA(1,1)/betaE * exp(-((pow(wA(1,1),2)*pow(wA(2,1),2)*pow(wA(3,1),2))/betaE)) + alphaE*sin(wA(1,1))*cos(wA(2,1))*cos(wA(3,1));
	gradient(2,1) = 2.0*wA(2,1)/betaE * exp(-((pow(wA(1,1),2)*pow(wA(2,1),2)*pow(wA(3,1),2))/betaE)) + alphaE*cos(wA(1,1))*sin(wA(2,1))*cos(wA(3,1));
	gradient(3,1) = 2.0*wA(3,1)/betaE * exp(-((pow(wA(1,1),2)*pow(wA(2,1),2)*pow(wA(3,1),2))/betaE)) + alphaE*cos(wA(1,1))*cos(wA(2,1))*sin(wA(3,1));
	return gradient;
}

//help-function calculate the change for gradientDescent
double GradientDescent::calculateChange(knn::matrix wInitOldA, knn::matrix wInitCurrentA)
{
	//Berechnung des Abstands der beiden Vektoren
	double distance =  pow(pow(wInitOldA(1,1)-wInitCurrentA(1,1),2)+pow(wInitOldA(2,1)-wInitCurrentA(2,1),2)+pow(wInitOldA(2,1)-wInitCurrentA(2,1),2), 0.5);
	return distance;
}

//c) 
knn::matrix GradientDescent::gradientDescent(knn::matrix wInitA)
{
	double errorStartL = error(wInitA);
	knn::matrix wInitCurrentL = wInitA;
	unsigned int counterL = 0;
	knn::matrix wRecentL = wInitA;

	wInitCurrentL(1,1) += (-etaE*gradient(wInitCurrentL)(1,1)); 
	wInitCurrentL(2,1) += (-etaE*gradient(wInitCurrentL)(2,1));
	wInitCurrentL(3,1) += (-etaE*gradient(wInitCurrentL)(3,1));

	while(error(wRecentL)>error(wInitCurrentL) && counterL<10000) //Bei Divergenz wird diese Differenz negativ. Bei Konvergenz wird sie immer kleiner.
	{
		wRecentL = wInitCurrentL;
		//w_neu=w_alt+DeltaW
		wInitCurrentL(1,1) += (-etaE*gradient(wInitCurrentL)(1,1)); 
		wInitCurrentL(2,1) += (-etaE*gradient(wInitCurrentL)(2,1));
		wInitCurrentL(3,1) += (-etaE*gradient(wInitCurrentL)(3,1));
		++counterL;
	}
	
	wInitCurrentL = wRecentL;
	double errorEndL = error(wInitCurrentL);

	if (!std::isnan(errorEndL) && (errorEndL<minErrorE))
	{
		minErrorE = errorEndL;
		minErrorWE = wInitCurrentL;
	}

	std::cout << "descent (eta: " << etaE << ") starting at: (" << wInitA(1, 1) << "," << wInitA(2, 1) << "," << wInitA(3, 1) << ")" << std::endl;
	std::cout << "with error: " << errorStartL << std::endl;
	std::cout << "stopped at: (" << wInitCurrentL(1, 1) << "," << wInitCurrentL(2, 1) << "," << wInitCurrentL(3, 1) << ")" << std::endl;
	std::cout << "after: " << counterL << " Iterations." << std::endl;
	std::cout << "with error: " << errorEndL << std::endl << std::endl;

	return wInitCurrentL;
}

//d)
void GradientDescent::executeD(unsigned int noInitVecsA)
{
	knn::matrix randomW = knn::matrix(3,1);
	knn::matrix minErrorW;
	double minError;
	std::cout << "run started for eta: " << etaE << std::endl;
	for (unsigned int kL = 0; kL<noInitVecsA; ++kL)
	{
		std::cout << "start iteration: " << kL+1 << std::endl << std::endl;
		randomW.fillRandom(-rangeE,rangeE);
		minErrorW = gradientDescent(randomW);
		minError = error(minErrorW);
	}
	std::cout << "run finished for eta: " << etaE << std::endl << std::endl;
	std::cout << "min error: " << minErrorE << " at: (" << minErrorWE(1, 1) << "," << minErrorWE(2, 1) << "," << minErrorWE(3, 1) << ") " << std::endl << std::endl;
}

void GradientDescent::berechneHessematrix(knn::matrix w)
{
	//Berechne H_11
	std::cout << "H_11: " << (((2.0/betaE)-(4.0*pow(w(1,1),2)/pow(betaE,2))) * exp(-((pow(w(1,1),2)*pow(w(2,1),2)*pow(w(3,1),2))/betaE)) + alphaE*cos(w(1,1))*cos(w(2,1))*cos(w(3,1))) << std::endl << std::endl;
	//Berechne H_12
	std::cout << "H_12: " << (((-4*w(1,1)*w(2,1)/pow(betaE,2))*exp(-((pow(w(1,1),2)*pow(w(2,1),2)*pow(w(3,1),2))/betaE)))-(alphaE*sin(w(1,1))*sin(w(2,1))*cos(w(3,1)))) << std::endl << std::endl;
	//Berechne H_13
	std::cout << "H_13: " << (((-4*w(1,1)*w(3,1)/pow(betaE,2))*exp(-((pow(w(1,1),2)*pow(w(2,1),2)*pow(w(3,1),2))/betaE)))-(alphaE*sin(w(1,1))*cos(w(2,1))*sin(w(3,1)))) << std::endl << std::endl;
	//Berechne H_21
	std::cout << "H_21: " << (((-4*w(1,1)*w(2,1)/pow(betaE,2))*exp(-((pow(w(1,1),2)*pow(w(2,1),2)*pow(w(3,1),2))/betaE)))-(alphaE*sin(w(1,1))*sin(w(2,1))*cos(w(3,1)))) << std::endl << std::endl;
	//Berechne H_22
	std::cout << "H_22: " << (((2.0/betaE)-(4.0*pow(w(2,1),2)/pow(betaE,2))) * exp(-((pow(w(1,1),2)*pow(w(2,1),2)*pow(w(3,1),2))/betaE)) + alphaE*cos(w(1,1))*cos(w(2,1))*cos(w(3,1))) << std::endl << std::endl;
	//Berechne H_23
	std::cout << "H_23: " << (((-4*w(2,1)*w(3,1)/pow(betaE,2))*exp(-((pow(w(1,1),2)*pow(w(2,1),2)*pow(w(3,1),2))/betaE)))-(alphaE*cos(w(1,1))*sin(w(2,1))*sin(w(3,1)))) << std::endl << std::endl;
	//Berechne H_31
	std::cout << "H_31: " << (((-4*w(1,1)*w(3,1)/pow(betaE,2))*exp(-((pow(w(1,1),2)*pow(w(2,1),2)*pow(w(3,1),2))/betaE)))-(alphaE*sin(w(1,1))*cos(w(2,1))*sin(w(3,1)))) << std::endl << std::endl;
	//Berechne H_32
	std::cout << "H_32: " << (((-4*w(2,1)*w(3,1)/pow(betaE,2))*exp(-((pow(w(1,1),2)*pow(w(2,1),2)*pow(w(3,1),2))/betaE)))-(alphaE*cos(w(1,1))*sin(w(2,1))*sin(w(3,1)))) << std::endl << std::endl;
	//Berechne H_33
	std::cout << "H_33: " << (((2.0/betaE)-(4.0*pow(w(3,1),2)/pow(betaE,2))) * exp(-((pow(w(1,1),2)*pow(w(2,1),2)*pow(w(3,1),2))/betaE)) + alphaE*cos(w(1,1))*cos(w(2,1))*cos(w(3,1))) << std::endl << std::endl;
}

//e) 1
void GradientDescent::executeE1(void)
{
	knn::matrix w = knn::matrix(3,1);
	knn::matrix minErrorW = gradientDescent(w);
	double minError = error(minErrorW);
	GradientDescent::berechneHessematrix(minErrorW);
	std::cout << "Wenn w ein Minimum ist, muss gelten, dass die Hessematrix an der Stelle w positiv definit ist. Da die Hessematrix symmetrisch ist, ist die Matrix positiv definit, wenn alle Elemente der Diagonalen (H_11, H_22, H_33) größer als Null sind. Dies ist jedoch nicht der Fall und somit wurde kein Minimum gefunden. Grund dafür ist das Verhältnis von alpha und beta (s.3.2.a)." << std::endl;
}

//e) 2
void GradientDescent::executeE2(void)
{
	knn::matrix w = knn::matrix(3,1,1);
	w(3,1) = 0;
	knn::matrix minErrorW = gradientDescent(w);
	double minError = error(minErrorW);
	GradientDescent::berechneHessematrix(minErrorW);
	std::cout << "Wenn w ein Minimum ist, muss gelten, dass die Hessematrix an der Stelle w positiv definit ist. Da die Hessematrix symmetrisch ist, ist die Matrix positiv definit, wenn alle Elemente der Diagonalen (H_11, H_22, H_33) größer als Null sind. Dies ist hier gegeben und somit wurde ein Minimum gefunden." << std::endl;

	std::cout << "\nd) Für alle eta werden zufällige w gefunden, für die das Verfahren konvergiert. Allerdings fällt bei eta=4 auf, dass die Laufzeit ohne Abbruch Bedingung für das Konvergieren sehr hoch ist und der Fehler dann nich in der Nähe von 0 sondern von 1 liegt."<< std::endl;
}
#endif // KNN3_GRADIENT_H
