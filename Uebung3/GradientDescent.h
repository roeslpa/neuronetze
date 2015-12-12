#ifndef KNN3_GRADIENT_H
#define KNN3_GRADIENT_H

#include "matrix.h"

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
	(...)
}

//b) gradient function
knn::matrix GradientDescent::gradient(knn::matrix wA)
{
	(...)
}

//help-function calculate the change for gradientDescent
double GradientDescent::calculateChange(knn::matrix wInitOldA, knn::matrix wInitCurrentA)
{
	(...)
}

//c) 
knn::matrix GradientDescent::gradientDescent(knn::matrix wInitA)
{
	double errorStartL = error(wInitA);
	knn::matrix wInitCurrentL = wInitA;
	unsigned int counterL = 0;
	(...)

	while (...)
	{
		(...)
		++counterL;
	}

	double errorEndL = error(wInitCurrentL);

	if (!isnan(errorEndL) && (errorEndL<minErrorE))
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
	std::cout << "run started for eta: " << etaE << std::endl;
	for (unsigned int kL = 0; kL<noInitVecsA; ++kL)
	{
		std::cout << "start iteration: " << kL+1 << std::endl << std::endl;
		(...)
	}
	std::cout << "run finished for eta: " << etaE << std::endl << std::endl;
	std::cout << "min error: " << minErrorE << " at: (" << minErrorWE(1, 1) << "," << minErrorWE(2, 1) << "," << minErrorWE(3, 1) << ") " << std::endl << std::endl;
}

//e) 1
void GradientDescent::executeE1(void)
{
	(...)
}

//e) 2
void GradientDescent::executeE2(void)
{
	(...)
}
#endif // KNN3_GRADIENT_H