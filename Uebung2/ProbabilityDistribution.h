#ifndef KNN1_ProbabilityDistribution_H
#define KNN1_ProbabilityDistribution_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "matrix.h"

using namespace std;

class ProbabilityDistribution {
public:

	ProbabilityDistribution(unsigned int nA, unsigned int noMeansA, double xMinA, double xMaxA, double yMinA, double yMaxA, double deltaA);
	void execute(void);

private:

	unsigned int nE;
	unsigned int noMeansE;
	double xMinE, xMaxE, yMinE, yMaxE, deltaE;

	knn::matrix meanValuesE;
	knn::matrix nuE;
	knn::matrix sigmaE;
	knn::matrix histogramE;

	void generateMeanValues(void);
	void estimateMeanAndVariance(void);
	void createHistogram(void);

};

ProbabilityDistribution::ProbabilityDistribution(unsigned int nA,unsigned int noMeansA,double xMinA,double xMaxA,double yMinA,double yMaxA,double deltaA)
{
	nE = nA;
	noMeansE = noMeansA;
	xMinE = xMinA;
	xMaxE = xMaxA;
	yMinE = yMinA;
	yMaxE = yMaxA;
	deltaE = deltaA;

	meanValuesE = new knn::matrix(noMeansE, 2, 0.0);
	nuE = new knn::matrix(1, 2, 0);
	sigmaE = new knn::matrix(1, 2, 0);
	histogramE = new knn::matrix(, , 0);

	std::srand(std::time(0));
}

void ProbabilityDistribution::execute(void)
{
	generateMeanValues();
	estimateMeanAndVariance();
	createHistogram();
}

void ProbabilityDistribution::generateMeanValues(void)
{
	double max4 = ;
	double max2 = 2.0/RAND_MAX;
	knn::matrix random = new knn::matrix(1, 2, 0.0);

	//Berechne Zufallsvariablen
	for(unsigned i=1; i<=noMeansE; i++) {
		//Berechne eine Zufallsvariable
		for(unsigned n=0; n<nE; n++) {
			randomVector(random);
			meanValuesE(i, 1) += random(1, 1);
			meanValuesE(i, 2) += random(1, 2);
		}
		meanValuesE(i, 1) /= nE;
		meanValuesE(i, 2) /= nE;
	}
}

void ProbabilityDistribution::randomVector(knn::matrix vector) {
	//Berechne einen Zufallsvektor: [-1,1]x[-3,1]
	vector(1, 1) = std::rand()/RAND_MAX*2.0 - 1;
	vector(1, 2) = std::rand()/RAND_MAX*4.0 - 3;
}

void ProbabilityDistribution::estimateMeanAndVariance(void)
{
	//Schätze Erwartungswert
	for(unsigned p=1; p<=noMeansE; i++) {
		nuE(1, 1) += meanValuesE(p, 1);
		nuE(1, 2) += meanValuesE(p, 2);
	}
	nuE(1, 1) /= noMeansE;
	nuE(1, 2) /= noMeansE;
	
	//Schätze Standardabweichung
	for(unsigned p=1; p<=noMeansE; i++) {
		sigmaE(1, 1) += pow(meanValuesE(p, 1)-nuE(1, 1),2);
		sigmaE(1, 2) += pow(meanValuesE(p, 2)-nuE(1, 2),2);
	}
	sigmaE(1, 1) /= noMeansE-1;
	sigmaE(1, 2) /= noMeansE-1;
}

void ProbabilityDistribution::createHistogram(void)
{
	(...)
}
#endif // KNN1_ProbabilityDistribution_H