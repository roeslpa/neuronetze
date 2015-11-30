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

	meanValuesE = knn::matrix(noMeansE, 2, 0.0);
	nuE = knn::matrix(1, 2, 0);
	sigmaE = knn::matrix(1, 2, 0);
	histogramE = knn::matrix(10 * (xMaxE-xMinE), 10 * (yMaxE-yMinE), 0.0);

	knn::init();
}

void ProbabilityDistribution::execute(void)
{
	generateMeanValues();
	estimateMeanAndVariance();
	createHistogram();
}

void ProbabilityDistribution::generateMeanValues(void)
{
	knn::matrix randomX1 = knn::matrix(nE, 1, 0.0);
	knn::matrix randomX2 = knn::matrix(nE, 1, 0.0);

	//Berechne Zufallsvariablen
	for(unsigned i=1; i<=noMeansE; i++) {
		//Berechne eine Zufallsvariable
		randomX1.fillRandom(-1,1);
		randomX2.fillRandom(-3,1);
		for(unsigned n=1; n<=nE; n++) {
			meanValuesE(i, 1) += randomX1(n, 1);
			meanValuesE(i, 2) += randomX2(n, 1);
		}
		meanValuesE(i, 1) /= nE;
		meanValuesE(i, 2) /= nE;
	}
}

void ProbabilityDistribution::estimateMeanAndVariance(void)
{
	//Schätze Erwartungswert
	for(unsigned p=1; p<=noMeansE; p++) {
		nuE(1, 1) += meanValuesE(p, 1);
		nuE(1, 2) += meanValuesE(p, 2);
	}
	nuE(1, 1) /= noMeansE;
	nuE(1, 2) /= noMeansE;
	
	//Schätze Standardabweichung
	for(unsigned p=1; p<=noMeansE; p++) {
		sigmaE(1, 1) += pow(meanValuesE(p, 1)-nuE(1, 1),2);
		sigmaE(1, 2) += pow(meanValuesE(p, 2)-nuE(1, 2),2);
	}
	sigmaE(1, 1) /= noMeansE-1;
	sigmaE(1, 2) /= noMeansE-1;
}

void ProbabilityDistribution::createHistogram(void)
{
	unsigned x, y;
	for(unsigned i=1; i<=noMeansE; i++) {
		//Diskretisieren
		x = (meanValuesE(i, 1)+1)*10+1;
		y = (meanValuesE(i, 2)+3)*10+1;

		histogramE(x,y) += 1;
	}

	//Pausibilitätscheck
	/*for(y=1; y<=10 * (yMaxE-yMinE); y++) {
		for(x=1; x<=10 * (xMaxE-xMinE); x++) {
			histogramE(x,y) /= noMeansE*0.01;
			printf("%lf ", histogramE(x,y));
		}
		printf("\n");
	}*/
}
#endif // KNN1_ProbabilityDistribution_H