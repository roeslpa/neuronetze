#ifndef KNN1_ProbabilityDistribution_H
#define KNN1_ProbabilityDistribution_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "matrix.h"
#include <math.h>

using namespace std;

class ProbabilityDistribution {
public:

	ProbabilityDistribution(unsigned int nA, unsigned int noMeansA, double xMinA, double xMaxA, double yMinA, double yMaxA, double deltaA);
	void execute(void);

private:

	unsigned int nE;
	unsigned int noMeansE;
	double xMinE, xMaxE, yMinE, yMaxE, deltaE, errorE;

	knn::matrix meanValuesE;
	knn::matrix nuE;
	knn::matrix sigmaE;
	knn::matrix histogramE;
	knn::matrix normVertE;

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
	histogramE = knn::matrix((xMaxE-xMinE)/deltaE, (yMaxE-yMinE)/deltaE, 0.0);
	normVertE = knn::matrix((xMaxE-xMinE)/deltaE, (yMaxE-yMinE)/deltaE, 0.0);
	errorE = 0.0;

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
	double x1, x2, zweiPiSigma;
	for(unsigned i=1; i<=noMeansE; i++) {
		//Diskretisieren
		x = (meanValuesE(i, 1)+1)/deltaE+1;
		y = (meanValuesE(i, 2)+3)/deltaE+1;

		histogramE(x,y) += 1;
	}

	//Normalverteilung berechnen, Histogramm durch S*delta^2
	zweiPiSigma = 1.0/(2.0*M_PI*sigmaE(1,1)*sigmaE(1,2));
	for(unsigned x=1; x<=(xMaxE-xMinE)/deltaE; x++) {
		x1 = (x-1)*deltaE-1+(deltaE/2);
		for(unsigned y=1; y<=(yMaxE-yMinE)/deltaE; y++) {
			histogramE(x,y) /= noMeansE * pow(deltaE,2);
			x2 = (y-1)*deltaE-3+(deltaE/2);
			normVertE(x,y) = zweiPiSigma * exp( ((-1.0)/2.0) * ( (pow(x1-nuE(1,1),2)/pow(sigmaE(1,1),2)) + (pow(x2-nuE(1,2),2)/pow(sigmaE(1,2),2)) ));
		}
	}

	//Error SSE berechnen
	for(unsigned x=1; x<=(xMaxE-xMinE)/deltaE; x++) {
		for(unsigned y=1; y<=(yMaxE-yMinE)/deltaE; y++) {
			errorE += pow(normVertE(x,y)-histogramE(x,y), 2);
		}
	}

	printf("Error: %lf\n", errorE);
	//Pausibilitätscheck
	for(y=1; y<=10 * (yMaxE-yMinE); y++) {
		for(x=1; x<=10 * (xMaxE-xMinE); x++) {
			printf("%lf ", normVertE(x,y));
		}
		printf("\n");
	}
	printf("\n");
	for(y=1; y<=10 * (yMaxE-yMinE); y++) {
		for(x=1; x<=10 * (xMaxE-xMinE); x++) {
			printf("%lf ", histogramE(x,y));
		}
		printf("\n");
	}
}
#endif // KNN1_ProbabilityDistribution_H