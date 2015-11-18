#ifndef KNN1_POLYNOMREGRESSION_H
#define KNN1_POLYNOMREGRESSION_H

#include <iostream>
#include "matrix.h"

using namespace std;

class PolynomRegression {
public:

    PolynomRegression(unsigned int mA);
    void setXandT(std::vector<double>xA,std::vector<double>tA);
    void computeAandBandW(void);
    inline double y(double xA);
    double error(void);


private:

    std::vector<double> xInE;
    std::vector<double> tInE;
    knn::matrix AE;
    knn::matrix bE;
    knn::matrix wE;
    unsigned int mE;
};

PolynomRegression::PolynomRegression(unsigned int mA)
{
    //(...) TODO 1
    //xInE = new std::std::vector<double>;
    mE = mA;
    AE = *(new knn::matrix(mA+1, mA+1, 0));
    bE = *(new knn::matrix(mA+1, 1, 0));
    wE = *(new knn::matrix(mA+1, 1, 0));
}


void PolynomRegression::setXandT(std::vector<double> xA, std::vector<double> tA)
{
    xInE=xA;
    tInE=tA;
}

double PolynomRegression::y(double xA)
{
    double y = 0;
    for(unsigned m=0; m<=mE; m++) {
        y += *(&wE.operator()(m+1,1)) * (pow(xA,m));
    }
    return y;
}

double PolynomRegression::error(void)
{
    double E = 0;
    for(int p=1; p<=xInE.size(); p++) {
        E += pow(y(xInE[p-1])-tInE[p-1],2);
    }
    return E;
}

void PolynomRegression::computeAandBandW(void)
{
    double* elementAE;
    double* elementbE;
    double* elementwE;
    double* aIK;
    double* aKK;
    double* aIJ;
    double* aKJ;
    //double elementwE;
	//a_km berechnen
    for(unsigned m=1; m<=mE+1; m++) {
        for(unsigned k=1; k<=mE+1; k++) {
            //Wähle Element a_km
            elementAE = &AE.operator()(m,k);
            *elementAE = 0;
            for(int p=1; p<=xInE.size(); ++p) {
                //x_p^k+m
                *elementAE += pow(xInE[p-1], k+m);                
            }
		}
    }
    //b_k berechnen
    for(int k=1; k<=mE+1; k++) {
        //Wähle Element b_k
        elementbE = &bE.operator()(k,1);
        *elementbE = 0;
        for(int p=1; p<=xInE.size(); ++p) {
            //t_p*x_p^k
            *elementbE += tInE[p-1] * pow(xInE[p-1], k);                
        }
	}

    //A^-1 berechnen
    AE.invert();

    //w = A^-1 * b
    for(int j=1; j<=mE+1; j++) {
        elementwE = &wE.operator()(j,1);
        for(int i=1; i<=mE+1; i++) {
            elementAE = &AE.operator()(j,i);
            elementbE = &bE.operator()(i,1);
            *elementwE += (*elementAE) * (*elementbE);
        }
    }

    //A berechnen
    AE.invert();
}

#endif KNN1_POLYNOMREGRESSION_H
