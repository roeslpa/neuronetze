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
        y += wE(m+1,1) * (pow(xA,m));
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

    //double elementwE;
	//a_km berechnen
    for(unsigned m=1; m<=mE+1; m++) {
        for(unsigned k=1; k<=mE+1; k++) {
            //Wähle Element a_km
            for(int p=1; p<=xInE.size(); ++p) {
                //x_p^k+m
                AE(m,k) += pow(xInE[p-1], k+m);
            }
		}
    }
    //b_k berechnen
    for(int k=1; k<=mE+1; k++) {
        //Wähle Element b_k
        for(int p=1; p<=xInE.size(); ++p) {
            //t_p*x_p^k
            bE(k,1) += tInE[p-1] * pow(xInE[p-1], k);                
        }
	}

    //A^-1 berechnen
    AE.invert();

    wE = AE * bE;

    //A berechnen
    AE.invert();
}

#endif KNN1_POLYNOMREGRESSION_H
