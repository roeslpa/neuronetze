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
    knn::matrix AE = new matrix(mA+1, mA+1, 0);
    knn::matrix bE = new matrix(mA+1, 1, 0);
    knn::matrix wE = new matrix(mA+1, 1, 0);
}


void PolynomRegression::setXandT(std::vector<double> xA, std::vector<double> tA)
{
    xInE=xA;
    tInE=tA;
}

double PolynomRegression::y(double xA)
{
    return 2.5;
}

double PolynomRegression::error(void)
{
    return 3.4;
}

void PolynomRegression::computeAandBandW(void)
{
    double* elementAE;
    double* elementbE;
    //double elementwE;
    int P = xInE.size();
    double xStep = 1.0;

    for(int m=0; m<=mE; m++) {
        for(int k=0; k<=mE; k++) {
            elementAE = AE.operator(k,m);
            for(int p=1; p<=P; i++) {
                *elementAE += xStep;
            }
            xStep *= xStep;
        }
    }
}

#endif KNN1_POLYNOMREGRESSION_H
