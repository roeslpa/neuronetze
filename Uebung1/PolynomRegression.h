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
	//a_km berechnen
    for(unsigned m=0; m<=mE; m++)
     {
        for(unsigned k=0; k<=mE; k++) 
        {
            //Wähle Element a_km
            elementAE = &AE.operator()(k,m);
            *elementAE = 0;
            for(int p=1; p<=xInE.size(); ++p) 
            {
                //x_p^k+m
                *elementAE += pow(xInE[p], k+m);                
            }
		}
    }
    //b_k berechnen
    for(int k=0.0; k<=mE; k++) 
        {
            //Wähle Element b_k
            elementbE = &bE.operator()(k,0);
            *elementbE = 0;
            for(int p=1; p<=xInE.size(); ++p) 
            {
                //t_p*x_p^k
                *elementbE += tInE[p] * pow(xInE[p], k);                
            }
		}
}

#endif KNN1_POLYNOMREGRESSION_H
