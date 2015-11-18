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
    //Initialisieren von A, b, w und der Modellkomplexität M
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
    double y;
    //Berechne y(w,x) = Summe von m=0 bis M von w_m*x^m
    for(unsigned m=0; m<=mE; m++) 
    {
    	//w_m*x^m
        y += wE(m+1,1) * (pow(xA,m));
    }
    return y;
}

double PolynomRegression::error(void)
{
    double E = 0;
    //Berechne Error(w)
    for(int p=1; p<=xInE.size(); p++) {
    	//(y(w,x_p)-t_p)^2
        E += pow((y(xInE[p-1])-tInE[p-1]),2);
    }
    return E;
}

void PolynomRegression::computeAandBandW(void)
{
    //a_km berechnen
    for(unsigned m=0; m<=mE; m++) {
        for(unsigned k=0; k<=mE; k++) {
            for(int p=1; p<=xInE.size(); ++p) {
                //x_p^k+m
                AE(m+1,k+1) += pow(xInE[p-1], k+m);
            }
		}
    }
    //b_k berechnen
    for(unsigned k=0; k<=mE; k++) {
        for(int p=1; p<=xInE.size(); ++p) {
            //t_p*x_p^k
            bE(k+1,1) += tInE[p-1] * pow(xInE[p-1], k);                
        }
	}

    //A^-1 berechnen
    AE.invert();
    
	//w berechnen als A^-1*b
    wE = AE * bE;

    //A berechnen
    AE.invert();
}

#endif KNN1_POLYNOMREGRESSION_H
