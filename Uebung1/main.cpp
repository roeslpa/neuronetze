#include "matrix.h"
#include "PolynomRegression.h"
#include <iostream>

std::vector<std::vector<double> > computeTrainingSin(unsigned int pA)
{
    std::vector<std::vector<double> > value(2);
    std::vector<double> x(pA+1);
    std::vector<double> t(pA+1);
    for(unsigned int p = 1; p<=pA; p++) {
        x.at(p) = ( ((2*(p))-1) * M_PI ) / pA - M_PI + 0.01;
        t.at(p) = sin(x.at(p)/2);
    }
    value.at(0) = x;
    value.at(1) = t;
    return value;
}

std::vector<std::vector<double> > computeTestSin(unsigned int pA)
{
    std::vector<std::vector<double> > value(2);
    std::vector<double> x(pA+1);
    std::vector<double> t(pA+1);
    for(unsigned int p = 1; p<=pA; p++) {
        x.at(p) = (2 * p * M_PI ) / pA - M_PI + 0.01;
        t.at(p) = sin(x.at(p)/2);
    }
    value.at(0) = x;
    value.at(1) = t;
    return value;
}

std::vector<std::vector<double> > computeTrainingSinc(unsigned int pA)
{
    std::vector<std::vector<double> > value(2);
    std::vector<double> x(pA+1);
    std::vector<double> t(pA+1);
    for(unsigned int p = 1; p<=pA; p++) {
        x.at(p) = ( ((2*p)-1) * M_PI ) / pA - M_PI + 0.01;
        t.at(p) = sin(x.at(p)*4)/x.at(p);
    }
    value.at(0) = x;
    value.at(1) = t;
    return value;
}

std::vector<std::vector<double> > computeTestSinc(unsigned int pA)
{
    std::vector<std::vector<double> > value(2);
    std::vector<double> x(pA+1);
    std::vector<double> t(pA+1);
    for(unsigned int p = 1; p<=pA; p++) {
        x.at(p) = (2 * p * M_PI ) / pA - M_PI + 0.01;
        t.at(p) = sin(x.at(p)*4)/x.at(p);
    }
    value.at(0) = x;
    value.at(1) = t;
    return value;
}


int main(int argc, char** argv) {
    
    knn::init();

    //number of training examples: parameter 1, if given, else 11
    unsigned int P = argc > 1 ? (unsigned)atoi(argv[1]) : 11;
    //maximum polynomal degree: parameter 2, if given, else 19
    unsigned int MMax = argc > 2 ? (unsigned)atoi(argv[2]) : 19;

    std::ofstream sinOutL;
    sinOutL.open("SinError.txt",std::ios::out);
    std::ofstream sincOutL;
    sincOutL.open("SincError.txt",std::ios::out);

    for (unsigned int mL=0;mL<=MMax;++mL)
    {
        PolynomRegression regressorL(mL);
        std::vector<std::vector<double> > trainingDataL=computeTrainingSin(P);
        regressorL.setXandT(trainingDataL[0],trainingDataL[1]);
        regressorL.computeAandBandW();
        sinOutL<<mL<<"	"<<regressorL.error()<<"	";
        std::vector<std::vector<double> > testDataL=computeTestSin(P);
        regressorL.setXandT(testDataL[0],testDataL[1]);
        sinOutL<<regressorL.error()<<std::endl;
    }

    for (unsigned int mL=0;mL<=MMax;++mL)
    {
        PolynomRegression regressorL(mL);
        std::vector<std::vector<double> > trainingDataL=computeTrainingSinc(P);
        regressorL.setXandT(trainingDataL[0],trainingDataL[1]);
        regressorL.computeAandBandW();
        sincOutL<<mL<<"	"<<regressorL.error()<<"	";
        std::vector<std::vector<double> > testDataL=computeTestSinc(P);
        regressorL.setXandT(testDataL[0],testDataL[1]);
        sincOutL<<regressorL.error()<<std::endl;
    }
    sinOutL.close();
    sincOutL.close();

    return 0;
}
