#include "matrix.h"
#include "ProbabilityDistribution.h"

int main(int argc, char** argv){

	//take command line parameters or use default values
	unsigned SL = (argc > 1) ? atoi(argv[1]) : 100000;
	unsigned NL = (argc > 2) ? atoi(argv[2]) : 10;
	double deltaL = (argc > 3) ? atof(argv[3]) : 0.1;
	double x1MinL = (argc > 4) ? atof(argv[4]) : -1.;
	double x1MaxL = (argc > 5) ? atof(argv[5]) : 1.;
	double x2MinL = (argc > 6) ? atof(argv[6]) : -3.;
	double x2MaxL = (argc > 7) ? atof(argv[7]) : 1.;

	ProbabilityDistribution probabilityL(NL, SL, x1MinL, x1MaxL, x2MinL, x2MaxL, deltaL);
	probabilityL.execute();
}