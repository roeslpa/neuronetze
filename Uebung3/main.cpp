#include "matrix.h"
#include "GradientDescent.h"

int main(int argc, char** argv)
{
	//d
	std::cout << "d) start " << std::endl << std::endl;
	std::vector<double> etas{ 0.01, 0.03, 0.09, 0.1, 0.3, 0.9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 6, 8, 10 };

	for (unsigned int i = 0; i < etas.size(); i++) 
	{
		GradientDescent gradientDescentDL(etas[i], 0.5, 500.0, 10.0);
		gradientDescentDL.executeD(100);
	}

	//e)
	GradientDescent gradientDescentEL(0.01, -0.05, 500.0, 10.0);

	//e) 1
	std::cout << "e) Part 1 " << std::endl << std::endl;
	gradientDescentEL.executeE1();

	//e) 2
	std::cout << "e) Part 2" << std::endl << std::endl;
	gradientDescentEL.executeE2();

	std::cout << "...reached the end of the universe..." << std::endl << std::endl;
	cin.get();
}