#include "matrix.h"
#include "MLP.h"

int main(int argc, char** argv)
{
	double xMin = 0;
	double xMax = 10;
	unsigned M = 20;
	double wMin = -4;
	double wMax = 4;
	unsigned P = 50;
	double lernrate = 0.01;
	
	knn::matrix w,tP, xP, epsilonP;
	double epsilonMin = -0.15;
	double epsilonMax = 0.15;
	
	double x;
	ofstream output41e;

	//Aufgabe 4.1 f)
	w = knn::matrix(3, M+1); 
	w.fillRandom(wMin, wMax);
	output41e.open("out4.1.e.txt"); //ist das nicht aufgabe f?

	MLP mlp = MLP(w, M);
	for(unsigned i=0; i<1000; i++) {
		x = knn::randomFromInterval(xMin, xMax);
		//MLP mlp = MLP(w, M); For die Schleife gezogen (muss doch nicht jedesmal neu initialisiert werden oder?)
		output41e << x << " " << mlp.getY(x) << "\n";
	}
	output41e.close();
	
	//Aufgabe 4.2 a)
	//Trainingsbeispiele erzeugen:
	tP= knn::matrix(1, P);
	xP= knn::matrix(1,P);
	epsilonP = knn::matrix(1,P);
	epsilonP.fillRandom(epsilonMin, epsilonMax);
	for (unsigned p=1; p<=P; p++)
	{
		xP(1,p)= (15*p)/(P-1)-7.5;
		tP(1,p)= exp(-(pow(xP(1,p),2)/10))+epsilonP(1,p);
	}
	//Gewichte zufällig füllen:
	wMin = -5;
	wMax = 5;
	w.fillRandom(wMin, wMax);
	//Gradientenabstieg für eine Epoche:
	for (unsigned p=1; p<=P;p++)
	{
		mlp.gradientDescent(xP(1,p), tP(1,p), lernrate);
	}
	
	//Aufgabe 4.2 b)
}
