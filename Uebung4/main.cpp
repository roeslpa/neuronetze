#include "matrix.h"
#include "MLP.h"

MLP trainingslauf(unsigned m);
void printTraining(void);
void printFunktion(MLP mlp2, MLP mlp10, MLP mlp60);

double lernrate = 0.01;
unsigned M = 20;
double wMin = -4;
double wMax = 4;
knn::matrix w,tP, xP, epsilonP;
double restfehler;
unsigned P;

int main(int argc, char** argv)
{
	double xMin = 0;
	double xMax = 10;

	P = 50;

	double epsilonMin = -0.15;
	double epsilonMax = 0.15;
	
	double x;
	ofstream output41f;

	//Aufgabe 4.1 f) (Da die Daten unsortiert geschrieben werden, müssen sie vor dem Plotten sortiert werden)
	w = knn::matrix(3, M+1); 
	w.fillRandom(wMin, wMax);
	output41f.open("out4.1.f.txt");
	MLP mlp = MLP(w, M);
	for(unsigned i=0; i<1000; i++) {
		x = knn::randomFromInterval(xMin, xMax);
		output41f << x << " " << mlp.getY(x) << "\n";
	}
	output41f.close();
	
	//Aufgabe 4.2 b)
	//Trainingsbeispiele erzeugen:
	tP= knn::matrix(1, P);
	xP= knn::matrix(1,P);
	epsilonP = knn::matrix(1,P);
	epsilonP.fillRandom(epsilonMin, epsilonMax);
	for (unsigned p=1; p<=P; p++)
	{
		xP(1,p)= (15.0*(p-1.0))/(P-1.0)-7.5;
		tP(1,p)= exp(-(pow(xP(1,p),2)/10.0))*sin(xP(1,p))+epsilonP(1,p);
	}
	//Gewichte zufällig füllen:
	wMin = -5;
	wMax = 5;
	w.fillRandom(wMin, wMax);
	mlp=MLP(w,M);
	//Gradientenabstieg für eine Epoche:
	for (unsigned p=1; p<=P;p++)
	{
		mlp.gradientDescent(xP(1,p), tP(1,p), lernrate);
	}
	
	//Aufgabe 4.2 c)
	//MLP mlp2, mlp10, mlp60;

	MLP mlp2 = trainingslauf(2);
	MLP mlp10 = trainingslauf(10);
	MLP mlp60 = trainingslauf(60);

	//Aufgabe 4.2 d)
	printTraining();
	printFunktion(mlp2, mlp10, mlp60);

	
}
MLP trainingslauf(unsigned m)
{
	M = m;
	w = knn::matrix(3, M+1); 
	w.fillRandom(wMin, wMax);
	MLP mlp = MLP(w,M);
	lernrate=0.01;
	restfehler = 0;
	for (unsigned i=0; i<10000;i++)
	{
		// Gradientenabstieg für eine Epoche
		for (unsigned p=1; p<=P;p++)
		{
			mlp.gradientDescent(xP(1,p), tP(1,p), lernrate);
		}	
		//Restfehlerberechnung
		for (unsigned p=1;p<=P;p++)
		{
			restfehler+= pow(mlp.getY(xP(1,p))-tP(1,p),2);
		}
		restfehler = restfehler/P;
	}

	return mlp;
}

//Aufgabe 4.2.e
void printTraining(void) {
	ofstream outputTraining;
	outputTraining.open("beispiele.txt");

	//Schreibe die Trainingspaare
	for(unsigned p=1; p<=P; p++) {
		outputTraining << xP(1,p) << " " << tP(1,p) << "\n";
	}
	outputTraining.close();
}

void printFunktion(MLP mlp2, MLP mlp10, MLP mlp60) {
	ofstream outputFunktion2, outputFunktion10, outputFunktion60;
	outputFunktion2.open("funktion2.txt");
	outputFunktion10.open("funktion10.txt");
	outputFunktion60.open("funktion60.txt");

	//Berechne und schreibe die Funktionswerte in 0.1 Schritten je Trainingslauf
	for(unsigned i=0; i<=100; i++) {
		outputFunktion2 << i/10.0 << " " << mlp2.getY(i/10.0) << "\n";
		outputFunktion10 << i/10.0 << " " << mlp10.getY(i/10.0) << "\n";
		outputFunktion60 << i/10.0 << " " << mlp60.getY(i/10.0) << "\n";
	}
	outputFunktion2.close();
	outputFunktion10.close();
	outputFunktion60.close();
}