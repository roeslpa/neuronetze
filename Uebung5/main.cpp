#include "matrix.h"
#include "MIMLP.h"
#include <stdio.h>
#include <stdlib.h>

// Daniel Teuchert und Paul Rösler

//Funktionen:
knn::matrix calcXVector(unsigned int x);
MIMLP trainingslauf(unsigned M, unsigned D, bool printErrors);
void plot(MIMLP mimlp, ofstream* file);

// Globale Variblen:
int D;
unsigned P;
double wMin;
double wMax;
double trainingsfehler;
double testfehler;
double lernrate;
knn::matrix xP;
knn::matrix tP;
knn::matrix xT;
knn::matrix tT;


int main(int argc, char** argv)
{
	//Variablen
	D = 10;
	int M = 10;
	P = (int)pow(2, D-1);
	xP = knn::matrix(1,P); // Eingabewerte der Trainingsbeispiele
	tP = knn::matrix(1,P); // Erwartete Ausgabewerte der Trainingsbeispiele
	xT = knn::matrix(1,P); // Eingabewerte der Testbeispiele
	tT = knn::matrix(1,P); // Erwartete Ausgabewerte der Testbeispiele
	unsigned xTLatest, xPLatest; // Hilfsvariablen um zu wissen wie viele Werte bereits in xP bzw. xT gespeichert sind.		
	lernrate = 0.01;
	wMin = -2;
	wMax = 2;
	
	// 5.2 b)
	// Die Test- und Trainingsbeispiele werden als ganze Zahlen gespeichert (dies spart Speicherplatz). Aus diesen kann bei Bedarf mithilfe der Funktion calcXVector der entsprechende Eingabevektor berechnet werden.
	xPLatest = 0;
	xTLatest = 0;
	// Mögliche eingabe Werte als ganze Zahl dargestellt gehen von 0 bis 2^D-1
	knn::matrix xd;
	for (unsigned int i = 0; i < 2*P;i++)
	{
		if ((xTLatest>=P)||(((rand()%2)==0) && xPLatest<P)) // Falls xT schon voll ist, oder der Zufallswert 1 modulo 2 entspricht und wird die aktuelle Zahl in xP gespeichert. Andernfalls in xT.
		{
			//Trainingsbeispiele
			xPLatest++;
			xP(1,xPLatest)=i;
			//Berechnung fon tP=f(xP)
			tP(1, xPLatest)=1;
			xd = calcXVector(i);
			for(unsigned d=2;d<=D+1;d++ )
			{
				tP(1, xPLatest)*= pow(-1, xd(1,d));
			}
		}
		else
		{
			//Testbeispiele
			xTLatest++;
			xT(1,xTLatest)=i;
			//Berechnung fon tP=f(xP)
			tT(1, xTLatest)=1;
			xd = calcXVector(i);
			for(unsigned d=2;d<=D+1;d++ )
			{
				tT(1, xTLatest)*= pow(-1, xd(1,d));
			}
		}
	}
	
	//5.2.c und d
	MIMLP mimlp22 = trainingslauf(2, 2, 1);
	MIMLP mimlp25 = trainingslauf(2, 5, 1);
	MIMLP mimlp28 = trainingslauf(2, 8, 1);
	MIMLP mimlp210 = trainingslauf(2, 10, 1);

	MIMLP mimlp52 = trainingslauf(5, 2, 0);
	MIMLP mimlp55 = trainingslauf(5, 5, 0);
	MIMLP mimlp58 = trainingslauf(5, 8, 0);
	MIMLP mimlp510 = trainingslauf(5, 10, 0);

	MIMLP mimlp102 = trainingslauf(10, 2, 1);
	MIMLP mimlp105 = trainingslauf(10, 5, 1);
	MIMLP mimlp108 = trainingslauf(10, 8, 1);
	MIMLP mimlp1010 = trainingslauf(10, 10, 1);

	/*
	5.2.d) Es ist zu erkennen, dass die Genauigkeit eher von der Anzahl der verdeckten Neuronen als von den Eingabeneuronen abhängt.
	Das kann sowohl anhand des Plots aus 5.2.c als auch 5.2.d gesehen werden.
	Besonders für D=10 kann sich das Netzwerk gut an die Funktion anpassen.
	*/
}

//Bit-Vektor aus Eingabewert berechnen (etwas anders als die Formel vom Hilfsblatt, aber dennoch korrekt)
knn::matrix calcXVector(unsigned int x)
{
	knn::matrix xd = knn::matrix(1, D+1);
	
	xd(1,1)=1; //x0=1
	for(unsigned d=2; d<=D+1; d++)
	{
		xd(1,d)= ((x&(int)pow(2, d-1))!=0?1:0);
	}
	return xd;
}


MIMLP trainingslauf(unsigned M, unsigned D, bool printErrors)
{
	knn::matrix w1 = knn::matrix(D+1,M+1);
	knn::matrix w2 = knn::matrix(1, M+1);
	w1.fillRandom(wMin, wMax);
	w2.fillRandom(wMin, wMax);
	MIMLP mimlp = MIMLP(w1,w2,M,D);
	trainingsfehler = 0;
	testfehler = 0;
	time_t zeit;
	ofstream errors;
	if(printErrors) {
		stringstream errorsName;
		errorsName << "plots/error" << M << "_" << D << ".txt";
		errors.open(errorsName.str());
	}
	long anfang = time(&zeit);
	for (unsigned i=0; i<10000;i++)
	{
		// Gradientenabstieg für eine Epoche
		for (unsigned p=1; p<=P;p++)
		{
			mimlp.gradientDescent(calcXVector(xP(1,p)), tP(1,p), lernrate);
		}

		//Trainingsfehlerberechnung
		for (unsigned p=1;p<=P;p++)
		{
			trainingsfehler+= pow(mimlp.getY(calcXVector(xP(1,p)))-tP(1,p),2);
		}
		trainingsfehler = trainingsfehler/P;

		//Testfehlerberechnung
		for (unsigned p=1;p<=P;p++)
		{
			testfehler+= pow(mimlp.getY(calcXVector(xT(1,p)))-tT(1,p),2);
		}
		testfehler = testfehler/P;
		//error ausgeben
		if(printErrors) {
			errors << i << " " << trainingsfehler << " " << testfehler << endl;
		}
		cout << i << " " << M << " " << D << endl;
	}
	ofstream plotFile;
	stringstream plotName;
	plotName << "plots/plot_" << M << "_" << D << ".txt";
	plotFile.open(plotName.str());
	plot(mimlp, &plotFile);
	cout << M << " " << D << " " << time(&zeit)-anfang << " " << trainingsfehler << " " << testfehler << endl;
	if(printErrors) {
		errors.close();
	}
	
	
	return mimlp;
}

//5.2.d ausgegeben des Grafen
void plot(MIMLP mimlp, ofstream *file) {
	for(unsigned i=0; i<100; i++) {
		*file << i+100 << " " << mimlp.getY(calcXVector(i+100)) << endl;
	}
	(*file).close();
}