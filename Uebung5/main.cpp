#include "matrix.h"
#include "MIMLP.h"
#include <stdio.h>
#include <stdlib.h>

//Funktionen:
knn::matrix calcXVector(unsigned int x);
MIMLP trainingslauf(unsigned M, unsigned D);

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


int main(int argc, char** argv)
{
	//Variablen
	D = 9;
	int M = 2;
	P = (int)pow(2, D-1);
	xP = knn::matrix(1,P); // Eingabewerte der Trainingsbeispiele
	tP = knn::matrix(1,P); // Erwartete Ausgabewerte der Trainingsbeispiele
	xT = knn::matrix(1,P); // Eingabewerte der Testbeispiele
	unsigned xTLatest, xPLatest; // Hilfsvariablen um zu wissen wie viele Werte bereits in xP bzw. xT gespeichert sind.		
	lernrate = 0.01;
	wMin = -2;
	wMax = 2;
	
	// 5.2 b)
	// Die Test- und Trainingsbeispiele werden als ganze Zahlen gespeichert (dies spart Speicherplatz). Aus diesen kann bei Bedarf mithilfe der Funktion calcXVector der entsprechende Eingabevektor berechnet werden.
	xPLatest = 0;
	xTLatest = 0;
	// Mögliche eingabe Werte als ganze Zahl dargestellt gehen von 0 bis 2^D-1
	for (unsigned int i = 0; i < 2*P;i++)

	{
		if ((xTLatest>=P)||(((rand()%2)==0) && xPLatest<P)) // Falls xT schon voll ist, oder der Zufallswert 1 modulo 2 entspricht und wird die aktuelle Zahl in xP gespeichert. Andernfalls in xT.
		{
			//Trainingsbeispiele
			xPLatest++;
			xP(1,xPLatest)=i;
			//Berechnung fon tP=f(xP)
			tP(1, xPLatest)=1;
			knn::matrix xd;
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
		}
	}
	

	MIMLP mimlp22 = trainingslauf(2, 2);
	MIMLP mimlp25 = trainingslauf(2, 5);
	MIMLP mimlp28 = trainingslauf(2, 8);
	MIMLP mimlp210 = trainingslauf(2, 10);

	MIMLP mimlp102 = trainingslauf(10, 2);
	MIMLP mimlp105 = trainingslauf(10, 5);
	MIMLP mimlp108 = trainingslauf(10, 8);
	MIMLP mimlp1010 = trainingslauf(10, 10);
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


MIMLP trainingslauf(unsigned M, unsigned D)
{
	knn::matrix w1 = knn::matrix(D+1,M+1);
	knn::matrix w2 = knn::matrix(1, M+1);
	w1.fillRandom(wMin, wMax);
	w2.fillRandom(wMin, wMax);
	MIMLP mimlp = MIMLP(w1,w2,M,D);
	trainingsfehler = 0;
	testfehler = 0;
	time_t zeit;
	ofstream file;
	stringstream filename;
	filename << "plots/error" << M << "_" << D << ".txt";
	file.open(filename.str());

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
			testfehler+= pow(mimlp.getY(calcXVector(yP(1,p)))-sP(1,p),2);
		}
		testfehler = testfehler/P;
		file << i << " " << trainingsfehler << " " << testfehler << endl;
		cout << i << " " << M << " " << D << endl;
	}
	cout << M << " " << D << " " << time(&zeit)-anfang << " " << trainingsfehler << " " << testfehler << endl;
	
	file.close();
	
	return mimlp;
