#include "matrix.h"
#include "MIMLP.h"

//Funktionen:
knn::matrix calcXVector(unsigned int x);

// Globale Variblen:
int D;



int main(int argc, char** argv)
{
	//Variablen
	D = 9;
	int M = 2;
	unsigned P = (int)pow(2, D-1);
	knn::matrix xP = knn::matrix(1,P); // Eingabewerte der Trainingsbeispiele
	knn::matrix tP = knn::matrix(1,P); // Erwartete Ausgabewerte der Trainingsbeispiele
	knn::matrix xT = knn::matrix(1,P); // Eingabewerte der Testbeispiele
	unsigned xTLatest, xPLatest; // Hilfsvariablen um zu wissen wie viele Werte bereits in xP bzw. xT gespeichert sind.		
	
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
			unsigned int sum = 0;
			knn::matrix xd;
			xd = calcXVector(i);
			for(unsigned d=2;d<=D+1;d++ )
			{
				sum = xd(1,d); 
			}
			tP(1, xPLatest)= pow(-1, sum+1);
		}
		else
		{
			//Testbeispiele
			xTLatest++;
			xT(1,xTLatest)=i;
		}
	}
	
	
	
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
