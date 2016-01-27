#include "matrix.h"
#include "MIMLP.h"

//Funktionen:
knn::matrix calcXVector(unsigned int x);

// Globale Variblen:
int D;


int main(int argc, char** argv)
{
	//Variablen
	int D = 9;
	int M = 2;
	unsigned P = (int)pow(2, D-1);
	knn::matrix xP = knn::matrix(1,P);
	knn::matrix tP = knn::matrix(1,P);
	unsigned tLatest, xLatest; // Hilfsvariablen um zu wissen wie viele Werte bereits in xP bzw. tP gespeichert sind.		
	
	// 5.2 b)
	// Die Test- und Trainingsbeispiele werden als ganze Zahlen gespeichert (dies spart Speicherplatz). Aus diesen kann bei Bedarf mithilfe der Funktion calcXVector der entsprechende Eingabevektor berechnet werden.
	tLatest = 0;
	xLatest = 0;
	// Mögliche eingabe Werte als ganze Zahl dargestellt gehen von 0 bis 2^D-1
	for (unsigned int i = 0; i < 2*P;i++)
	{
		if ((tLatest>=P)||(((rand()%2)==0) && xLatest<P)) // Falls xP schon voll ist, oder der Zufallswert 1 modulo 2 entspricht und wird die aktuelle Zahl in tP gespeichert. Andernfalls in xP.
		{
			xLatest++;
			xP(1,xLatest)=i;
		}
		else
		{
			tLatest++;
			tP(1,tLatest)=i;
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
