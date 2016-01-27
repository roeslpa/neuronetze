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
	double lernrate = 0.01;
	double wMin = -2;
	double wMax = 2;
	double trainingsfehler;
	double testfehler;
	
	
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
	

	MIMLP mimlp22 = trainingslauf(2, 2);
	MIMLP mimlp25 = trainingslauf(2, 5);
	MIMLP mimlp28 = trainingslauf(2, 8);
	MIMLP mimlp210 = trainingslauf(2, 10);

	MIMLP mimlp102 = trainingslauf(10, 2);
	MIMLP mimlp105 = trainingslauf(10, 5);
	MIMLP mimlp108 = trainingslauf(10, 8);
	MIMLP mimlp1010 = trainingslauf(10, 10);

	/*Beispielausgabe:
	2 1 0.017133
	10 1 0.00936311
	60 7 0.00701061
	Die Laufzeit verlängert sich, wie erwartet mit steigender Komplexität des Netzwerkes (Pardon für die Ungenauigkeit).
	Die Genauigkeit steigt von M=2 zu M=10 rapide, von M=10 zu M=60 aber nur noch sehr wenig. Dies steht in keinem Verhältnis zur
	deutlich längeren Laufzeit.
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
	}
	cout << M << " " << D << " " << time(&zeit)-anfang << " " << trainingsfehler << " " << testfehler << endl;
	
	return mimlp;
}