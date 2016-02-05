#include "matrix.h"
#include "KM.h"

int main(int argc, char** argv) {
	KM kohonenMap1 = KM(true);
	knn::matrix x;

	for(unsigned i=0; i<1000; i++) {
		x = kohonenMap1.getRandomX();
		kohonenMap1.kohonenLernregel(x);
	}
	
	kohonenMap1.ausgabe((char*)"NormalerExponent1000Iterationen.txt");
	for(unsigned i=1000; i<10000; i++) {
		x = kohonenMap1.getRandomX();
		kohonenMap1.kohonenLernregel(x);
	}
	kohonenMap1.ausgabe((char*)"NormalerExponent10000Iterationen.txt");
	for(unsigned i=10000; i<100000; i++) {
		x = kohonenMap1.getRandomX();
		kohonenMap1.kohonenLernregel(x);
	}
	kohonenMap1.ausgabe((char*)"NormalerExponent100000Iterationen.txt");
	for(unsigned i=100000; i<500000; i++) {
		x = kohonenMap1.getRandomX();
		kohonenMap1.kohonenLernregel(x);
	}
	kohonenMap1.ausgabe((char*)"NormalerExponent500000Iterationen.txt");

	KM kohonenMap2 = KM(false);

	for(unsigned i=0; i<1000; i++) {
		x = kohonenMap2.getRandomX();
		kohonenMap2.kohonenLernregel(x);
	}
	kohonenMap2.ausgabe((char*)"ModifizierterExponent1000Iterationen.txt");
	for(unsigned i=1000; i<10000; i++) {
		x = kohonenMap2.getRandomX();
		kohonenMap2.kohonenLernregel(x);
	}
	kohonenMap2.ausgabe((char*)"ModifizierterExponent10000Iterationen.txt");
	for(unsigned i=10000; i<100000; i++) {
		x = kohonenMap2.getRandomX();
		kohonenMap2.kohonenLernregel(x);
	}
	kohonenMap2.ausgabe((char*)"ModifizierterExponent100000Iterationen.txt");
	for(unsigned i=100000; i<500000; i++) {
		x = kohonenMap2.getRandomX();
		kohonenMap2.kohonenLernregel(x);
	}
	kohonenMap2.ausgabe((char*)"ModifizierterExponent500000Iterationen.txt");
}
