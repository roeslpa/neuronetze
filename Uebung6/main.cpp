#include "matrix.h"
#include "KM.h"
// Paul Rösler und Daniel Teuchert
// Es ist zu beobachten, dass die Zentren der Neuronen mit höherer Iterationenanzahl immer genauer die obere Hälfte eines Torus' formen. Ab 100000 Iterationen kann man jedoch beobachten, dass diese Form bereits wieder zerstört wird und bei 500000 Iterationen ist inzwischen nur noch weniger als die Hälfte eines Torus' erkennbar. Zwischen den Ausgaben der Unterschiedlichen Exponenten gibt es keine allzu großen Unterschiede.  Mit dem modifizierten Exponenten scheinen die Zentren der Neuronen jedoch etwas schneller die Form eines Torus' zu bilden.
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
