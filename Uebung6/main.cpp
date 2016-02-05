#include "matrix.h"
#include "KM.h"

int main(int argc, char** argv) {
	KM kohonenMap1 = KM(true);
	knn::matrix x;

	for(unsigned i=0; i<1000; i++) {
		x = kohonenMap1.getRandomX();
		kohonenMap1.kohonenLernregel(x);
	}
	for(unsigned i=1000; i<10000; i++) {
		x = kohonenMap1.getRandomX();
		kohonenMap1.kohonenLernregel(x);
	}
	for(unsigned i=10000; i<100000; i++) {
		x = kohonenMap1.getRandomX();
		kohonenMap1.kohonenLernregel(x);
	}
	for(unsigned i=100000; i<500000; i++) {
		x = kohonenMap1.getRandomX();
		kohonenMap1.kohonenLernregel(x);
	}

	KM kohonenMap2 = KM(false);

	for(unsigned i=0; i<1000; i++) {
		x = kohonenMap2.getRandomX();
		kohonenMap2.kohonenLernregel(x);
	}
	for(unsigned i=1000; i<10000; i++) {
		x = kohonenMap2.getRandomX();
		kohonenMap2.kohonenLernregel(x);
	}
	for(unsigned i=10000; i<100000; i++) {
		x = kohonenMap2.getRandomX();
		kohonenMap2.kohonenLernregel(x);
	}
	for(unsigned i=100000; i<500000; i++) {
		x = kohonenMap2.getRandomX();
		kohonenMap2.kohonenLernregel(x);
	}
}