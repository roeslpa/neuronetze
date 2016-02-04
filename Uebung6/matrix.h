#ifndef KNN1_MATRIX_H
#define KNN1_MATRIX_H
#define _CRT_SECURE_NO_WARNING 1
#define _SCL_SECURE_NO_WARNINGS 1
#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <functional>
#include <time.h>
#include <set>
#include <sstream>


using namespace std;


namespace knn {
	// initializes the random function with the current time; only possible on Linux Systems
	void initRNG(unsigned valueA){
		// check if the value is greater than 0
		if (valueA){
			// take the given value to init the RNG
			srand(valueA);
		}
		else{
			// take the current time to init the RNG
			srand((unsigned)time(NULL));
		}
	}

	// global random function; generates quasi-random values
	double randomFromInterval(double lowerBoundA, double upperBoundA) {
		assert(lowerBoundA <= upperBoundA);
		// get a random value
		double valueL = rand();
		// generate random value between 0 and (upperBoundA - lowerBoundA)
		valueL *= (upperBoundA - lowerBoundA) / RAND_MAX;

		//return shifted random value
		return valueL + lowerBoundA;
	}

	void init(unsigned valueA = 0) {
		knn::initRNG(valueA);
	}

	// matrix class for rectangular matrices
	class matrix {
	public:
		// creates matrix with given number of columns and rows
		matrix(unsigned nrRowsA = 0, unsigned nrColsA = 0, double initValA = 0);
		// copy constructor; copies size and matrix entries
		matrix(const matrix & srcA);
		// default destructor; frees the memory
		~matrix();

		// assignement operator, copies a matrix
		matrix & operator = (const matrix & srcA);

		// returnes the number of rows of the matrix
		unsigned rowSize() const;

		// returnes the number of columns of the matrix
		unsigned colSize() const;

		// return or update the element of the matrix at position [xA][yA]
		double& operator()(unsigned rowA, unsigned colA);

		// matrix addition: A = B + C
		matrix operator +(const matrix & mA) const;
		matrix & operator +=(const matrix & mA);

		// matrix multiplication: A = B * C
		matrix operator *(const matrix & mA) const;
		matrix & operator *=(const matrix & mA);

		// scalar multiplication: A = b * C
		matrix operator *(const double bA) const;
		matrix & operator *=(const double bA);

		// inverts this matrix using singular value decomposition
		// all eigenvalues < epsilonA are set to 0
		void invert(double epsilonA = 1e-12);

		// fills the matrix with quasi-random values between lowerBoundA and upperBoundA
		void fillRandom(double lowerBoundA, double upperBoundA);

	protected:
		// number of rows and columns
		unsigned xSizeE, ySizeE;

		// data array
		double* dataE;

	private:
		/*
		* WARNING!!! ACHTUNG!!!!
		* Using private Functions is strictly forbidden = cheating!
		* Private Funktionen zu verwenden ist ausdrücklich verboten und wird als Betrugsversuch gewertet!
		*/
		const double item(unsigned xA, unsigned yA) const;
		void item(unsigned xA, unsigned yA, double valueA);
		void init(double valueA);
		double* operator [](unsigned xA);
		const double* const operator[](unsigned xA) const;
		void computeSVD(matrix & u, matrix & v, double* w);
		/*
		* WARNING END
		*/
	};

	// overload ostream operator << for class matrix
	std::ostream& operator <<(std::ostream& streamA, const matrix & matrixA);


	// --------------------------------------
	//          Implementations
	// --------------------------------------
	// construct a new matrix with the given size
	inline matrix::matrix(unsigned nrRowsA, unsigned nrColsA, double initValA)
		:xSizeE(nrRowsA), ySizeE(nrColsA){
		// generate the new data array
		if (xSizeE*ySizeE > 0){
			dataE = new double[xSizeE * ySizeE];
			init(initValA);
		}
		else{
			dataE = 0;
		}
	}

	// copy constructor
	inline matrix::matrix(const matrix & srcA)
		:xSizeE(srcA.xSizeE), ySizeE(srcA.ySizeE){
		//generate new data array
		if (xSizeE*ySizeE > 0)
			dataE = new double[xSizeE * ySizeE];
		else
			dataE = 0;
		// copy data from the source matrix
		for (unsigned i = 0; i < xSizeE * ySizeE; ++i){
			dataE[i] = srcA.dataE[i];
		}
	}

	// destructor
	inline matrix::~matrix(){
		// delete data
		if (dataE)
			delete[] dataE;
	}

	// initializes all entries to the same value
	inline void matrix::init(double valueA){
		// init the data array with the given value
		std::fill(dataE, dataE + xSizeE * ySizeE, valueA);
	}

	// return the number of rows
	inline unsigned matrix::rowSize() const{
		return xSizeE;
	}

	// return the number of columns
	inline unsigned matrix::colSize() const{
		return ySizeE;
	}

	// assignment operator
	inline matrix &matrix::operator =(const matrix & srcA){
		// check if we assign me to me.
		if (&srcA == this)
			return *this;

		// assign new size
		xSizeE = srcA.xSizeE;
		ySizeE = srcA.ySizeE;

		// delete old data array
		if (dataE)
			delete[] dataE;
		// generate new data array
		if (xSizeE*ySizeE > 0)
			dataE = new double[xSizeE * ySizeE];
		else
			dataE = 0;
		// copy data from the source matrix
		std::copy(srcA.dataE, srcA.dataE + xSizeE * ySizeE, dataE);

		// return *this as usual in assignment operators
		return *this;
	}

	double& matrix::operator()(unsigned rowA, unsigned colA){
		// check, whether row and column is ok
		assert(rowA > 0 && colA > 0);
		rowA -= 1;
		colA -= 1;
		assert(rowA < xSizeE && colA < ySizeE);
		// return the desired element
		return dataE[rowA * ySizeE + colA];
	}


	// return matrix data at [xA][yA]
	inline const double matrix::item(unsigned xA, unsigned yA) const{
		// check, whether row and column is ok
		assert(xA < xSizeE && yA < ySizeE);
		// return the desired element
		return dataE[xA * ySizeE + yA];
	}

	// set  matrix data at [xA][yA] to valueA
	inline void matrix::item(unsigned xA, unsigned yA, double valueA) {
		// check, whether row and column is ok
		assert(xA < xSizeE && yA < ySizeE);
		// set the element
		dataE[xA * ySizeE + yA] = valueA;
	}


	// overloaded version of ostream::operator << for matrix class
	inline std::ostream& operator <<(std::ostream& streamA, matrix & matrixA){
		// write all rows
		for (unsigned xL = 0; xL < matrixA.rowSize(); xL++){
			// write all entries of the row
			for (unsigned yL = 0; yL < matrixA.colSize(); ++yL){
				streamA << matrixA(xL + 1, yL + 1) << '\t';
			}
			streamA << '\n';
		}
		// return the new position in the stream, as usual for overloads of ostream::operator <<
		return streamA;
	}

	// overload of operator + for two matrices
	inline matrix matrix::operator+(const matrix & mA) const{
		assert(mA.xSizeE == xSizeE && mA.ySizeE == ySizeE);
		// generate matrix with needed size
		matrix resL(xSizeE, ySizeE);
		// do the elementwise addition
		std::transform(dataE, dataE + xSizeE * ySizeE, mA.dataE, resL.dataE, std::plus<double>());
		// return the summed matrices
		return resL;
	}
	inline matrix & matrix::operator+=(const matrix & mA) {
		*this = mA + *this;
		return *this;
	}

	// overload of operator * for matrices; implements matrix-multiplication; doesn't change *this matrix
	inline matrix matrix::operator*(const matrix& mA) const{
		assert(mA.xSizeE == ySizeE);
		// generate Matrix with the needed size
		matrix resL(xSizeE, mA.ySizeE);
		// fill the result matrix element by element
		for (unsigned rowL = 0; rowL < resL.xSizeE; ++rowL){
			for (unsigned colL = 0; colL < resL.ySizeE; ++colL){
				// init counter
				double resValueL = 0.;
				// iterate over the current row/column of the source matrices
				for (unsigned iL = 0; iL < ySizeE; ++iL){
					// add up the multiplied source matrices
					resValueL += item(rowL, iL) * mA.item(iL, colL);
				}
				// set the value
				resL.item(rowL, colL, resValueL);
			}
		}
		// return the resulting Matrix
		return resL;
	}
	inline matrix & matrix::operator*=(const matrix & mA) {
		*this = *this * mA;
		return *this;
	}

	// overload of operator * for matrices; implements scalar-matrix-multiplication; doesn't change *this matrix
	inline matrix matrix::operator*(const double bA) const{
		// generate matrix with the needed size
		matrix resL(xSizeE, ySizeE);
		// fill the result matrix element by element
		for (unsigned rowL = 0; rowL < resL.xSizeE; ++rowL){
			for (unsigned colL = 0; colL < resL.ySizeE; ++colL){
				// set the value
				resL.item(rowL, colL, item(rowL, colL)*bA);
			}
		}
		// return the resulting matrix
		return resL;
	}
	inline matrix & matrix::operator*=(const double bA){
		*this = *this * bA;
		return *this;
	}

	// fills the matrix randomly with uniformly distributed values between lowerBoundA and upperBoundA
	inline void matrix::fillRandom(double lowerBoundA, double upperBoundA){
		// iterate over all elements
		for (unsigned iL = 0; iL < xSizeE * ySizeE; ++iL){
			// fill in a random element
			dataE[iL] = randomFromInterval(lowerBoundA, upperBoundA);
		}
	}


	// invert this matrix using singular value decomposition
	inline void matrix::invert(double epsilonA){
		// copy this matrix to U
		matrix U(*this);
		matrix V(ySizeE, ySizeE);
		double* W = new double[ySizeE];
		// fill U, V and W with the singular value decomposition matrices
		computeSVD(U, V, W);

		// delete all singular values (i.e. the eigenvalues less than epsilonA)
		for (unsigned i = 0; i < ySizeE; ++i){
			if (W[i] < epsilonA){
				W[i] = 0.;
			}
			else{
				W[i] = 1. / W[i];
			}
		}

		// fill this matrix with the new values
		unsigned t = xSizeE;
		xSizeE = ySizeE;
		ySizeE = t;
		// no need to get new memory, because the size (xSizeE * ySizeE) haven't changed

		// add up A = V^T W U
		for (unsigned i = 0; i < xSizeE; ++i){
			for (unsigned j = 0; j < ySizeE; ++j){
				double sumL = 0.;
				for (unsigned k = 0; k < xSizeE; ++k){
					sumL += V[i][k] * W[k] * U[j][k];
				}
				item(i, j, sumL);
			}
		}

		delete[] W;
		// done! :)
	}



	//////////////////////////////////////////////////////
	// SVD stuff copied from Shark
	// no more good documentation from here on
	static double SIGN(double a, double b){
		return b > 0 ? fabs(a) : -fabs(a);
	}

	// operator [] for usual [row][col]-access to matrix data, inaccassable from outside
	inline double*matrix::operator [](unsigned xA){
		assert(xA < xSizeE);
		return &dataE[xA * ySizeE];
	}

	// operator [] for usual [row][col]-read-only-access to matrix data
	inline const double* const matrix::operator[](unsigned xA) const{
		assert(xA < xSizeE);
		return &dataE[xA * ySizeE];
	}


	// computes the singular value decomposition
	inline void matrix::computeSVD(matrix & u, matrix & v, double* w){

		double* rv1 = new double[ySizeE];
		int m = xSizeE, n = ySizeE;

		int flag;
		int i, its, j, jj, k, l, nm(0);
		double anorm, c, f, g, h, p, s, scale, x, y, z;

		// householder reduction to bidiagonal form
		g = scale = anorm = 0.0;

		for (i = 0; i < n; i++){
			l = i + 1;
			rv1[i] = scale * g;
			g = s = scale = 0.0;

			if (i < m){
				for (k = i; k < m; k++){
					scale += fabs(u[k][i]);
				}

				if (scale != 0.0){
					for (k = i; k < m; k++){
						u[k][i] /= scale;
						s += u[k][i] * u[k][i];
					}

					f = u[i][i];
					g = -SIGN(sqrt(s), f);
					h = f * g - s;
					u[i][i] = f - g;

					for (j = l; j < n; j++){
						s = 0.0;
						for (k = i; k < m; k++){
							s += u[k][i] * u[k][j];
						}

						f = s / h;
						for (k = i; k < m; k++){
							u[k][j] += f * u[k][i];
						}
					}

					for (k = i; k < m; k++){
						u[k][i] *= scale;
					}
				}
			}

			w[i] = scale * g;
			g = s = scale = 0.0;

			if (i < m && i != n - 1){
				for (k = l; k < n; k++){
					scale += fabs(u[i][k]);
				}

				if (scale != 0.0){
					for (k = l; k < n; k++){
						u[i][k] /= scale;
						s += u[i][k] * u[i][k];
					}

					f = u[i][l];
					g = -SIGN(sqrt(s), f);
					h = f * g - s;
					u[i][l] = f - g;

					for (k = l; k < n; k++){
						rv1[k] = u[i][k] / h;
					}

					for (j = l; j < m; j++){
						s = 0.0;
						for (k = l; k < n; k++){
							s += u[j][k] * u[i][k];
						}

						for (k = l; k < n; k++){
							u[j][k] += s * rv1[k];
						}
					}

					for (k = l; k < n; k++){
						u[i][k] *= scale;
					}
				}
			}

			anorm = std::max(anorm, fabs(w[i]) + fabs(rv1[i]));
		}

		// accumulation of right-hand transformations
		for (l = i = n; i--; l--){
			if (l < n){
				if (g != 0.0){
					for (j = l; j < n; j++){
						// double division avoids possible underflow
						v[j][i] = (u[i][j] / u[i][l]) / g;
					}

					for (j = l; j < n; j++){
						s = 0.0;
						for (k = l; k < n; k++){
							s += u[i][k] * v[k][j];
						}

						for (k = l; k < n; k++){
							v[k][j] += s * v[k][i];
						}
					}
				}

				for (j = l; j < n; j++){
					v[i][j] = v[j][i] = 0.0;
				}
			}

			v[i][i] = 1.0;
			g = rv1[i];
		}

		// accumulation of left-hand transformations
		for (l = i = std::min(m, n); i--; l--){
			g = w[i];

			for (j = l; j < n; j++){
				u[i][j] = 0.0;
			}

			if (g != 0.0){
				g = 1.0 / g;

				for (j = l; j < n; j++){
					s = 0.0;
					for (k = l; k < m; k++){
						s += u[k][i] * u[k][j];
					}

					// double division avoids possible underflow
					f = (s / u[i][i]) * g;

					for (k = i; k < m; k++){
						u[k][j] += f * u[k][i];
					}
				}

				for (j = i; j < m; j++){
					u[j][i] *= g;
				}
			}
			else{
				for (j = i; j < m; j++){
					u[j][i] = 0.0;
				}
			}

			u[i][i]++;
		}

		// diagonalization of the bidiagonal form
		for (k = n; k--;){
			for (its = 1; its <= 30; its++){
				flag = 1;

				// test for splitting
				for (l = k + 1; l--;){
					// rv1 [0] is always zero, so there is no exit
					nm = l - 1;

					if (fabs(rv1[l]) + anorm == anorm){
						flag = 0;
						break;
					}

					if (fabs(w[nm]) + anorm == anorm){
						break;
					}
				}

				if (flag){
					// cancellation of rv1 [l] if l greater than 0
					c = 0.0;
					s = 1.0;

					for (i = l; i <= k; i++){
						f = s * rv1[i];
						rv1[i] *= c;

						if (fabs(f) + anorm == anorm){
							break;
						}

						g = w[i];
						h = hypot(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = -f * h;

						for (j = 0; j < m; j++){
							y = u[j][nm];
							z = u[j][i];
							u[j][nm] = y * c + z * s;
							u[j][i] = z * c - y * s;
						}
					}
				}

				// test for convergence
				z = w[k];

				if (l == k){
					if (z < 0.0){
						w[k] = -z;
						for (j = 0; j < n; j++){
							v[j][k] = -v[j][k];
						}
					}
					break;
				}

				if (its == 30){
					throw k;
				}

				// shift from bottom 2 by 2 minor
				x = w[l];
				nm = k - 1;
				y = w[nm];
				g = rv1[nm];
				h = rv1[k];
				f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
				g = hypot(f, 1.0);
				f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

				// next qr transformation
				c = s = 1.0;

				for (j = l; j < k; j++){
					i = j + 1;
					g = rv1[i];
					y = w[i];
					h = s * g;
					g *= c;
					z = hypot(f, h);
					rv1[j] = z;
					c = f / z;
					s = h / z;
					f = x * c + g * s;
					g = g * c - x * s;
					h = y * s;
					y *= c;

					for (jj = 0; jj < n; jj++)
					{
						x = v[jj][j];
						z = v[jj][i];
						v[jj][j] = x * c + z * s;
						v[jj][i] = z * c - x * s;
					}

					z = hypot(f, h);
					w[j] = z;

					// rotation can be arbitrary if z is zero
					if (z != 0.0){
						z = 1.0 / z;
						c = f * z;
						s = h * z;
					}

					f = c * g + s * y;
					x = c * y - s * g;

					for (jj = 0; jj < m; jj++){
						y = u[jj][j];
						z = u[jj][i];
						u[jj][j] = y * c + z * s;
						u[jj][i] = z * c - y * s;
					}
				}

				rv1[l] = 0.0;
				rv1[k] = f;
				w[k] = x;
			}
		}

		//////////////////////////////////////////////
		//sort the eigenvalues in descending order
		for (i = 0; i < n - 1; i++){
			p = w[k = i];

			for (j = i + 1; j < n; j++){
				if (w[j] >= p){
					p = w[k = j];
				}
			}

			if (k != i){
				w[k] = w[i];
				w[i] = p;

				for (j = 0; j < n; j++){
					p = v[j][i];
					v[j][i] = v[j][k];
					v[j][k] = p;
				}

				for (j = 0; j < m; j++){
					p = u[j][i];
					u[j][i] = u[j][k];
					u[j][k] = p;
				}
			}
		}

		//free the acquired memory
		delete[] rv1;
	}
}

#endif //KNN1_MATRIX_H
