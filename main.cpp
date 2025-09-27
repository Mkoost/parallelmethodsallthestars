

#if 0
#include <omp.h>
#include <iostream>
#include<vector>
#include<cmath>
#include "labs.h"


double mulIKJ(
	const double * A,
	const double * B,
	double * C, int n)
{
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < n; ++i)
		for (int k = 0; k < n; ++k)
			for (int j = 0; j < n; ++j)
				C[i * n + j] += A[i * n + k] * B[k * n + j];

	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;
}


double mulIJKblock(
	const double* A,
	const double* B,
	double* C, int n, int b)
{
	auto start = std::chrono::steady_clock::now();

	std::vector<double> blA(b * b), blB(b * b), blC(b * b);


	for (int i = 0; i < n; i += b)
		for (int j = 0; j < n; j += b)
		{
			for (int p = 0; p < b * b; ++p)
				blC[p] = 0;

			for (int k = 0; k < n; k += b)
			{
				for (int p = 0; p < b; ++p)
					for (int q = 0; q < b; ++q)
					{
						blA[p * b + q] = A[(i + p) * n + (k + q)];
						blB[p * b + q] = B[(k + p) * n + (j + q)];
					}//for q //for p

				for (int p = 0; p < b; ++p)
					for (int r = 0; r < b; ++r)
						for (int q = 0; q < b; ++q)
							blC[p * b + q] += blA[p * b + r] * blB[r * b + q];
			}//for k

			for (int p = 0; p < b; ++p)
				for (int q = 0; q < b; ++q)
					C[(i + p) * n + (j + q)] = blC[p * b + q];
		}


	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;
}

int main() {
	
	int N = 256;
	int NxN = N * N;
	double *A, *B, *C;
	A = new double[NxN];
	B = new double[NxN];
	C = new double[NxN];

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
		{
			A[i * N + j] = cos(i + j);
			B[i * N + j] = sin(i - j);
		}
	for (int i = 0; i < NxN; ++i) C[i] = 0;

	std::cout << "strassen: " << strassenmul(A, B, C, N) << "s, C[n/2][n/2] = " << C[(N / 2) * (N / 2)] << "\n";

	for (int i = 0; i < NxN; ++i) C[i] = 0;
	std::cout << "IKJ: " << mulIKJ(A, B, C, N) << "s, C[n/2][n/2] = " << C[(N / 2) * (N / 2)] << "\n";

	

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}

#endif 

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

int ind(int i, int j, int n) { return i * n + j; };

double LU(std::vector<double>& A, int n) {
	auto start = std::chrono::steady_clock::now();
	std::vector<double> column(n);
	std::vector<double> row(n);

	for (int i = 0; i < n; ++i) {
		double aii = A[i * n + i];

		for (int j = i + 1; j < n; ++j)
			column[j - i - 1] = (A[j * n + i] /= aii);

		for (int j = i + 1; j < n; ++j)
			row[j - i - 1] = A[i * n + j];

		for (int j = 0; j < n - 1 - i; ++j){
			double c = column[j];
			for (int k = 0; k < n - 1 - i; ++k)
				A[(i + 1 + j) * n + (i + 1 + k)] -= c * row[k];
		}

	}
	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;

}

void LUmul(std::vector<double>& A, int n) {
	std::vector<double> row(n);

	for (int i = 0; i < n - 1; ++i) {
		for (int j = 0; j < n - i - 1; ++j) { 
			row[j] = A[(n - i - 1) * n + j]; 
			A[(n - i - 1) * n + j] = 0; 
		}

		for (int j = 0; j < n - i - 1; ++j) 
			for (int k = j; k < n; ++k)
				A[(n - i - 1) * n + k] += A[j * n + k] * row[j];
		
		
	}
}
double LUmxn(std::vector<double>& A, int m, int n){
	int N = std::min(m-1, n);
	auto start = std::chrono::steady_clock::now();
	std::vector<double> column(m);
	std::vector<double> row(n);

	for (int i = 0; i < N; ++i) {
		double aii = A[i * n + i];

		for (int j = i + 1; j < m; ++j)
			column[j - i - 1] = (A[j * n + i] /= aii);

		for (int j = i + 1; j < n; ++j)
			row[j - i - 1] = A[i * n + j];
		if(i < n)
			for (int j = 0; j < m - 1 - i; ++j) {
				double c = column[j];
				for (int k = 0; k < n - 1 - i; ++k)
					A[(i + 1 + j) * n + (i + 1 + k)] -= c * row[k];
			}

	}
	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;

}

double LUBlock(std::vector<double>& A, int n, int b) {
	auto start = std::chrono::steady_clock::now();
	std::vector<double> column(n * b);
	std::vector<double> row(n * b);




	
	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;
}

int main() {
	int n = 100;
	std::vector<double> A;
	A.resize(n * n);
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
		{
			A[i * n + j] = cos(i + j);
		}

	std::vector<double> Acopy(A);
	std::cout << "standart LU: " << LU(Acopy, n) << "s, ";
	LUmul(Acopy, n);
	
	double maxA = std::abs(A[0] - Acopy[0]), minA = std::abs(A[0] - Acopy[0]);

	for (int i = 0; i < n * n; ++i) {
		maxA = std::max(std::abs(A[i] - Acopy[i]), maxA);
		minA = std::min(std::abs(A[i] - Acopy[i]), minA);
	}
	std::cout << "max|A - A*| = " << maxA << ", min|A - A*| = " << minA << std::endl;
#pragma omp parallel
	{
		std::cout << "!";
	}
	

	return 0;
}

