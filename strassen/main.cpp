#include <omp.h>
#include <iostream>
#include<vector>
#include<cmath>
#include "util.inl"
#include "strassen.inl"


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

