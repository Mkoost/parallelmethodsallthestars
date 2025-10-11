#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cmath>

void fill_random(std::vector<double>& A, int n) {

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			A[i * n + j] = rand();

		}
	}
}

// KIND OF STANDART LU -------------------------------------------------------------------------------

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

		for (int j = 0; j < n - 1 - i; ++j) {
			for (int k = 0; k < n - 1 - i; ++k)
				A[(i + 1 + j) * n + (i + 1 + k)] -= column[j] * row[k];
		}

	}
	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;

}

double LUstandart(std::vector<double>& A, int n) {
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < n; ++i) {
		double aii = A[i * n + i];

		for (int j = i + 1; j < n; ++j)
			A[j * n + i] /= aii;

		for (int j = i + 1; j < n; ++j) {
			double c = A[j * n + i];
			for (int k = i + 1; k < n; ++k)
				A[j * n + k] -= A[j * n + i] * A[i * n + k];
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


double LUmxn(std::vector<double>& A, int m, int n) {
	auto start = std::chrono::steady_clock::now();
	int N = std::min(m, n);

	for (int i = 0; i < N; ++i) {
		double aii = A[i * n + i];

		for (int j = i + 1; j < m; ++j) {
			A[j * n + i] /= aii;
		}

		for (int j = i + 1; j < m; ++j) {
			for (int k = i + 1; k < n; ++k) {
				A[j * n + k] -= A[j * n + i] * A[i * n + k];
			}
		}
	}

	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	return elapsed.count();
}


void myprint(std::vector<double> A, int m, int n, int ii = 0, int jj = 0, int m_ = 0, int n_ = 0) {
	for (int i = 0; i < m + m_; ++i) {
		for (int j = 0; j < n + n_; ++j)
			std::cout << A[(i + ii) * n + (j + jj)] << " ";
		std::cout << "\n\r";
	}
	std::cout << "\n\r";

}

// LU BLOCK -------------------------------------------------------------------------------

void __LU_blocks_сopy(
	std::vector<double>& A,
	std::vector<double>& A22,
	std::vector<double>& A23,
	std::vector<double>& A32,
	int ii, int b, int n) {
	int N = n - ii - b;

	for (int i = 0; i < b; ++i)
		for (int j = 0; j < b; ++j)
			A22[i * b + j] = A[(ii + i) * n + j + ii];

	for (int i = 0; i < b; ++i)
		for (int j = 0; j < N; ++j)
			A23[i * N + j] = A[(ii + i) * n + j + ii + b];

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < b; ++j)
			A32[i * b + j] = A[(ii + i + b) * n + j + ii];
}

void __LU_blocks_paste(
	std::vector<double>& A,
	std::vector<double>& A22,
	std::vector<double>& A23,
	std::vector<double>& A32,
	int ii, int b, int n) {
	int N = n - ii - b;

	for (int i = 0; i < b; ++i)
		for (int j = 0; j < b; ++j)
			A[(ii + i) * n + j + ii] = A22[i * b + j];

	for (int i = 0; i < b; ++i)
		for (int j = 0; j < N; ++j)
			A[(ii + i) * n + j + ii + b] = A23[i * N + j];

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < b; ++j)
			A[(ii + i + b) * n + j + ii] = A32[i * b + j];
}

void __find_LU22(std::vector<double>& A22, int b) {
	for (int i_ = 0; i_ < b; ++i_) {
		double aii = A22[i_ * b + i_];

		for (int j_ = i_ + 1; j_ < b; ++j_)
			A22[j_ * b + i_] /= aii;

		for (int j_ = i_ + 1; j_ < b; ++j_) {
			double c = A22[j_ * b + i_];
			for (int k_ = i_ + 1; k_ < b; ++k_)
				A22[j_ * b + k_] -= c * A22[i_ * b + k_];
		}
	}

}

void __find_U23(std::vector<double>& A22, std::vector<double>& A23, int b, int n) {
	for (int i = 0; i < b; ++i)
		for (int j = i + 1; j < b; ++j) {
			double tmp = A22[j * b + i];
			for (int k = 0; k < n; ++k)
				A23[j * n + k] -= tmp * A23[i * n + k];
		}
}

void __find_L32(std::vector<double>& A22, std::vector<double>& A32, int b, int n) {
	for (int i = 0; i < b; ++i) {
		double tmp = A22[i * b + i];
		for (int k = 0; k < n; ++k)
			A32[k * b + i] /= tmp;  // A[k][i] /= U[i][i]

		for (int j = i + 1; j < b; ++j) {
			double tmp2 = A22[i * b + j];
			for (int k = 0; k < n; ++k)
				A32[k * b + j] -= tmp2 * A32[k * b + i];
		}
	}
}

double LUBlock(std::vector<double>& A, int n, int b) {
	auto start = std::chrono::steady_clock::now();
	std::vector<double> A22(b * b);
	std::vector<double> A32((n - b) * b);
	std::vector<double> A23((n - b) * b);

	//memcpy

	for (int ii = 0; ii < n; ii += b) {
		// copy blocks
		__LU_blocks_сopy(A, A22, A23, A32, ii, b, n);

		// find LU decomp for A22
		__find_LU22(A22, b);

		// Find U23
		__find_U23(A22, A23, b, n - b - ii);

		// Find L32
		__find_L32(A22, A32, b, n - b - ii);

		// Find new A33

		for (int j = ii + b; j < n; ++j)
			for (int k = ii + b; k < n; ++k) {
				for (int l = 0; l < b; ++l) {
					A[j * n + k] -= A32[(j - ii - b) * b + l] * A23[l * (n - ii - b) + k - ii - b];
				}
			}
		// paste blocks
		__LU_blocks_paste(A, A22, A23, A32, ii, b, n);
	}


	// myprint(A, n, n);  
	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;
}

// LU BLOCK PARALLEL -------------------------------------------------------------------------------




double LUBlockParallel(std::vector<double>& A, int n, int b, int nthreads = 4) {
	auto start = std::chrono::steady_clock::now();

	std::vector<double> A22(b * b);
	std::vector<double> A32((n - b) * b);
	std::vector<double> A23((n - b) * b);
	//memcpy
#pragma omp parallel num_threads(nthreads) shared(A22, A32, A23, A, n, b) 
	{
		for (int ii = 0; ii < n; ii += b) {
			int N = n - ii - b;
#pragma omp single
			{
				// copy blocks
				__LU_blocks_сopy(A, A22, A23, A32, ii, b, n);

				// find LU decomp for A22
				__find_LU22(A22, b);
			}

#pragma omp flush

			// Find U23

			for (int i = 0; i < b; ++i)
#pragma omp for schedule(static)  
				for (int j = i + 1; j < b; ++j) {
					double tmp = A22[j * b + i];
					for (int k = 0; k < N; ++k)
						A23[j * N + k] -= tmp * A23[i * N + k];
				}
			// Find L32

			for (int i = 0; i < b; ++i) {

				double tmp = A22[i * b + i];
#pragma omp for
				for (int k = 0; k < N; ++k)
					A32[k * b + i] /= tmp;

#pragma omp for schedule(static) 
				for (int j = i + 1; j < b; ++j) {
					double tmp2 = A22[i * b + j];
					for (int k = 0; k < N; ++k)
						A32[k * b + j] -= tmp2 * A32[k * b + i];
				}
			}

#pragma omp flush

			// Find new A33
#pragma omp for schedule(static)
			for (int j = ii + b; j < n; ++j)
				for (int k = ii + b; k < n; ++k) {
					for (int l = 0; l < b; ++l) {
						A[j * n + k] -= A32[(j - ii - b) * b + l] * A23[l * (n - ii - b) + k - ii - b];
					}
				}

			// paste blocks
#pragma omp single
			{
				__LU_blocks_paste(A, A22, A23, A32, ii, b, n);
			}
		}
	}

	// myprint(A, n, n);  
	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();

	//	std::cout << "copy: " << ops[0] << "s , LU22: " << ops[1] << "s , U23: " << ops[2] << "s , L32: " << ops[3] << "s, A'33: " << ops[5] << "s , paste: " << ops[4] << "s" << "\n\r";

	return time;
}

// LU PRALLEL -------------------------------------------------------------------------------

double LUParallel(std::vector<double>& A, int n, int nthreads = 4) {
	auto start = std::chrono::steady_clock::now();
	std::vector<double> column(n);
	std::vector<double> row(n);

	for (int i = 0; i < n; ++i) {
		double aii = A[i * n + i];

		for (int j = i + 1; j < n; ++j)
			column[j - i - 1] = (A[j * n + i] /= aii);

		for (int j = i + 1; j < n; ++j)
			row[j - i - 1] = A[i * n + j];

#pragma omp parallel for collapse(2) num_threads(nthreads) if(nthreads < (n - i))
		for (int j = 0; j < n - 1 - i; ++j) {
			//int c = column[j];
			for (int k = 0; k < n - 1 - i; ++k)
				A[(i + 1 + j) * n + (i + 1 + k)] -= column[j] * row[k];
		}

	}
	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;

}

// MAIN -------------------------------------------------------------------------------

int main(int argn, char** argc) {

	int n = 2048;
	int m = 10;
	int b1 = 32;
	int b = b1;
	int nthreads = 4;
	double tm1 = 0, tm2 = 0, maxA, minA;
	std::vector<double> A = { 1, 2, 3, 4, 5, 6,

							6, 1, 2, 3, 4, 5,
							5, 6, 1, 2, 3, 4,
							4, 5, 6, 1, 2, 3,
							3, 4, 5, 6, 1, 2,
							2, 3, 4, 5, 6, 1 };

	A.resize(n * n);
	fill_random(A, n);
	//myprint(A, n, n);

	std::vector<double> Acopy(A);


	//-------------------------------- LU


	Acopy = A;
	tm1 = LU(Acopy, n);
	//myprint(Acopy, n, n);
	std::cout << "standart LU: " << tm1 << "s\n\n\r";



	//-------------------------------- LU block
	Acopy = A;
	tm2 = LUBlock(Acopy, n, b);


	std::cout << "block LU: " << tm2 << "s, ";
	std::cout << "Speedup: " << tm1 / tm2 << std::endl << std::endl;

	//-------------------------------- LU block parallel
	Acopy = A;
	double tm3 = LUBlockParallel(Acopy, n, b1, nthreads);


	std::cout << "block LU parallel: " << tm3 << "s, ";
	std::cout << "Speedup: " << tm2 / tm3 << ", Efficiency: " << tm2 / tm3 / nthreads << std::endl;

	LUmul(Acopy, n);

	maxA = std::abs(A[0] - Acopy[0]), minA = std::abs(A[0] - Acopy[0]);

	for (int i = 0; i < n * n; ++i) {
		maxA = std::max(std::abs(A[i] - Acopy[i]), maxA);
		minA = std::min(std::abs(A[i] - Acopy[i]), minA);
	}

	std::cout << "max|A - A*| = " << maxA << ", min|A - A*| = " << minA << std::endl << std::endl;

	//-------------------------------- LU block parallel
	Acopy = A;
	double tm4 = LUBlockParallel(Acopy, n, 1, nthreads);


	std::cout << "LU parallel: " << tm4 << "s, ";
	std::cout << "Speedup: " << tm1 / tm4 << ", Efficiency: " << tm1 / tm4 / nthreads << std::endl;

	return 0;
}

