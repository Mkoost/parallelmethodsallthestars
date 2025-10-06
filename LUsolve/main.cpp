#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>


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

	for (int i = 0; i < n; i += b) {

		for (int j = i; j < n; ++j)
			for (int k = i; k < i + b; ++k)
				column[(j - i) * b + k - i] = A[j * n + k];
		
		//LU(column, b);
		LUmxn(column, n - i, b);


		for (int j = i; j < n; ++j)
			for (int k = i; k < i + b; ++k)
				A[j * n + k] = column[(j - i) * b + k - i];

		for (int j = 0; j < b; ++j) {
			double tmp = column[j * b + j];
			int c = (i + j) * n;

			for (int k = i + b; k < n ; ++k)
				A[c + k] /= tmp;

			for (int k = j + 1; k < b; ++k){
				double tmp2 = column[k * b + j];
				for (int l = i + b; l < n; ++l)
					A[(i + k) * n + l] -= tmp2 * A[c + l];
			}
		}

		for (int j = i; j < i + b; ++j)
			for (int k = i + b; k < n; ++k)
				row[(j - i) * n + k - i - b] = A[j * n + k];

		for (int j = i + b; j < n; ++j)
			for (int k = i + b; k < n; ++k) {
				double sum = 0;
				for (int l = 0; l < b; ++l) 
					sum += column[(j - i) * b + l] * row[l * n + k - i - b];
				
				A[j * n + k] -= sum;
			}



	}

	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;
}


int main() {
	int n = 4;
	int m = 10;
	int b = 2;
	std::vector<double> A = { 1, 0.540302, -0.416147, -0.989992, 0.540302, -0.416147, -0.989992, -0.653644, -0.416147, -0.989992, -0.653644, 0.283662, -0.989992, -0.653644, 0.283662, 0.96017 };
	A.resize(n * n);
	/*for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
		{
			A[i * n + j] = std::cos(i + j);
		}*/

	std::vector<double> Acopy(A);
	double tm = LUBlock(Acopy, n, b);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j)
			std::cout << Acopy[i * n + j] << " ";
		std::cout << "\n";
	}
	std::cout << "\n\n";


	std::cout << "standart LU: " << tm << "s, ";

	LUmul(Acopy, n);
	
	double maxA = std::abs(A[0] - Acopy[0]), minA = std::abs(A[0] - Acopy[0]);

	for (int i = 0; i < n * n; ++i) {
		maxA = std::max(std::abs(A[i] - Acopy[i]), maxA);
		minA = std::min(std::abs(A[i] - Acopy[i]), minA);
	}
	
	std::cout << "max|A - A*| = " << maxA << ", min|A - A*| = " << minA << std::endl;

	return 0;
}

