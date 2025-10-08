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

 double LUstandart(std::vector<double>& A, int n) {
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < n; ++i) {
		double aii = A[i * n + i];

		for (int j = i + 1; j < n; ++j)
			A[j * n + i] /= aii;

			for (int j = i + 1; j < n; ++j){
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


void myprint(std::vector<double> A, int m, int n, int ii=0, int jj=0, int m_=0, int n_=0){
  for (int i = 0; i < m+m_; ++i){
    for (int j = 0; j < n+n_; ++j)
      std::cout << A[(i + ii) * n + (j + jj)] << " ";
    std::cout << "\n";
  }
    std::cout <<"\n";    
  
}

double LUBlock(std::vector<double>& A, int n, int b) {
	auto start = std::chrono::steady_clock::now();
  std::vector<double> A22(b * b);
  std::vector<double> A32((n-b) * b);
  std::vector<double> A23((n-b) * b);

	for (int ii = 0; ii < n; ii += b) {
    // find LU decomp for A22
 
    for (int i = ii; i < ii + b; ++i) {
		  int i_ = i - ii;
      double aii = A[i * n + i];
      A22[i_ * b + i_] = aii;

		  for (int j = i + 1; j < ii + b; ++j){
			  int j_ = j - ii;
        A22[j_ * b + i_] = (A[j * n + i] /= aii);
      }
      
     for (int j = i + 1; j < ii + b; ++j){
			  int j_ = j - ii;
        A22[i_ * b + j_] = A[i * n + j];
      }

			for (int j = i + 1; j < ii + b; ++j){
        int j_ = j - ii;
        double c = A[j * n + i];
			  for (int k = i + 1; k < ii + b; ++k){
          int k_ = k - ii;
				  A22[j_ * b + k_] = (A[j * n + k] -= A22[j_ * b + i_] * A22[i_ * b + k_]);
        }
		  }
	  }
    
  // Find U23
  for (int i = 0; i < n - ii - b; ++i) A23[i] = A[(ii) * n + i + ii + b] ;


  for (int i = 0; i < b; ++i)
    for(int j = i + 1; j < b; ++j){
      double tmp = A22[j * b + i];
      for(int k = ii + b; k < n; ++k)
        A23[j * (n - ii - b) + (k - ii - b)] = (A[(ii + j) * n + k] -= tmp * A[(ii + i) * n + k]);
    }
	// Find L32

  for (int i = 0; i < b; ++i){
    double tmp = A22[i * b + i];
    for (int k = 0; k < n - ii - b; ++k) 
       A32[k * b + i] = (A[(k + ii + b) * n + i + ii] /= tmp);  // A[k][i] /= U[i][i]
    
    for(int j = i + 1; j < b; ++j){
      double tmp2 = A22[i * b + j];
      for(int k = ii + b; k < n; ++k)
         A[k  * n + j + ii] -= tmp2 * A[k * n + i + ii];
    }
  }

  //myprint(A32, b, n-ii-b); 
  // Find new A33
    

  for (int j = ii + b; j < n; ++j)
			for (int k = ii + b; k < n; ++k) {
			  for(int l = 0; l < b; ++l) 	
				  A[j * n + k] -= A32[(j - ii - b) * b + l] * A23[l * (n - ii - b) + k - ii - b];
			}

  
  }

  // myprint(A, n, n);  
	auto finish = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;
}

int main(int argn, char** argc) {
	int n = 2000;
	int m = 10;
	int b = 8;
  double tm1 = 0, tm2 = 0, maxA, minA;
	std::vector<double> A = { 1, 2, 3, 4, 5, 6,

                            6, 1, 2, 3, 4, 5,
                            5, 6, 1, 2, 3, 4,
                            4, 5, 6, 1, 2, 3,
                            3, 4, 5, 6, 1, 2,
                            2, 3, 4, 5, 6, 1};

  A.resize(n * n);
  fill_random(A, n);
  //myprint(A, n, n);

  std::vector<double> Acopy(A);
  

//-------------------------------- LU
  

  Acopy = A;
	tm1 = LU(Acopy, n);
  //myprint(Acopy, n, n);
  std::cout << "standart LU: " << tm1 << "s, ";

	LUmul(Acopy, n);
	
	maxA = std::abs(A[0] - Acopy[0]), minA = std::abs(A[0] - Acopy[0]);

	for (int i = 0; i < n * n; ++i) {
		maxA = std::max(std::abs(A[i] - Acopy[i]), maxA);
		minA = std::min(std::abs(A[i] - Acopy[i]), minA);
  }
	std::cout << "max|A - A*| = " << maxA << ", min|A - A*| = " << minA << std::endl;


	//-------------------------------- LU block
  Acopy = A;
   tm2 = LUBlock(Acopy, n, b);
	

	std::cout << "block LU: " << tm2 << "s, ";

	LUmul(Acopy, n);
	
	maxA = std::abs(A[0] - Acopy[0]), minA = std::abs(A[0] - Acopy[0]);

	for (int i = 0; i < n * n; ++i) {
		maxA = std::max(std::abs(A[i] - Acopy[i]), maxA);
		minA = std::min(std::abs(A[i] - Acopy[i]), minA);
	}
	
	std::cout << "max|A - A*| = " << maxA << ", min|A - A*| = " << minA << std::endl;
  std::cout << "Speedup: " << tm1/tm2 << std::endl;
	return 0;
}

