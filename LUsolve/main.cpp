#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>


// STANDART FUNCS
// -------------------------------------------------------------------------------
void fill_random(std::vector<double> &A, int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      if (i == j)
        A[i * n + j] = n;
      else
        A[i * n + j] = dis(gen);
  }
}

void fill_sin(std::vector<double> &A, int n) {

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      A[i * n + j] = std::sin(i + j);
}

void myprint(std::vector<double> A, int m, int n, int ii = 0, int jj = 0,
             int m_ = 0, int n_ = 0) {
  for (int i = 0; i < m + m_; ++i) {
    for (int j = 0; j < n + n_; ++j)
      std::cout << A[(i + ii) * n + (j + jj)] << " ";
    std::cout << "\n\r";
  }
  std::cout << "\n\r";
}

void LUmul(std::vector<double> &A, int n) {
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

// KIND OF STANDART LU
// -------------------------------------------------------------------------------

double LU(std::vector<double> &A, int n) {
  double start = omp_get_wtime();
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
  double finish = omp_get_wtime();
  double time = finish - start;
  return time;
}

double LUstandart(std::vector<double> &A, int n) {
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


double LUmxn(std::vector<double> &A, int m, int n) {
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

// LU BLOCK PARALLEL
// -------------------------------------------------------------------------------

double LUBlockParallel(std::vector<double> &A, int n, int b, int nthreads = 8) {
  double start = omp_get_wtime();

  std::vector<double> A22(b * b);
  std::vector<double> A32((n - b) * b);
  std::vector<double> A23((n - b) * b);
  // memcpy
#pragma omp parallel num_threads(nthreads) if (omp_get_max_threads() > 1)
  {
    for (int ii = 0; ii < n; ii += b) {
      int N = n - ii - b;

#pragma omp single nowait
      {
        // copy block A22
        for (int i = 0; i < b; ++i)
          for (int j = 0; j < b; ++j)
            A22[i * b + j] = A[(ii + i) * n + j + ii];

        // Find LU of A22
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

      // Copy A23
#pragma omp single nowait
      for (int i = 0; i < b; ++i)
        for (int j = 0; j < N; ++j)
          A23[i * N + j] = A[(ii + i) * n + j + ii + b];

      // Copy A32
#pragma omp single nowait
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < b; ++j)
          A32[i * b + j] = A[(ii + i + b) * n + j + ii];

#pragma omp barrier

      // Find U23
#pragma omp single nowait
      for (int i = 0; i < b; ++i)
        for (int j = i + 1; j < b; ++j) {
          double tmp = A22[j * b + i];
          for (int k = 0; k < N; ++k)
            A23[j * N + k] -= tmp * A23[i * N + k];
        }

      // Find L32
#pragma omp single nowait
      for (int i = 0; i < b; ++i) {
        double tmp = A22[i * b + i];
        for (int k = 0; k < N; ++k)
          A32[k * b + i] /= tmp;

        for (int j = i + 1; j < b; ++j) {
          double tmp2 = A22[i * b + j];
          for (int k = 0; k < N; ++k)
            A32[k * b + j] -= tmp2 * A32[k * b + i];
        }
      }

#pragma omp barrier

      // Find new A33
#pragma omp for
      for (int j = ii + b; j < n; ++j)
        for (int k = ii + b; k < n; ++k) {
          for (int l = 0; l < b; ++l) {
            A[j * n + k] -= A32[(j - ii - b) * b + l] * A23[l * N + k - ii - b];
          }
        }

      // paste blocks
#pragma omp single nowait
      for (int i = 0; i < b; ++i)
        for (int j = 0; j < b; ++j)
          A[(ii + i) * n + j + ii] = A22[i * b + j];

#pragma omp single nowait
      for (int i = 0; i < b; ++i)
        for (int j = 0; j < N; ++j)
          A[(ii + i) * n + j + ii + b] = A23[i * N + j];

#pragma omp single nowait
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < b; ++j)
          A[(ii + i + b) * n + j + ii] = A32[i * b + j];

#pragma omp barrier
    }
  }

  // myprint(A, n, n);
  double finish = omp_get_wtime();
  double time = finish - start;

  //	std::cout << "copy: " << ops[0] << "s , LU22: " << ops[1] << "s , U23: "
  //<< ops[2] << "s , L32: " << ops[3] << "s, A'33: " << ops[5] << "s , paste: "
  //<< ops[4] << "s" << "\n\r";

  return time;
}

// MAIN
// -------------------------------------------------------------------------------

int main(int argn, char **argc) {
  int n = 2 * 2048;
  int b = 32;

  int nthreads = omp_get_max_threads();

  double tm1 = 0, tm2 = 0, maxA, minA;

  std::vector<double> A;
  A.resize(n * n);

  fill_random(A, n);

  std::vector<double> Acopy(A);

  std::cout << "MY LU TEST \n\r"
            << "b = " << b << ", n = " << n
            << "\n\n\r" << "--------------------\n\n\r";


  //-------------------------------- LU block
  Acopy = A;

  tm1 = LUBlockParallel(Acopy, n, b, 1);

  std::cout << "Block LU: " << tm1 << "s, " << std::endl << std::endl;

  //-------------------------------- LU block parallel
  Acopy = A;

  tm2 = LUBlockParallel(Acopy, n, b, nthreads);

  std::cout << "Threads number: " << nthreads << std::endl;
  std::cout << "Block LU parallel: " << tm2 << "s, ";
  std::cout << "speedup: " << tm1 / tm2                                 \
            << ", efficiency: " << tm1 / tm2 / nthreads << std::endl;

  LUmul(Acopy, n);

  maxA = std::fabs(A[0] - Acopy[0]), minA = std::fabs(A[0] - Acopy[0]);

  for (int i = 0; i < n * n; ++i) {
    maxA = std::max(std::fabs(A[i] - Acopy[i]), maxA);
    minA = std::min(std::fabs(A[i] - Acopy[i]), minA);
  }

  std::cout << "max|A - A*| = " << maxA << ", min|A - A*| = " << minA
            << std::endl
            << std::endl;

  
  return 0;
}
