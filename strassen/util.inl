#ifndef MYUTIL__
#define MYUTIL__
constexpr int ind(const int& i, const int& j, const int& n) { return j + i * n; }

void copymat(double* A, double* B,
	const int i1, const int j1, const int N1,
	const int i2, const int j2, const int n2, const int N2) {

	for (int i = 0; i < n2; ++i)
		for (int j = 0; j < n2; ++j)
			A[ind(i + i1, j + j1, N1)] = B[ind(i + i2, j + j2, N2)];
}

void copymat(double* A, double* B,
	const int i1, const int j1, const int N1,
	const int i2, const int j2, const int a2, const int b2,  const int N2) {

	for (int i = 0; i < a2; ++i)
		for (int j = 0; j < b2; ++j)
			A[ind(i + i1, j + j1, N1)] = B[ind(i + i2, j + j2, N2)];
}

void plusmat(double* A, double* B,
	const int i1, const int j1, const int N1,
	const int i2, const int j2, const int n2, const int N2) {

	for (int i = 0; i < n2; ++i)
		for (int j = 0; j < n2; ++j)
			A[ind(i + i1, j + j1, N1)] += B[ind(i + i2, j + j2, N2)];
}

void minusmat(double* A, double* B,
	const int i1, const int j1, const int N1,
	const int i2, const int j2, const int n2, const int N2) {

	for (int i = 0; i < n2; ++i)
		for (int j = 0; j < n2; ++j)
			A[ind(i + i1, j + j1, N1)] -= B[ind(i + i2, j + j2, N2)];
}

void matrix_mul(double* A, double* B, double* C, int n) {
	for (int i = 0; i < n; ++i)
	{
		double* c = C + i * n;
		for (int j = 0; j < n; ++j)
			c[j] = 0;
		for (int k = 0; k < n; ++k)
		{
			const double* b = B + k * n;
			float a = A[i * n + k];
			for (int j = 0; j < n; ++j)
				c[j] += a * b[j];
		}
	}

}
#endif