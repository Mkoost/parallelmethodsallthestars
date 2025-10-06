#include "util.inl"

void strassenmul(double* A, double* B, double* C, double* tmp, int N);

constexpr int const STRASSEN_BLOCK_SIZE_SWITCH = 2;



void strassenmul2x2(double* A, double* B, double* C) {

	double tmp = (A[ind(0, 0, 2)] + A[ind(1, 1, 2)]) * (B[ind(0, 0, 2)] + B[ind(1, 1, 2)]); // D
	C[0] = tmp; C[3] = tmp;

	C[0] += (A[ind(0, 1, 2)] - A[ind(1, 1, 2)]) * (B[ind(1, 0, 2)] + B[ind(1, 1, 2)]); // D1

	C[3] += (A[ind(1, 0, 2)] - A[ind(0, 0, 2)]) * (B[ind(0, 0, 2)] + B[ind(0, 1, 2)]); // D2
	
	tmp = (A[ind(0, 0, 2)] + A[ind(0, 1, 2)]) * B[ind(1, 1, 2)]; // H1
	C[0] -= tmp; C[1] = tmp;

	tmp = (A[ind(1, 0, 2)] + A[ind(1, 1, 2)]) * B[ind(0, 0, 2)]; // H2
	C[2] = tmp; C[3] -= tmp;

	tmp = A[ind(1, 1, 2)] * (B[ind(1, 0, 2)] - B[ind(0, 0, 2)]); // V1
	C[0] += tmp; C[2] += tmp;

	tmp = A[ind(0, 0, 2)] * (B[ind(0, 1, 2)] - B[ind(1, 1, 2)]); // V2
	C[1] += tmp; C[3] += tmp;
}

double strassenmul(double* A, double* B, double* C, int n) {
	if (n == 2) { strassenmul2x2(A, B, C); return 0.0; }
	auto start = std::chrono::steady_clock::now();
	double* tmp = new double[n * n];
	strassenmul(A, B, C, tmp, n);
	delete[] tmp;
	auto finish = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed = finish - start;
	double time = elapsed.count();
	return time;

}

void strassenmul(double* A, double* B, double* C, double* tmp, int N) {
	if (N == STRASSEN_BLOCK_SIZE_SWITCH) { matrix_mul(A, B, C, N); return; }
	else if (N == 2) { strassenmul2x2(A, B, C); return; }

	int n = N >> 1;
	int nxn = n * n;

	// D = (A_11 + A_22) (B11 + B22)

	// A_11 + A_22

	copymat(tmp, A,
		0, 0, n,
		0, 0, n, N);

	plusmat(tmp, A,
		0, 0, n,
		n, n, n, N);
	
	
	// B_11 + B_22

	copymat(tmp + nxn, B,
			0, 0, n,
			0, 0, n, N);

	plusmat(tmp + nxn, B,
			0, 0, n,
			n, n, n, N);

	// (A_11 + A_22) (B11 + B22)
	strassenmul(tmp, tmp + nxn, tmp + 2 * nxn, tmp + 3 * nxn, n);
	

	// C[0] = D C[3] = D
	copymat(C, tmp + 2 * nxn,
		0, 0, N,
		0, 0, n, n);

	copymat(C, tmp + 2 * nxn,
			n, n, N,
			0, 0, n, n);
	
	// D_1 = (A_12 - A_22)(B_21 + B_22)

	// A_12 - A_22
	copymat(tmp, A,
		0, 0, n,
		0, n, n, N);

	minusmat(tmp, A,
		0, 0, n,
		n, n, n, N);

	// B_21 + B_22
	copymat(tmp + nxn, B,
		0, 0, n,
		n, 0, n, N);

	plusmat(tmp + nxn, B,
		0, 0, n,
		n, n, n, N);

	// (A_12 - A_22)(B_21 + B_22)
	strassenmul(tmp, tmp + nxn, tmp + 2 * nxn, tmp + 3 * nxn, n);

	// C[0] += D_1
	plusmat(C, tmp + 2 * nxn,
			0, 0, N,
			0, 0, n, n);


	// D_2 = (A_21 - A_11)(B_11 + B_12)
	
	// A_21 - A_11
	copymat(tmp, A,
		0, 0, n,
		n, 0, n, N);

	minusmat(tmp, A,
		0, 0, n,
		0, 0, n, N);

	// B_11 + B_12
	copymat(tmp + nxn, B,
		0, 0, n,
		0, 0, n, N);

	plusmat(tmp + nxn, B,
		0, 0, n,
		0, n, n, N);

	// (A_21 - A_11)(B_11 + B_12)
	strassenmul(tmp, tmp + nxn, tmp + 2 * nxn, tmp + 3 * nxn, n);

	// C[3] += D_2
	plusmat(C, tmp + 2 * nxn,
		n, n, N,
		0, 0, n, n);

	// H_1 = (A_11 + A_12) B_22
	
	//A_11 + A_12
	copymat(tmp, A,
		0, 0, n,
		0, 0, n, N);

	plusmat(tmp, A,
		0, 0, n,
		0, n, n, N);

	// B_22
	copymat(tmp + nxn, B,
		0, 0, n,
		n, n, n, N);

	// (A_11 + A_12) B_22
	strassenmul(tmp, tmp + nxn, tmp + 2 * nxn, tmp + 3 * nxn, n);

	// C[0] -= H_1
	minusmat(C, tmp + 2 * nxn,
		0, 0, N,
		0, 0, n, n);

	//C[1] = H_1
	copymat(C, tmp + 2 * nxn,
			0, n, N,
			0, 0, n, n);

	// H_2 = (A_21 + A_22) B_11

	// A_21 + A_22
	copymat(tmp, A,
		0, 0, n,
		n, 0, n, N);

	plusmat(tmp, A,
		0, 0, n,
		n, n, n, N);

	// B_11
	copymat(tmp + nxn, B,
		0, 0, n,
		0, 0, n, N);
	
	// (A_21 + A_22) B_11
	strassenmul(tmp, tmp + nxn, tmp + 2 * nxn, tmp + 3 * nxn, n);
	
	// C[2] = H2
	copymat(C, tmp + 2 * nxn,
			n, 0, N,
			0, 0, n, n);

	// C[3] -= H2
	minusmat(C, tmp + 2 * nxn,
			n, n, N,
			0, 0, n, n);

	// V_1 = A_22 (B_21 - B_11)
	
	// A_22
	copymat(tmp, A,
		0, 0, n,
		n, n, n, N);

	// B_21 - B_11
	copymat(tmp + nxn, B,
		0, 0, n,
		n, 0, n, N);

	minusmat(tmp + nxn, B,
		0, 0, n,
		0, 0, n, N);

	// A_22 (B_21 - B_11)

	strassenmul(tmp, tmp + nxn, tmp + 2 * nxn, tmp + 3 * nxn, n);

	// C[0] += V_1
	plusmat(C, tmp + 2 * nxn,
			0, 0, N,
			0, 0, n, n);

	// C[2] += V_1
	plusmat(C, tmp + 2 * nxn,
			n, 0, N,
			0, 0, n, n);

	// V_2 = A_11 (B_12 - B_22)
	
	// A_11
	copymat(tmp, A,
		0, 0, n,
		0, 0, n, N);

	//B_12 - B_22
	copymat(tmp + nxn, B,
		0, 0, n,
		0, n, n, N);

	minusmat(tmp + nxn, B,
		0, 0, n,
		n, n, n, N);

	// A_11(B_12 - B_22)
	strassenmul(tmp, tmp + nxn, tmp + 2 * nxn, tmp + 3 * nxn, n);

	// C[1] += V_2
	plusmat(C, tmp + 2 * nxn,
		0, n, N,
		0, 0, n, n);

	// C[3] += V_2
	plusmat(C, tmp + 2 * nxn,
		n, n, N,
		0, 0, n, n);

}

