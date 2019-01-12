#include <math.h>


void coulombKernel( int numberOfTargets, int numberOfInterpolationPoints, int indexOfFirstTarget,
					double *targetX, double *targetY, double *targetZ, double *targetVal,
					double *interpolationX, double interpolationY, double *interpolationZ, double *interpolationVal,
					double *kernelMatrix){

	// indexOfFirstTarget = batch_ind[0] - 1 with current convention

	int i,j;
	double tx, ty, tz;
	double dx, dy, dz;

	for (i = 0; i < numberOfTargets; i++){
		tx = targetX[ indexOfFirstTarget + i];
		ty = targetY[ indexOfFirstTarget + i];
		tz = targetZ[ indexOfFirstTarget + i];

		for (j = 0; j < numberOfInterpolationPoints; j++){

			// Compute x, y, and z distances between target i and interpolation point j
			dx = targetX[ indexOfFirstTarget + i] - interpolationX[j];
			dy = targetY[ indexOfFirstTarget + i] - interpolationY[j];
			dz = targetZ[ indexOfFirstTarget + i] - interpolationZ[j];

			// Evaluate Kernel, store in kernelMatrix[i][j]
			kernelMatrix[i*numberOfInterpolationPoints + j] = interpolationVal[j] / sqrt( dx*dx + dy*dy + dz*dz);

		}

	}

	return;
}
