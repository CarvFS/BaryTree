#include <math.h>
#include <float.h>
#include <stdio.h>

#include "cuda_coulomb_pp.h"


__global__ 
void Coulomb_PP_CUDA(
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    double target_xmin,    double target_ymin,    double target_zmin,

    double target_xdd,     double target_ydd,     double target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,

    int cluster_num_sources, int cluster_idx_start,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *potential)
{
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;

    // Compute 3D thread and block indices
    int ix = blockIdx.x * blockDim.x + threadIdx.x + target_x_low_ind;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + target_y_low_ind;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + target_z_low_ind;

    if (ix > target_x_high_ind || iy > target_y_high_ind || iz > target_z_high_ind) return;

    int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;
    double temporary_potential = 0.0;

    double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
    double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
    double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;

    for (int j = 0; j < cluster_num_sources; j++) {
        int jj = cluster_idx_start + j;

        double dx = tx - source_x[jj];
        double dy = ty - source_y[jj];
        double dz = tz - source_z[jj];
        double r  = sqrt(dx*dx + dy*dy + dz*dz);

        if (r > DBL_MIN) {
            temporary_potential += source_q[jj] / r;
        }
    }

    // Use atomicAdd to safely update the shared potential array
    // atomicAdd(&potential[ii], temporary_potential);
    // No atomicAdd needed, since each thread handles a unique ii
    potential[ii] = temporary_potential;
}


__host__
void K_CUDA_Coulomb_PP(
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    double target_xmin,    double target_ymin,    double target_zmin,
    double target_xdd,     double target_ydd,     double target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    int cluster_num_sources, int cluster_idx_start,
    double *source_x, double *source_y, double *source_z, double *source_q,
    struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{
    // Grid and block dimensions
    dim3 blockDim(8, 8, 8); // Adjust block size for your GPU's occupancy
    dim3 gridDim(
        (target_x_high_ind - target_x_low_ind + blockDim.x) / blockDim.x,
        (target_y_high_ind - target_y_low_ind + blockDim.y) / blockDim.y,
        (target_z_high_ind - target_z_low_ind + blockDim.z) / blockDim.z);

    // Launch the CUDA kernel asynchronously
    Coulomb_PP_CUDA<<<gridDim, blockDim>>>(
        target_x_low_ind, target_x_high_ind,
        target_y_low_ind, target_y_high_ind,
        target_z_low_ind, target_z_high_ind,
        target_xmin, target_ymin, target_zmin,

        target_xdd, target_ydd, target_zdd,
        target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,

        cluster_num_sources, cluster_idx_start,
        source_x, source_y, source_z, source_q, potential);

    // Synchronize the CUDA stream to ensure execution completion
    cudaDeviceSynchronize();

    return;
}
