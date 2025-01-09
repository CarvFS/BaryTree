#include <math.h>
#include <float.h>
#include <stdio.h>

#include "cuda_tcf_pp.h"


// CUDA Kernel
__global__ void TCF_PP_CUDA(
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    double target_xmin,    double target_ymin,    double target_zmin,
        
    double target_xdd,     double target_ydd,     double target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,

    int cluster_num_sources, int cluster_idx_start,
    double *source_x, double *source_y, double *source_z, double *source_q,

    double kap, double eta, double *potential)
{
    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;
    double kap_eta_2 = kap * eta / 2.0;

    // Calculate 3D grid indices
    int ix = blockIdx.x * blockDim.x + threadIdx.x + target_x_low_ind;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + target_y_low_ind;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + target_z_low_ind;

    if (ix > target_x_high_ind || iy > target_y_high_ind || iz > target_z_high_ind)
        return; // Out-of-bounds check

    // Compute target index in potential array
    int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;

    // Compute target coordinates
    double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
    double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
    double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;

    double temporary_potential = 0.0;

    // Loop over sources
    for (int j = 0; j < cluster_num_sources; j++) {
        int jj = cluster_idx_start + j;
        double dx = tx - source_x[jj];
        double dy = ty - source_y[jj];
        double dz = tz - source_z[jj];
        double r  = sqrt(dx * dx + dy * dy + dz * dz);

        if (r > DBL_MIN) {
            double kap_r = kap * r;
            double r_eta = r / eta;
            temporary_potential += source_q[jj] / r
                                 * (exp(-kap_r) * erfc(kap_eta_2 - r_eta)
                                 -  exp( kap_r) * erfc(kap_eta_2 + r_eta));
        }
    }

    // Atomic update to prevent race conditions
    // atmicAdd(&potential[ii], temporary_potential);
    // No atomicAdd needed, since each thread handles a unique ii
    potential[ii] = temporary_potential;
}


__host__
void K_CUDA_TCF_PP(
    int target_x_low_ind,  int target_x_high_ind,
    int target_y_low_ind,  int target_y_high_ind,
    int target_z_low_ind,  int target_z_high_ind,
    double target_xmin,    double target_ymin,    double target_zmin,
    double target_xdd,     double target_ydd,     double target_zdd,
    int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
    int cluster_num_sources, int cluster_idx_start,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double kap, double eta, double *potential, int gpu_async_stream_id)
{
    // kap: charge smearing parameter
    // eta: contribution to the inverse Debye length of ionic co-solvent 
    // All other parameters are the same as in the K_CUDA_Coulomb_PP function

    // Grid and block dimensions
    dim3 blockDim(8, 8, 8); // Adjust block size for your GPU's occupancy
    dim3 gridDim(
        (target_x_high_ind - target_x_low_ind + blockDim.x) / blockDim.x,
        (target_y_high_ind - target_y_low_ind + blockDim.y) / blockDim.y,
        (target_z_high_ind - target_z_low_ind + blockDim.z) / blockDim.z);

    // Launch CUDA Kernel
    TCF_PP_CUDA<<<gridDim, blockDim>>>(
        target_x_low_ind, target_x_high_ind,
        target_y_low_ind, target_y_high_ind,
        target_z_low_ind, target_z_high_ind,
        target_xmin, target_ymin, target_zmin,
        target_xdd, target_ydd, target_zdd,
        target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,
        cluster_num_sources, cluster_idx_start,
        source_x, source_y, source_z, source_q,
        kap, eta, potential);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    return;
}
