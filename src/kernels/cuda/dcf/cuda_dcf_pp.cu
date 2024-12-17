#include <math.h>
#include <float.h>
#include <stdio.h>

#include "cuda_dcf_pp.h"

// OLD VERSION WITH DIFFERENT RESULTS: KEEPING JUST TO SEE THE SOURCE OF DIFFERENCE

// __global__ void DCF_PP_CUDA(
//     int target_x_low_ind,  int target_x_high_ind,
//     int target_y_low_ind,  int target_y_high_ind,
//     int target_z_low_ind,  int target_z_high_ind,
//     double target_xmin,    double target_ymin,    double target_zmin,

//     double target_xdd,     double target_ydd,     double target_zdd,
//     int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,

//     int cluster_num_sources, int cluster_idx_start,
//     double *source_x, double *source_y, double *source_z, double *source_q,

//     double eta, double *potential)
// {
//     int target_yz_dim = target_y_dim_glob * target_z_dim_glob;

//     // Calculate 3D grid indices
//     int ix = blockIdx.x * blockDim.x + threadIdx.x + target_x_low_ind;
//     int iy = blockIdx.y * blockDim.y + threadIdx.y + target_y_low_ind;
//     int iz = blockIdx.z * blockDim.z + threadIdx.z + target_z_low_ind;

//     if (ix > target_x_high_ind || iy > target_y_high_ind || iz > target_z_high_ind)
//         return; // Out-of-bounds check

//     // Compute target index in potential array
//     int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;

//     // Compute target coordinates
//     double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
//     double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
//     double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;

//     double temporary_potential = 0.0;

//     // Loop over sources
//     for (int j = 0; j < cluster_num_sources; j++) {
//         int jj = cluster_idx_start + j;
//         double dx = tx - source_x[jj];
//         double dy = ty - source_y[jj];
//         double dz = tz - source_z[jj];
//         double r  = sqrt(dx * dx + dy * dy + dz * dz);

//         if (r > DBL_MIN) {
//             temporary_potential += source_q[jj] * erf(r / eta) / r;
//         }
//     }

//     // Atomic update to prevent race conditions
//     atomicAdd(&potential[ii], temporary_potential);
// }


// __host__
// void K_CUDA_DCF_PP(
//     int target_x_low_ind,  int target_x_high_ind,
//     int target_y_low_ind,  int target_y_high_ind,
//     int target_z_low_ind,  int target_z_high_ind,
//     double target_xmin,    double target_ymin,    double target_zmin,
//     double target_xdd,     double target_ydd,     double target_zdd,
//     int target_x_dim_glob, int target_y_dim_glob, int target_z_dim_glob,
//     int cluster_num_sources, int cluster_idx_start,
//     double *source_x, double *source_y, double *source_z, double *source_q,
//     struct RunParams *run_params, double *potential, int gpu_async_stream_id)
// {
//     double eta = run_params->kernel_params[1];

//     // Grid and block dimensions
//     dim3 blockDim(8, 8, 8); // Adjust block size for your GPU's occupancy
//     dim3 gridDim(
//         (target_x_high_ind - target_x_low_ind + blockDim.x) / blockDim.x,
//         (target_y_high_ind - target_y_low_ind + blockDim.y) / blockDim.y,
//         (target_z_high_ind - target_z_low_ind + blockDim.z) / blockDim.z);

//     // Launch CUDA Kernel
//     DCF_PP_CUDA<<<gridDim, blockDim>>>(
//         target_x_low_ind, target_x_high_ind,
//         target_y_low_ind, target_y_high_ind,
//         target_z_low_ind, target_z_high_ind,
//         target_xmin, target_ymin, target_zmin,
//         target_xdd, target_ydd, target_zdd,
//         target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,
//         cluster_num_sources, cluster_idx_start,
//         source_x, source_y, source_z, source_q,
//         eta, potential);

//     // Synchronize to ensure kernel completion
//     cudaDeviceSynchronize();

//     return;
// }

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void DCF_PP_kernel(
    int target_x_low_ind,    int target_x_high_ind,
    int target_y_low_ind,    int target_y_high_ind,
    int target_z_low_ind,    int target_z_high_ind,
    double target_xmin,      double target_ymin,      double target_zmin,
    double target_xdd,       double target_ydd,       double target_zdd,
    int target_x_dim_glob,   int target_y_dim_glob,   int target_z_dim_glob,
    int cluster_num_sources, int cluster_idx_start,
    const double *source_x, const double *source_y, const double *source_z, const double *source_q,
    double eta, double *potential)
{
    // Compute global indices for this thread
    int ix = blockIdx.x * blockDim.x + threadIdx.x + target_x_low_ind;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + target_y_low_ind;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + target_z_low_ind;

    // Check bounds
    if (ix > target_x_high_ind || iy > target_y_high_ind || iz > target_z_high_ind) {
        return;
    }

    int target_yz_dim = target_y_dim_glob * target_z_dim_glob;
    int ii = (ix * target_yz_dim) + (iy * target_z_dim_glob) + iz;

    double tx = target_xmin + (ix - target_x_low_ind) * target_xdd;
    double ty = target_ymin + (iy - target_y_low_ind) * target_ydd;
    double tz = target_zmin + (iz - target_z_low_ind) * target_zdd;

    double temporary_potential = 0.0;

    // Each thread processes all source points for one (ix, iy, iz)
    for (int j = 0; j < cluster_num_sources; j++) {
        int jj = cluster_idx_start + j;
        double dx = tx - source_x[jj];
        double dy = ty - source_y[jj];
        double dz = tz - source_z[jj];
        double r  = sqrt(dx*dx + dy*dy + dz*dz);

        if (r > DBL_MIN) {
            temporary_potential += source_q[jj] * erf(r / eta) / r;
        }
    }

    // No atomicAdd needed, since each thread handles a unique ii
    potential[ii] = temporary_potential;
}


void K_CUDA_DCF_PP(
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
    // Extract parameter eta from run_params
    double eta = run_params->kernel_params[0];

    // Compute grid and block dimensions
    int x_range = target_x_high_ind - target_x_low_ind + 1;
    int y_range = target_y_high_ind - target_y_low_ind + 1;
    int z_range = target_z_high_ind - target_z_low_ind + 1;

    // Choose block dimensions (tune as necessary)
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((x_range + blockDim.x - 1) / blockDim.x,
                 (y_range + blockDim.y - 1) / blockDim.y,
                 (z_range + blockDim.z - 1) / blockDim.z);

    // Choose a CUDA stream if gpu_async_stream_id corresponds to a valid cudaStream_t.
    // Here we assume you have a pre-created cudaStream_t array or handle mapping to gpu_async_stream_id.
    // If not using streams, just use the default stream (0).
    cudaStream_t stream = 0; // Replace with appropriate stream handle if available

    // Launch the CUDA kernel
    DCF_PP_kernel<<<gridDim, blockDim, 0, stream>>>(
        target_x_low_ind, target_x_high_ind,
        target_y_low_ind, target_y_high_ind,
        target_z_low_ind, target_z_high_ind,
        target_xmin, target_ymin, target_zmin,
        target_xdd, target_ydd, target_zdd,
        target_x_dim_glob, target_y_dim_glob, target_z_dim_glob,
        cluster_num_sources, cluster_idx_start,
        source_x, source_y, source_z, source_q,
        eta, potential);

    // If desired, you can wait for the kernel to finish here
    // cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    return;
}