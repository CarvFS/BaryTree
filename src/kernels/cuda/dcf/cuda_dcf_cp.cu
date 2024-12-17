#include <math.h>
#include <float.h>
#include <stdio.h>

#include "cuda_dcf_cp.h"


__global__ 
static void compute(double eta, int batch_num_sources, int batch_idx_start,
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q)
{
    // printf("Hello DCF_CP thread %d, block %d\n", threadIdx.x, blockIdx.x);
}


__host__
void K_CUDA_DCF_CP_Lagrange(
    int batch_num_sources, int batch_idx_start, 
    int cluster_q_start, int cluster_pts_start, int interp_order_lim,
    double *source_x, double *source_y, double *source_z, double *source_q,
    double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_q,
    struct RunParams *run_params, int gpu_async_stream_id)
{
    double eta = run_params->kernel_params[0];

    compute<<<1,1>>>(eta, batch_num_sources, batch_idx_start,
                      cluster_q_start, cluster_pts_start, interp_order_lim,
                      source_x,  source_y,  source_z,  source_q,
                      cluster_x, cluster_y, cluster_z, cluster_q);
    cudaDeviceSynchronize();

    return;
}
