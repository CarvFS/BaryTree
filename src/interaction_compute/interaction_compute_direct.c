#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../utilities/array.h"

#include "../particles/struct_particles.h"
#include "../run_params/struct_run_params.h"

#include "../kernels/coulomb/coulomb.h"
#include "../kernels/tcf/tcf.h"
#include "../kernels/dcf/dcf.h"

#ifdef CUDA_ENABLED
    #include "../kernels/cuda/coulomb/cuda_coulomb.h"
    #include "../kernels/cuda/tcf/cuda_tcf.h"
    #include "../kernels/cuda/dcf/cuda_dcf.h"
#endif

#include "interaction_compute.h"


void InteractionCompute_Direct(double *potential,
                               struct Particles *sources, struct Particles *targets,
                               struct RunParams *run_params)
{

    // define local variables and copy/point to 
    // data on source (solute) and target (grid) structures
    int num_sources   = sources->num; //number of solute atoms
    double *source_x  = sources->x; // x coordinates of solute atoms
    double *source_y  = sources->y; // y coordinates of solute atoms
    double *source_z  = sources->z; // z coordinates of solute atoms
    double *source_q  = sources->q; // charges of solute atoms

    int num_targets   = targets->num; // number of grid points

    double target_xdd = targets->xdd; // grid spacing in x dimension of grid points
    double target_ydd = targets->ydd; // grid spacing in y dimension of grid points
    double target_zdd = targets->zdd; // grid spacing in z dimension of grid points

    double target_xmin = targets->xmin; // minimum x coordinate of grid points
    double target_ymin = targets->ymin; // minimum y coordinate of grid points
    double target_zmin = targets->zmin; // minimum z coordinate of grid points

    int target_xdim = targets->xdim; // number of grid points in x dimension
    int target_ydim = targets->ydim; // number of grid points in y dimension
    int target_zdim = targets->zdim; // number of grid points in z dimension


#ifdef OPENACC_ENABLED
    #pragma acc data copyin(source_x[0:num_sources], source_y[0:num_sources], \
                            source_z[0:num_sources], source_q[0:num_sources]), \
                       copy(potential[0:num_targets])
#endif
    {

/* * ********************************************************/
/* * ************** COMPLETE DIRECT SUM *********************/
/* * ********************************************************/


    /* * *************************************/
    /* * ******* Coulomb *********************/
    /* * *************************************/

    if (run_params->kernel == COULOMB) {

#ifdef CUDA_ENABLED
        #pragma acc host_data use_device(potential, \
                source_x, source_y, source_z, source_q)
        {
        K_CUDA_Coulomb_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            potential, 0);
        }
#else
        K_Coulomb_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
#endif


    /* * *************************************/
    /* * ******* TCF *************************/
    /* * *************************************/

    } else if (run_params->kernel == TCF) {

#ifdef CUDA_ENABLED
        #pragma acc host_data use_device(potential, \
                source_x, source_y, source_z, source_q)
        {
        // Define kap and eta variables here and pass them to the kernel, instead of passing run_params struct
        double kap = run_params->kernel_params[0]; // charge smearing parameter
        double eta = run_params->kernel_params[1]; // contribution to the inverse Debye length of ionic co-solvent 

        K_CUDA_TCF_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            kap, eta, potential, 0);
        }
#else
        K_TCF_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
#endif

                        
    /* * *************************************/
    /* * ******* DCF *************************/
    /* * *************************************/

    } else if (run_params->kernel == DCF) {

#ifdef CUDA_ENABLED
        #pragma acc host_data use_device(potential, \
                source_x, source_y, source_z, source_q)
        {
        K_CUDA_DCF_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
        }
#else
        K_DCF_PP(
            0,  target_xdim-1,
            0,  target_ydim-1,
            0,  target_zdim-1,
            target_xmin, target_ymin, target_zmin,
            
            target_xdd,  target_ydd,  target_zdd,
            target_xdim, target_ydim, target_zdim,

            num_sources, 0,
            source_x, source_y, source_z, source_q,

            run_params, potential, 0);
#endif
                        
    }

#ifdef OPENACC_ENABLED
        #pragma acc wait
#endif
    } // end acc data region

    return;
}
