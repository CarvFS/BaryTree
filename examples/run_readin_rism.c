#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <float.h>

#include "../src/utilities/tools.h"
#include "../src/utilities/timers.h"
#include "../src/utilities/array.h"

#include "../src/particles/struct_particles.h"
#include "../src/run_params/struct_run_params.h"
#include "../src/run_params/run_params.h"

#include "../src/drivers/treedriver.h"
#include "../src/drivers/directdriver.h"

#include "support_fns.h"


int main(int argc, char **argv)
{
    printf("[random cube example] Beginning random cube example.\n");

    /* run parameters */
    // N: Number of sources (e.g. The number atoms in the solute molecule)
    int N; 
    // run_direct: Flag to run direct sum and compare with treecode version
    int run_direct; 
    // slice: Will get slices of the grid to carry out calculation on a subset. 
    // Use 1,1,1 to get full grid
    int slice[3]; 
    // xyz_limits: min, max pairs that defines the box size in x,y,z dimensions
    double xyz_limits[6];
    // grid_dim: the number of grids on each dimension
    int grid_dim[3];
    // grid_dd: grid spacing on each dimension
    double grid_dd[3];
    // file_pqr: file containing coordinates and charge of each solute atom.
    // Each line has the information: x, y, z, q
    char file_pqr[256];
    
    struct RunParams *run_params = NULL;
    
    FILE *fp = fopen(argv[1], "r");
    Params_Parse_Readin(fp, &run_params, &N, file_pqr, &run_direct, slice, xyz_limits, grid_dim);

    // Calculate grid spacing from the limits given
    grid_dd[0] = (xyz_limits[1] - xyz_limits[0]) / (grid_dim[0] - 1);
    grid_dd[1] = (xyz_limits[3] - xyz_limits[2]) / (grid_dim[1] - 1);
    grid_dd[2] = (xyz_limits[5] - xyz_limits[4]) / (grid_dim[2] - 1);

    /* data structures for BaryTree calculation and comparison */
    struct Particles *sources = NULL;
    struct Particles *targets = NULL;
    struct Particles *targets_sample = NULL;
    double *potential = NULL, *potential_direct = NULL;
    
    /* variables for collecting accuracy info */
    double potential_engy = 0, potential_engy_direct = 0;
    double inf_err = 0, n2_err = 0, relinf_err = 0, reln2_err = 0;

    /* variables for date-time calculation */
    double time_run[4], time_tree[13], time_direct[4];


    /* Beginning total runtime timer */
    START_TIMER(&time_run[3]);
    
    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Setup
    //~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    START_TIMER(&time_run[0]);


    /* Setting up sources */
    // Allocate memory for source (solute) pointer
    sources = malloc(sizeof(struct Particles));

    // Assign number of particles in the source (solute)
    sources->num = N;

    // Allocating memory for x,y,z,q arrays containing N values
    make_vector(sources->x, sources->num);
    make_vector(sources->y, sources->num);
    make_vector(sources->z, sources->num);
    make_vector(sources->q, sources->num);

    FILE *points_fp = fopen(file_pqr, "r");

    // Copy data coordinates and charges from file to source->x,y,z,q
    for (int i = 0; i < N; ++i) {
        fscanf(points_fp, "%lf %lf %lf %lf", &(sources->x[i]), &(sources->y[i]),
                                             &(sources->z[i]), &(sources->q[i]));
    }
    fclose(points_fp);

    
    /* Setting up targets */
    // Allocate memory for target (grid) pointer
    targets = malloc(sizeof(struct Particles));

    //Assigning total number of grid points
    targets->num = grid_dim[0] * grid_dim[1] * grid_dim[2];
    
    // Assign xyz minimum and maximum values
    targets->xmin = xyz_limits[0];
    targets->ymin = xyz_limits[2];
    targets->zmin = xyz_limits[4];

    targets->xmax = xyz_limits[1];
    targets->ymax = xyz_limits[3];
    targets->zmax = xyz_limits[5];

    // Assign number of grid points in each dimension
    targets->xdim = grid_dim[0];
    targets->ydim = grid_dim[1];
    targets->zdim = grid_dim[2];

    // Assign grid spacing
    targets->xdd = grid_dd[0];
    targets->ydd = grid_dd[1];
    targets->zdd = grid_dd[2];

    printf("[random cube example] Setup has finished.\n");

    /* Initializing direct and treedriver runs */

    // Allocate memory for the potential: size of array
    // is the total number of grid points
    potential = malloc(sizeof(double) * targets->num);

    // If using slices of data, target_sample will store only
    // these grid points
    targets_sample = malloc(sizeof(struct Particles));

    // Allocate memory for potential obtained with direct sum
    potential_direct = malloc(sizeof(double) * targets->num);

    // Set both potential variables to zero
    memset(potential, 0, targets->num * sizeof(double));
    memset(potential_direct, 0, targets->num * sizeof(double));


#ifdef OPENACC_ENABLED
    #pragma acc set device_num(0) device_type(acc_device_nvidia)
    #pragma acc init device_type(acc_device_nvidia)
#endif

    STOP_TIMER(&time_run[0]);


    //~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Running direct comparison
    //~~~~~~~~~~~~~~~~~~~~~~~~~~
    // If running direct sum:
    if (run_direct == 1) {

        // Setup target_sample struct's members by taking into account the slicing
        targets_sample->num = grid_dim[0]/slice[0] * grid_dim[1]/slice[1] * grid_dim[2]/slice[2];
        
        targets_sample->xmin = xyz_limits[0];
        targets_sample->ymin = xyz_limits[2];
        targets_sample->zmin = xyz_limits[4];

        targets_sample->xmax = xyz_limits[0] + grid_dd[0] * (grid_dim[0]-slice[0]);
        targets_sample->ymax = xyz_limits[2] + grid_dd[1] * (grid_dim[1]-slice[1]);
        targets_sample->zmax = xyz_limits[4] + grid_dd[2] * (grid_dim[2]-slice[2]);

        targets_sample->xdim = grid_dim[0]/slice[0];
        targets_sample->ydim = grid_dim[1]/slice[1];
        targets_sample->zdim = grid_dim[2]/slice[2];

        targets_sample->xdd = grid_dd[0]*slice[0];
        targets_sample->ydd = grid_dd[1]*slice[1];
        targets_sample->zdd = grid_dd[2]*slice[2];

        printf("[random cube example] Running direct comparison...\n");

        START_TIMER(&time_run[1]);
        directdriver(sources, targets_sample, run_params, potential_direct, time_direct);
        STOP_TIMER(&time_run[1]);

        free(targets_sample);

    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Running treecode
    //~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    printf("[random cube example] Running treedriver...\n");

    START_TIMER(&time_run[2]);
    treedriver(sources, targets, run_params, potential, time_tree);
    STOP_TIMER(&time_run[2]);

    
    /* Ending total runtime timer */
    STOP_TIMER(&time_run[3]);


    //~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Calculate results
    //~~~~~~~~~~~~~~~~~~~~~~~~~~

    Timing_Print(time_run, time_tree, time_direct, run_direct, run_params);
    
    if (run_direct == 1) {
        Accuracy_Calculate(&potential_engy, &potential_engy_direct,
                           &inf_err, &relinf_err, &n2_err, &reln2_err,
                           potential, potential_direct, grid_dim, slice);
        Accuracy_Print(potential_engy, potential_engy_direct,
                           inf_err, relinf_err, n2_err, reln2_err, slice);
    }
    
    CSV_Print(N, targets->num, run_params, time_run, time_tree, time_direct,
              potential_engy, potential_engy_direct,
              inf_err, relinf_err, n2_err, reln2_err);


    //~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Cleanup
    //~~~~~~~~~~~~~~~~~~~~~~~~~~

    free_vector(sources->x);
    free_vector(sources->y);
    free_vector(sources->z);
    free_vector(sources->q);
    free(sources);
    free(targets);

    free(potential);
    free(potential_direct);

    RunParams_Free(&run_params);

    return 0;
}
