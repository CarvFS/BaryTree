#ifndef H_PARTICLE_FUNCTIONS_H
#define H_PARTICLE_FUNCTIONS_H

#include "struct_particles.h"


void Particles_Alloc(struct Particles **sources_addr, int length);

void Particles_Free(struct Particles *sources);

void Particles_Targets_Reorder(struct Particles *targets, double *potential);

void Particles_Sources_Reorder(struct Particles *sources);

void Particles_ConstructOrder(struct Particles *particles);

void Particles_FreeOrder(struct Particles *particles);


#endif /* H_PARTICLE_FUNCTIONS */
