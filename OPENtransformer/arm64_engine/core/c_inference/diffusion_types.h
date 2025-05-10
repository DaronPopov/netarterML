#ifndef DIFFUSION_TYPES_H
#define DIFFUSION_TYPES_H

#include <stdint.h>

// Callback function type for progress reporting
typedef void (*DiffusionPerfCallback)(int step, int total_steps, float step_time, void* user_data);

#endif // DIFFUSION_TYPES_H 