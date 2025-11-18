#include "random.h"

#include <cstdlib>

float random() {
	return rand() / (RAND_MAX + 1.0f);
}

Vec3 randomNormalSphere() {
	Vec3 p(2.0f * Vec3(random(), random(), random()) - Vec3(1, 1, 1));
	while (p.squared_length() >= 1.0) {
		p = 2.0f * Vec3(random(), random(), random()) - Vec3(1, 1, 1);
	}
	return p;
}

Vec3 randomNormalDisk() {
	Vec3 p(2.0f * Vec3(random(), random(), 0) - Vec3(1, 1, 0));
	while (dot(p, p) >= 1.0f) {
		p = 2.0f * Vec3(random(), random(), 0) - Vec3(1, 1, 0);
	}
	return p;
}

/*****************************************************************************/
/* GPU                                                                       */
/*****************************************************************************/

//__device__ Vec3 randomNormalSphereGPU(curandState* local_rand_state) {
//	Vec3 p(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
//	while (p.squared_length() >= 1.0f) {
//		p = 2.0f * Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - Vec3(1, 1, 1);
//	}
//	return p;
//}

//__device__ Vec3 randomNormalDiskGPU(curandState* local_rand_state) {
//	Vec3 p(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0);
//	while (dot(p, p) >= 1.0f) {
//		p = 2.0f * Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vec3(1, 1, 0);
//	}
//	return p;
//}
