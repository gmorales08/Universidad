#pragma once

#include "Vec3.h"
#include "Material.h"

#include <curand_kernel.h>

class Diffuse : public Material {
public:
	__host__ __device__ Diffuse(const Vec3& color) : color(color) {}

	__host__ bool scatter(const Ray& ray, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const {
		Vec3 target = cd.p + cd.normal + randomNormalSphere();
		scattered = Ray(cd.p, target - cd.p);
		attenuation = color;
		return true;
	}

	__device__ bool scatter(const Ray& ray, const CollisionData& cd, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		Vec3 target = cd.p + cd.normal + randomNormalSphereGPU(local_rand_state);
		scattered = Ray(cd.p, target - cd.p);
		attenuation = color;
		return true;
	}

private:
	Vec3 color;

	__device__ Vec3 randomNormalSphereGPU(curandState* local_rand_state) const {
		Vec3 p(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
		while (p.squared_length() >= 1.0f) {
			p = 2.0f * Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - Vec3(1, 1, 1);
		}
		return p;
	}
};
