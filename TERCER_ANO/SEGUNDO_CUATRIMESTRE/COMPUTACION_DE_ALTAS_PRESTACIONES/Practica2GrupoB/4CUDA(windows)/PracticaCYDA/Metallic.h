#pragma once

#include "utils.h"
#include "random.h"

#include "Material.h"

#include <curand_kernel.h>

class Metallic : public Material {
public:
	__host__ __device__ Metallic(const Vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }

	__host__ bool scatter(const Ray& r_in, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const;

	__device__ bool scatter(const Ray& r_in, const CollisionData& cd, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		Vec3 reflected = reflectGPU(unit_vector(r_in.direction()), cd.normal);
		scattered = Ray(cd.p, reflected + fuzz * randomNormalSphereGPU(local_rand_state));
		attenuation = albedo;
		return (dot(scattered.direction(), cd.normal) > 0);
	}

private:
	Vec3 albedo;
	float fuzz;

	__device__ Vec3 randomNormalSphereGPU(curandState* local_rand_state) const {
		Vec3 p(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
		while (p.squared_length() >= 1.0f) {
			p = 2.0f * Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - Vec3(1, 1, 1);
		}
		return p;
	}

	__device__ Vec3 reflectGPU(const Vec3& v, const Vec3& n) const {
		return v - 2 * dot(v, n) * n;
	}

};
