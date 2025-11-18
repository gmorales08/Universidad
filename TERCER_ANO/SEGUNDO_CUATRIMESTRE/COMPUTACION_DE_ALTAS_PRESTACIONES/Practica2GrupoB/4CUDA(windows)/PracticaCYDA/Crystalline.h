#pragma once

#include "utils.h"
#include "random.h"

#include "Material.h"

class Crystalline : public Material {
public:
	__host__ __device__ Crystalline(float ri) : ref_idx(ri) {}

	__host__ bool scatter(const Ray& r_in, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const;
	__device__ bool scatter(const Ray& r_in, const CollisionData& cd, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		Vec3 outward_normal;
		Vec3 reflected = reflectGPU(r_in.direction(), cd.normal);
		float ni_over_nt;
		attenuation = Vec3(1.0, 1.0, 1.0);
		Vec3 refracted;
		float reflect_prob;
		float cosine;
		if (dot(r_in.direction(), cd.normal) > 0) {
			outward_normal = -cd.normal;
			ni_over_nt = ref_idx;
			// cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = dot(r_in.direction(), cd.normal) / r_in.direction().length();
			cosine = sqrt(1 - ref_idx * ref_idx * (1 - cosine * cosine));
		}
		else {
			outward_normal = cd.normal;
			ni_over_nt = 1.0f / ref_idx;
			cosine = -dot(r_in.direction(), cd.normal) / r_in.direction().length();
		}
		if (refractGPU(r_in.direction(), outward_normal, ni_over_nt, refracted))
			reflect_prob = schlickGPU(cosine, ref_idx);
		else
			reflect_prob = 1.0;
		if (curand_uniform(local_rand_state) < reflect_prob)
			scattered = Ray(cd.p, reflected);
		else
			scattered = Ray(cd.p, refracted);
		return true;
	}

private:
	float ref_idx;

	__device__ Vec3 reflectGPU(const Vec3& v, const Vec3& n) const {
		return v - 2 * dot(v, n) * n;
	}

	__device__ float schlickGPU(float cosine, float ref_idx) const {
		float r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}


	__device__ bool refractGPU(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted) const {
		Vec3 uv = unit_vector(v);
		float dt = dot(uv, n);
		float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
		if (discriminant > 0) {
			refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
			return true;
		}
		else
			return false;
	}

};

