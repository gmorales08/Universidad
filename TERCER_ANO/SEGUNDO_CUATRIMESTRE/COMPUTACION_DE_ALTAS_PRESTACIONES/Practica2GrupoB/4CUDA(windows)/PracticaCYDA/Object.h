#pragma once

#include <curand_kernel.h>

#include "Shape.h"
#include "Material.h"

class Object {
public:
	__host__ __device__ Object(Shape* shape, Material* material) : s(shape), m(material) {}

	__host__ __device__ bool checkCollision(const Ray& ray, float t_min, float t_max, CollisionData& cd) {
		return (s->collide(ray, t_min, t_max, cd));
	}

	__host__ bool scatter(const Ray& ray, const CollisionData& cd, Vec3& attenuation, Ray& scattered) {
		return (m->scatter(ray, cd, attenuation, scattered));
	}

	__device__ bool scatter(const Ray& ray, const CollisionData& cd, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) {
		return (m->scatter(ray, cd, attenuation, scattered, local_rand_state));
	}

private:
	Shape* s;
	Material* m;
};
