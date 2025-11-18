#pragma once

#include <curand_kernel.h>

#include "ray.h"
#include "CollisionData.h"

class Material  {
public:
    __host__ virtual bool scatter(const Ray& ray, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const = 0;
    __device__ virtual bool scatter(const Ray& ray, const CollisionData& cd, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};
