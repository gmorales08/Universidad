#pragma once

#include "vec3.h"

void writeBMP(const char* filename, unsigned char* data, int w, int h);

__host__ float schlick(float cosine, float ref_idx);
__host__ bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted);

__host__ Vec3 reflect(const Vec3& v, const Vec3& n);
