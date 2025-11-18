#pragma once

/*****************************************************************************/
/* Based on the code written in 2016 by Peter Shirley <ptrshrl@gmail.com>    */
/* Check COPYING.txt for copyright license                                   */
/*****************************************************************************/

#include "vec3.h"

class Ray {
public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3& a, const Vec3& b) { A = a; B = b; }

    __host__ __device__ Vec3 origin() const       { return A; }
    __host__ __device__ Vec3 direction() const    { return B; }

    __host__ __device__ Vec3 point_at_parameter(float t) const { return A + t*B; }
private:
    Vec3 A;
    Vec3 B;
};
