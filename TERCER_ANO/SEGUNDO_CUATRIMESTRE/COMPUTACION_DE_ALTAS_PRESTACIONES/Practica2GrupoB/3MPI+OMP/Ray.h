#pragma once

/*****************************************************************************/
/* Based on the code written in 2016 by Peter Shirley <ptrshrl@gmail.com>    */
/* Check COPYING.txt for copyright license                                   */
/*****************************************************************************/

#include "Vec3.h"

class Ray {
public:
    Ray() {}
    Ray(const Vec3& a, const Vec3& b) : a(a), b(b) {}

    Vec3 origin() const       { return a; }
    Vec3 direction() const    { return b; }

    Vec3 point_at_parameter(float t) const { return a + t*b; }
private:
    Vec3 a;
    Vec3 b;
};
