#pragma once

/*****************************************************************************/
/* Based on the code written in 2016 by Peter Shirley <ptrshrl@gmail.com>    */
/* Check COPYING.txt for copyright license                                   */
/*****************************************************************************/

#include "Ray.h"
#include "CollisionData.h"

class Material  {
public:
    virtual bool scatter(const Ray& ray, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const = 0;
};
