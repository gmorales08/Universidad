#include "random.h"

#include <cstdlib>

float randomF() {
    return rand() / (RAND_MAX + 1.0f);
}

Vec3 randomNormalSphere() {
    Vec3 p(2.0f * Vec3(randomF(), randomF(), randomF()) - Vec3(1, 1, 1));
    while (p.squared_length() >= 1.0) {
        p = 2.0f * Vec3(randomF(), randomF(), randomF()) - Vec3(1, 1, 1);
    }
    return p;
}

Vec3 randomNormalDisk() {
    Vec3 p(2.0f * Vec3(randomF(), randomF(), 0) - Vec3(1, 1, 0));
    while (dot(p, p) >= 1.0f) {
        p = 2.0f * Vec3(randomF(), randomF(), 0) - Vec3(1, 1, 0);
    }
    return p;
}
