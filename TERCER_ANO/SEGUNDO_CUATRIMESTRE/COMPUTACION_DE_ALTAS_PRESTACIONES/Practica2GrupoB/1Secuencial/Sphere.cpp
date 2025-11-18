#include "Sphere.h"

bool Sphere::collide(const Ray& ray, float t_min, float t_max, CollisionData& cd) const {
	Vec3 oc = ray.origin() - center;
	float a = dot(ray.direction(), ray.direction());
	float b = dot(oc, ray.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			cd.time = temp;
			cd.p = ray.point_at_parameter(cd.time);
			cd.normal = (cd.p - center) / radius;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			cd.time = temp;
			cd.p = ray.point_at_parameter(cd.time);
			cd.normal = (cd.p - center) / radius;
			return true;
		}
	}
	return false;
}
