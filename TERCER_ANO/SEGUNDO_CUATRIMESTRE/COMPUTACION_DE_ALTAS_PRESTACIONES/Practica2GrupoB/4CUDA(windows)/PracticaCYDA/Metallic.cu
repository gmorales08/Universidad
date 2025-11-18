#include "Metallic.h"

__host__ bool Metallic::scatter(const Ray& r_in, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const {
	Vec3 reflected = reflect(unit_vector(r_in.direction()), cd.normal);
	scattered = Ray(cd.p, reflected + fuzz * randomNormalSphere());
	attenuation = albedo;
	return (dot(scattered.direction(), cd.normal) > 0);
}
