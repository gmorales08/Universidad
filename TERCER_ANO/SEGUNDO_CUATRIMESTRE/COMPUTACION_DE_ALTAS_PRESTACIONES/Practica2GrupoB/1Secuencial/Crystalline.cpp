#include "Crystalline.h"

bool Crystalline::scatter(const Ray& r_in, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const {
	Vec3 outward_normal;
	Vec3 reflected = reflect(r_in.direction(), cd.normal);
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
	if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
		reflect_prob = schlick(cosine, ref_idx);
	else
		reflect_prob = 1.0;
	if (random() < reflect_prob)
		scattered = Ray(cd.p, reflected);
	else
		scattered = Ray(cd.p, refracted);
	return true;
}
