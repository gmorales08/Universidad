#include "Scene.h"
#include "limits"

Vec3 Scene::getSceneColor(const Ray& r) {
	return getSceneColor(r, 0);
}

Vec3 Scene::getSceneColor(const Ray& r, int depth) {
	CollisionData cd;
	Object* aux = nullptr;
	bool hasCollided = false;
	float closest = std::numeric_limits<float>::max();  // initially tmax = std::numeric_limits<float>::max()
	for (auto& o : ol) {
		if (o->checkCollision(r, 0.001f, closest, cd)) { // tmin = 0.001
			aux = o;
			closest = cd.time;
		}
	}

	if (aux) {
		Ray scattered;
		Vec3 attenuation;
		if (depth < d && aux->scatter(r, cd, attenuation, scattered)) {
			return attenuation *getSceneColor(scattered, depth + 1);
		}
		else {
			return Vec3(0, 0, 0);
		}
	}
	else {
		Vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
	}
}
