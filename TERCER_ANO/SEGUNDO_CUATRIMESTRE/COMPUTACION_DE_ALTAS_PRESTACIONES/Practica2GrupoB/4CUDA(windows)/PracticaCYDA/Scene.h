#pragma once

#include <vector>

#include <curand_kernel.h>

#include "Vec3.h"
#include "Ray.h"
#include "Object.h"

class Scene {
public:
	Scene(int depth = 50) : ol(), sky(), inf(), d(depth) {}
	Scene(const Scene& list) = default;

	void add(Object* o) { ol.push_back(o); }
	void setSkyColor(Vec3 sky) { this->sky = sky; }
	void setInfColor(Vec3 inf) { this->inf = inf; }

	Vec3 getSceneColor(const Ray& r);

protected:
	Vec3 getSceneColor(const Ray& r, int depth);

private:
	std::vector<Object*> ol;
	Vec3 sky;
	Vec3 inf;
	int d;
};

class SceneGPU {
public:
	__device__ SceneGPU() : ol(nullptr), capacity(0), size(0), sky(), inf() {}
	__device__ SceneGPU(Object** list, size_t nobjects, size_t tmax, int depth = 50) : ol(list), capacity(nobjects), size(0), sky(), inf() {}

	__device__ void setList(Object** list, int numobjets) { ol = list; capacity = numobjets; }
	__device__ void add(Object* o) { ol[size] = o; size++; }
	__device__ void setSkyColor(Vec3 sky) { this->sky = sky; }
	__device__ void setInfColor(Vec3 inf) { this->inf = inf; }

	__device__ Vec3 getSceneColor(const Ray& r, curandState* local_rand_state) {
		Ray tempr = r;
		Vec3 tempv = Vec3(1.0, 1.0, 1.0);
		for (int i = 0; i < 50; i++) {
			CollisionData cd;
			Object* aux = nullptr;
			float closest = FLT_MAX;  // std::numeric_limits<float>::max();  // initially tmax = std::numeric_limits<float>::max()
			for (int j = 0; j < size; j++) {
				if (ol[j]->checkCollision(tempr, 0.001f, closest, cd)) { // tmin = 0.001
					aux = ol[j];
					closest = cd.time;
				}
			}

			if (aux) {
				Ray scattered;
				Vec3 attenuation;
				if (aux->scatter(tempr, cd, attenuation, scattered, local_rand_state)) {
					tempv *= attenuation;
					tempr = scattered;
				} else {
					return Vec3(0.0f, 0.0f, 0.0f);
				}
			}
			else {
				Vec3 unit_direction = unit_vector(tempr.direction());
				float t = 0.5f * (unit_direction.y() + 1.0f);
				Vec3 c = (1.0f - t) * inf + t * sky;
				return tempv * c;
			}
		}
		return Vec3(0.0, 0.0, 0.0);
	}


private:
	Object** ol;
	size_t capacity;
	size_t size;
	Vec3 sky;
	Vec3 inf;
};
