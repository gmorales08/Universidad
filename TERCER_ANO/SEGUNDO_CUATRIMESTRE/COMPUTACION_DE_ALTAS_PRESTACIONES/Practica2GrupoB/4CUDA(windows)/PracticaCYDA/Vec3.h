#pragma once

/*****************************************************************************/
/* Based on the code written in 2016 by Peter Shirley <ptrshrl@gmail.com>    */
/* Check COPYING.txt for copyright license                                   */
/*****************************************************************************/

#include <cmath>
#include <iostream>

class Vec3 {
public:
	__host__ __device__ Vec3() : e{0, 0, 0} {}
	__host__ __device__ Vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

    __host__ __device__ inline const Vec3& operator+() const { return *this; }
    __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; }

	__host__ __device__ inline Vec3& operator+=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator-=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator/=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const float t);
    __host__ __device__ inline Vec3& operator/=(const float t);

	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }
	__host__ __device__ inline float r() const { return e[0]; }
	__host__ __device__ inline float g() const { return e[1]; }
	__host__ __device__ inline float b() const { return e[2]; }

	__host__ __device__ inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
	__host__ __device__ inline void make_unit_vector() { *this /= length(); }

	__host__ __device__ inline float dot(const Vec3& v1, const Vec3& v2);
	__host__ __device__ inline Vec3 cross(const Vec3& v1, const Vec3& v2);
private:
        float e[3];
};

__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3& v) {
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3& v) {
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3& v) {
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v) {
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float t) {
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t) {
	float k = 1.0f / t;

	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

/*****************************************************************************/
/* Extern operators Functions                                                */
/*****************************************************************************/

__host__ __device__ inline Vec3 operator+(Vec3 v1, const Vec3& v2) {
	v1 += v2;
	return (v1);
}

__host__ __device__ inline Vec3 operator-(Vec3 v1, const Vec3& v2) {
	v1 -= v2;
	return (v1);
}

__host__ __device__ inline Vec3 operator*(Vec3 v1, const Vec3& v2) {
	v1 *= v2;
	return (v1);
}

__host__ __device__ inline Vec3 operator/(Vec3 v1, const Vec3& v2) {
	v1 /= v2;
	return (v1);
}

__host__ __device__ inline Vec3 operator*(float t, Vec3 v) {
	v *= t;
	return (v);
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
	v /= t;
	return (v);
}

__host__ __device__ inline Vec3 operator*(Vec3 v, float t) {
	v *= t;
	return (v);
}

__host__ __device__ inline float dot(const Vec3& v1, const Vec3& v2) {
	return v1[0] * v2[0]
		+ v1[1] * v2[1]
		+ v1[2] * v2[2];
}

__host__ __device__ inline Vec3 cross(const Vec3& v1, const Vec3& v2) {
	return Vec3(v1[1] * v2[2] - v1[2] * v2[1],
				v1[2] * v2[0] - v1[0] * v2[2],
				v1[0] * v2[1] - v1[1] * v2[0]);
}

/*****************************************************************************/
/* Other Functions                                                          */
/*****************************************************************************/

__host__ __device__ inline Vec3 unit_vector(Vec3 v) {
	return v / v.length();
}

inline std::istream& operator>>(std::istream& is, Vec3& t) {
	is >> t[0] >> t[1] >> t[2];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const Vec3& t) {
	os << t[0] << " " << t[1] << " " << t[2];
	return os;
}
