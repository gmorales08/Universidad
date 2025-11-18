#include "utils.h"

#include <cmath>

void writeBMP(const char* filename, unsigned char* data, int w, int h) {
	if (w % 8) {
		printf("El ancho de la imagen ha de ser multiplo de 8.\n");
		exit(-1);
	}

	const int hs = 54;
	int is = w * h * 3;
	int fs = hs + is;
	const int offset = 40;

	char header[hs];
	header[0] = 'B';
	header[1] = 'M';
	int* pi = (int*)(header + 2);
	*pi = fs;
	pi++;
	pi++;
	*pi = hs;
	pi++;
	*pi = offset;
	pi++;
	*pi = w;
	pi++;
	*pi = h;
	short* ps = (short*)(pi + 1);
	*ps = 1;
	ps++;
	*ps = 24;
	pi = (int*)(ps + 1);
	*pi = 0;
	pi++;
	*pi = is;
	pi++;
	*pi = 0;
	pi++;
	*pi = 0;
	pi++;
	*pi = 0;
	pi++;
	*pi = 0;


	FILE* f;
	fopen_s(&f, filename, "wb");
	if (!f) {
		printf("No se ha podido crear el archivo.\n");
		exit(-3);
	}
	fwrite(header, 1, 54, f);
	fwrite(data, 1, w * h * 3, f);
	fclose(f);
}

__host__ float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}


__host__ bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted) {
	Vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
	if (discriminant > 0) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}

__host__ Vec3 reflect(const Vec3& v, const Vec3& n) {
	return v - 2 * dot(v, n) * n;
}
