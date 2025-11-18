//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#include <cstdio>
#include <cstdlib>

#include "raytracing.h"

#include "Vec3.h"
#include "Camera.h"
#include "Object.h"
#include "Scene.h"
#include "Sphere.h"
#include "Diffuse.h"
#include "Metallic.h"
#include "Crystalline.h"
#include <chrono>
#include <iostream>

#include "random.h"
#include "utils.h"

int main() {
	int w = 512;// 1200;
	int h = 256;// 800;
	int ns = 10;
	//clock_t start, stop;
	double timer_seconds;

	size_t size = sizeof(unsigned char) * w * h * 3;
	unsigned char* data = (unsigned char*)malloc(size);
	unsigned char* imgCPU = (unsigned char*)malloc(size);

	Vec3* img;
	size_t isize = w * h * sizeof(Vec3);
	cudaMallocManaged((void**)&img, isize);

	//std::cerr << "--- CPU ---\n";
	//start = clock();
	//rayTracingCPU(img, w, h, ns);

	for (int i = h - 1; i >= 0; i--) {
		for (int j = 0; j < w; j++) {
			size_t idx = i * w + j;
			data[idx * 3 + 0] = char(255.99 * img[idx].b());
			data[idx * 3 + 1] = char(255.99 * img[idx].g());
			data[idx * 3 + 2] = char(255.99 * img[idx].r());
		}
	}
	//stop = clock();
	//timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	//std::cerr << "CPU took " << timer_seconds << " seconds.\n\n";

	//writeBMP("imgCPU-prueba.bmp", data, w, h);
	//printf("Imagen CPU creada.\n");

	std::cerr << "--- GPU ---\n";
	//start = clock();

	auto start = std::chrono::high_resolution_clock::now();

	rayTracingGPU(img, imgCPU, w, h, size, ns);

	// Capturar el tiempo final
	auto end = std::chrono::high_resolution_clock::now();

	// Calcular la duración de la ejecución del kernel
	std::chrono::duration<float, std::milli> duration = end - start;

	// Imprimir el tiempo de ejecución
	std::cout << "Tiempo total procesado: " << duration.count() << " ms" << std::endl;

	/*
	for (int i = h - 1; i >= 0; i--) {
		for (int j = 0; j < w; j++) {
			size_t idx = i * w + j;
			data[idx * 3 + 0] = char(255.99 * img[idx].b());
			data[idx * 3 + 1] = char(255.99 * img[idx].g());
			data[idx * 3 + 2] = char(255.99 * img[idx].r());
		}
	}*/

	//stop = clock();
	//timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	//std::cerr << "GPU took " << timer_seconds << " seconds.\n";

	//writeBMP("imgGPU-prueba.bmp", data, w, h);
	//printf("Imagen GPU creada.\n");

	free(data);
	free(imgCPU);
	cudaDeviceReset();

	getchar();
	return (0);
}
