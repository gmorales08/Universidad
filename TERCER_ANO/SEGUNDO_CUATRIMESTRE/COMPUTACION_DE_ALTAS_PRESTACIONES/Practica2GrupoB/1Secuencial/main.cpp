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

#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <string>

#include <mpi.h>

#include "Camera.h"
#include "Object.h"
#include "Scene.h"
#include "Sphere.h"
#include "Diffuse.h"
#include "Metallic.h"
#include "Crystalline.h"

#include "random.h"
#include "utils.h"

#define SLAVES FRAMES
#define FRAMES 9
#define W 512
#define H 225


#define TAG_SEMILLA 0
#define TAG_FRAME 1

Scene randomScene() {
    int n = 500;
    Scene list;
    list.add(new Object(
        new Sphere(Vec3(0, -1000, 0), 1000),
        new Diffuse(Vec3(0.5, 0.5, 0.5))
    ));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = randomF();
            Vec3 center(a + 0.9f * randomF(), 0.2f, b + 0.9f * randomF());
            if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f) {
                if (choose_mat < 0.8f) {  // diffuse
                    list.add(new Object(
                        new Sphere(center, 0.2f),
                        new Diffuse(Vec3(randomF() * randomF(),
                            randomF() * randomF(),
                            randomF() * randomF()))
                    ));
                }
                else if (choose_mat < 0.95f) { // metal
                    list.add(new Object(
                        new Sphere(center, 0.2f),
                        new Metallic(Vec3(0.5f * (1 + randomF()),
                            0.5f * (1 + randomF()),
                            0.5f * (1 + randomF())),
                            0.5f * randomF())
                    ));
                }
                else {  // glass
                    list.add(new Object(
                        new Sphere(center, 0.2f),
                        new Crystalline(1.5f)
                    ));
                }
            }
        }
    }

    list.add(new Object(
        new Sphere(Vec3(0, 1, 0), 1.0),
        new Crystalline(1.5f)
    ));
    list.add(new Object(
        new Sphere(Vec3(-4, 1, 0), 1.0f),
        new Diffuse(Vec3(0.4f, 0.2f, 0.1f))
    ));
    list.add(new Object(
        new Sphere(Vec3(4, 1, 0), 1.0f),
        new Metallic(Vec3(0.7f, 0.6f, 0.5f), 0.0f)
    ));

    return list;
}

void rayTracingCPU(unsigned char* img, int w, int h, int ns = 10, int px = 0, int py = 0, int pw = -1, int ph = -1) {
    if (pw == -1) pw = w;
    if (ph == -1) ph = h;
    int patch_w = pw - px;
    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);

    for (int j = 0; j < (ph - py); j++) {
        for (int i = 0; i < (pw - px); i++) {

            Vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float u = float(i + px + randomF()) / float(w);
                float v = float(j + py + randomF()) / float(h);
                Ray r = cam.get_ray(u, v);
                col += world.getSceneColor(r);
            }
            col /= float(ns);
            col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            img[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
            img[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
            img[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
        }
    }
}

int main() {
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double tstart = MPI_Wtime();

    constexpr int sizePatch = sizeof(unsigned char) * W * H * 3;

    if (rank == 0) {
        /* MAESTRO */
        unsigned char* data = (unsigned char*)calloc(sizePatch, 1);
        for (int i = 0; i < FRAMES; i++) {

            srand(i);

            int ns = 10;

            int patch_x_size = W;
            int patch_y_size = H;
            int patch_x_idx = 1;
            int patch_y_idx = 1;


            int patch_x_start = (patch_x_idx - 1) * patch_x_size;
            int patch_x_end = patch_x_idx * patch_x_size;
            int patch_y_start = (patch_y_idx - 1) * patch_y_size;
            int patch_y_end = patch_y_idx * patch_y_size;

            rayTracingCPU(data, W, H, ns, patch_x_start, patch_y_start, patch_x_end, patch_y_end);

            std::string nombreImg = "imgCPUImg" +
                std::to_string(i) + ".bmp";
            writeBMP(nombreImg.c_str(), data, W, H);
            printf("%d: Imagen %d creada.\n",rank, i);
        }
        free(data);
        double tend = MPI_Wtime();
        std::cout << "El programa ha tardado " << tend - tstart << " segundos\n";
    }

    MPI_Finalize();

    return (0);
}
