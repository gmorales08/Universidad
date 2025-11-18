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
#include <vector>

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

#define SLAVES 5
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

    double progstart = MPI_Wtime();

    constexpr int sizePatch = sizeof(unsigned char) * W * H * 3;

    if (rank == 0) {
        /* MAESTRO */
        int seed = -1;
        std::vector<bool> esclavoLibre(SLAVES);
        std::vector<unsigned char *> datas(FRAMES);
        std::vector<MPI_Request> request(SLAVES);

        for (int i = 0; i < SLAVES; i++) {
            esclavoLibre.at(i) = true;
            request.at(i) = MPI_REQUEST_NULL;
        }

        for (int i = 0; i < FRAMES; i++) {
            datas.at(i) = (unsigned char*)calloc(sizePatch, 1);
        }

        int framesRestantes = FRAMES; /* Frames pendientes de ser recibidos */
        int framesEnviados = 0; /* Frames enviados para ser procesados */
        while (framesRestantes > 0) {
            /* M0. Comprobar si hay algun esclavo libre */
            for (int i = 0; i < SLAVES; i++) {
                if (esclavoLibre.at(i) && framesEnviados < FRAMES) {
                    framesEnviados++;
            /* M1. Enviar la semilla */
                    esclavoLibre.at(i) = false;
                    seed++;
                    MPI_Send(&seed,
                             1,
                             MPI_INT,
                             i + 1,
                             TAG_SEMILLA,
                             comm);
                    printf("%d: Envio la semilla %d a %d\n", rank, seed, i + 1);
            /* M2. Solicitar la informacion de la imagen */
                    MPI_Irecv(datas.at(seed),
                              sizePatch,
                              MPI_UNSIGNED_CHAR,
                              i + 1,
                              TAG_FRAME,
                              comm,
                              &request.at(i));
                    printf("%d: Encargo el frame %d a %d\n", rank, seed, i + 1);
                } else {
                    /* Se ha enviado trabajo al esclavo y esta ocupado */
                    /* Comprobar si ha terminado */
                    MPI_Status status;
                    int haTerminado;
                    MPI_Test(&request.at(i),
                             &haTerminado,
                             &status);
                    if (haTerminado && status.MPI_SOURCE <= SLAVES + 1 &&
                            status.MPI_SOURCE > 0) {
            /* M3. Guardar la imagen */
                        printf("%d: El nodo %d ha terminado su frame\n",
                        rank, status.MPI_SOURCE);
                        esclavoLibre.at(status.MPI_SOURCE - 1) = true;
                        framesRestantes--;
                    }
                }
            }
        }
        /* Avisar a los esclavos que no hay trabajo */
        seed = -1;
        for (int i = 0; i < SLAVES; i++) {
            MPI_Send(&seed,
                     1,
                     MPI_INT,
                     i + 1,
                     TAG_SEMILLA,
                     comm);
        }
            /* M4. Procesar todos los frames */
        for (int i = 0; i < FRAMES; i++) {
            std::string nombreImg = "imgCPUImg" +
                std::to_string(i) + ".bmp";
            writeBMP(nombreImg.c_str(), datas.at(i), W, H);
            printf("Escribo la imagen %s\n", nombreImg.c_str());
        }

    } else {
        /* ESCLAVO */
        while (true) {
            /* E0. Recibir una semilla */
            int seed;
            MPI_Recv(&seed,
                     1,
                     MPI_INT,
                     0,
                     TAG_SEMILLA,
                     comm,
                     MPI_STATUS_IGNORE);
            printf("%d: He recibido la semilla %d\n", rank, seed);
            if (seed == -1) {
                break;
            } else {
            /* E1. Procesar la imagen */
                srand(seed);

                int ns = 10;

                int patch_x_size = W;
                int patch_y_size = H;
                int patch_x_idx = 1;
                int patch_y_idx = 1;

                unsigned char* data = (unsigned char*)calloc(sizePatch, 1);

                int patch_x_start = (patch_x_idx - 1) * patch_x_size;
                int patch_x_end = patch_x_idx * patch_x_size;
                int patch_y_start = (patch_y_idx - 1) * patch_y_size;
                int patch_y_end = patch_y_idx * patch_y_size;
                double tstart = MPI_Wtime();
                rayTracingCPU(data, W, H, ns, patch_x_start, patch_y_start, patch_x_end, patch_y_end);
                double tend = MPI_Wtime();
                printf("%d: He tardado %f en procesar la imagen\n", rank,
                       tend - tstart);

                printf("%d: Voy a enviar la imagen %d\n", rank, seed);
                MPI_Send(data,
                         sizePatch,
                         MPI_UNSIGNED_CHAR,
                         0,
                         TAG_FRAME,
                         comm);
                free(data);
            }
        }
    }
    MPI_Finalize();
    double progend = MPI_Wtime();
    printf("EL PROGRAMA HA DURADO %f\n", progend - progstart);

    return (0);
}
