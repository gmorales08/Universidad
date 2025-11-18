#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include "cliente.h"
#include "cola.h"

#define TAG_AYUDA    1 /* Se pide/ofrece ayuda */
#define TAG_CLIENTE  2 /* Se envia/recibe un cliente */
#define TAG_CANCELAR 3 /* Se cancela la peticion/recepcion de ayuda */
#define TAG_INFORME  4 /* Se informa del estado la cola propia */

#define MUCHOS_CLIENTES 0 /* Si hay mas de estos clientes en cola esta muy ocupada */

//  #define CASO_DE_PRUEBA   /* Comentar en entrega final */
//  #define CASO_DE_PRUEBA_2 /* Comentar en entrega final */

int main(int argc, char *argv[]) {
    int rank;
    int size;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    srand(time(NULL) + rank); /* Semilla diferente para cada rank */

    int i;
    int nCajas = size;
    int flagAyudaRecibida;
    int flagEnviaCliente;
    int flagCancelaEnvio;
    int flagAyudaOfrecida;
    int flagInformeRecibido;
    int clientesEnCola;
    int cRecibido[] = {-1, -1, -1};
    int *estaOcupada       = (int *) malloc(nCajas * sizeof(int));
    int *seHaOfrecidoAyuda = (int *) malloc(nCajas * sizeof(int));
    MPI_Request requestInforme    = MPI_REQUEST_NULL;
    MPI_Request *requestsAyudaOfr = (MPI_Request *) malloc(nCajas * sizeof(MPI_Request));

    /* Se distribuyen los clientes entre las cajas */
    int nClientesTotal    = atoi(argv[1]);
    int nClientesPorCaja  = nClientesTotal / nCajas;
    int clientesRestantes = nClientesTotal % nCajas;
    int nClientes = nClientesPorCaja + (rank < clientesRestantes ? 1 : 0);
    int cInicial = rank * nClientesPorCaja +
                   (rank < clientesRestantes ? rank : clientesRestantes);
    clientesEnCola = nClientes;
    colaPrioridad_t *cola = crearColaPrioridad();

    #ifdef CASO_DE_PRUEBA
    nClientesTotal   = 16;
    nClientesPorCaja = nClientesTotal / nCajas;
    clientesRestantes = nClientesTotal % nCajas;
    nClientes = nClientesPorCaja + (rank < clientesRestantes ? 1 : 0);
    cInicial = rank * nClientesPorCaja +
                   (rank < clientesRestantes ? rank : clientesRestantes);
    int clientes[16][3] = {
        {0, 0, 6},
        {1, 0, 9},
        {2, 1, 17},
        {3, 0, 6}, /* Da problemas */
        {4, 1, 18},
        {5, 0, 7},
        {6, 0, 9},
        {7, 0, 9},
        {8, 0, 5},
        {9, 0, 6}, /* Da problemas */
        {10, 1, 18},
        {11, 0, 7},
        {12, 1, 16},
        {13, 1, 10},
        {14, 0, 6},
        {15, 0, 7}
    };
    for (int i = 0; i < nClientes; i++) {
        cliente_t *cliente = crearCliente();
        cliente->id = clientes[cInicial + i][0];
        cliente->prioritario = clientes[cInicial + i][1];
        cliente->tiempoEspera = clientes[cInicial + i][2];
        push(cola, cliente);
        fprintf(stderr, "%d: Cliente %d asignado\n", rank, cliente->id);
    }
    #endif
    #ifdef CASO_DE_PRUEBA_2
    nClientesTotal   = 24;
    nClientesPorCaja = nClientesTotal / nCajas;
    clientesRestantes = nClientesTotal % nCajas;
    nClientes = nClientesPorCaja + (rank < clientesRestantes ? 1 : 0);
    cInicial = rank * nClientesPorCaja +
                   (rank < clientesRestantes ? rank : clientesRestantes);
    /* Para 6 procs, 0-4 deberian ayudar a 5 y 6*/
    int clientes[24][3] = {
        {0, 0, 5},
        {1, 0, 5},
        {2, 0, 5},
        {3, 0, 5},
        {4, 0, 5},
        {5, 0, 5},
        {6, 0, 5},
        {7, 0, 5},
        {8, 0, 5},
        {9, 0, 5},
        {10, 0, 5},
        {11, 0, 5},
        {12, 0, 5},
        {13, 0, 5},
        {14, 0, 5},
        {15, 0, 5},
        {16, 0, 5},
        {17, 0, 5},
        {18, 0, 5},
        {19, 1, 20},
        {20, 1, 20},
        {21, 1, 20},
        {22, 1, 20},
        {23, 1, 20}
    };
    for (int i = 0; i < nClientes; i++) {
        cliente_t *cliente = crearCliente();
        cliente->id = clientes[cInicial + i][0];
        cliente->prioritario = clientes[cInicial + i][1];
        cliente->tiempoEspera = clientes[cInicial + i][2];
        push(cola, cliente);
        fprintf(stderr, "%d: Cliente %d asignado\n", rank, cliente->id);
    }
    #endif
    #ifndef CASO_DE_PRUEBA
    #ifndef CASO_DE_PRUEBA_2
    /*  Se asignan clientes a cajas */
    for (i = 0; i < nClientes; i++) {
        cliente_t *clienteCreado = crearCliente();
        clienteCreado->id = cInicial + i;
        push(cola, clienteCreado);
        fprintf(stderr, "%d: Cliente %d asignado\n", rank, clienteCreado->id);
    }
    #endif
    #endif
    /* Se inicializan los vectores */
    for (i = 0; i < nCajas; i++) {
        estaOcupada[i]       = (i < nClientesTotal) ? 1 : 0;
        seHaOfrecidoAyuda[i] = 0;
        requestsAyudaOfr[i]  = MPI_REQUEST_NULL; /* Se ofrece ayuda */
    }
    /* Bucle principal */
    MPI_Barrier(comm); /* Esperamos a que todos los nodos este preparados */
    fprintf(stderr, "%d: Comenzamos\n", rank);
    while (1) {
        /* 1. Atender a los clientes propios */
        if (clientesEnCola > 0) {
            estaOcupada[rank] = 1;
            cliente_t *cliente = pop(cola);
            clientesEnCola--;
            nClientesTotal--;
            fprintf(stderr, "%d: El cliente %d tardara %d seg\n", rank, cliente->id, cliente->tiempoEspera);
            sleep(cliente->tiempoEspera);
            fprintf(stderr, "%d:-----------------------------------------C%d atendido\n", rank, cliente->id);
            /* Avisar a los demas que hay un cliente menos */
            for (i = 0; i < nCajas; i++) {
                if (i != rank) {
                    /* Envio el numero de clientes en cola, pero de momento no lo uso */
                    MPI_Isend(&clientesEnCola,
                              1,
                              MPI_INT,
                              i,
                              TAG_INFORME,
                              comm,
                              &requestInforme);
                }
            }
        } else {
            estaOcupada[rank] = 0;
        }

        /* 2. Ayudar a otras cajas si estamos libres */
        if (!estaOcupada[rank]) {
            /* 2a. Solicitar clientes */
            for (i = 0; i < nCajas; i++) {
                if (i != rank && !seHaOfrecidoAyuda[i] && estaOcupada[i]) {
                    fprintf(stderr, "%d: No tengo clientes, %d te ayudo?\n", rank, i);
                    MPI_Isend(&rank,
                              1,
                              MPI_INT,
                              i,
                              TAG_AYUDA,
                              comm,
                              &requestsAyudaOfr[i]);
                    seHaOfrecidoAyuda[i] = 1;
                }
            }
            /* 2b. Recibir clientes de otras cajas */
            for (i = 0; i < nCajas; i++) {
                if (i != rank && seHaOfrecidoAyuda[i]) {
                    MPI_Test(&requestsAyudaOfr[i],
                             &flagAyudaRecibida,
                             MPI_STATUS_IGNORE);
                    if (flagAyudaRecibida) {
                        MPI_Iprobe(i,
                                   TAG_CLIENTE,
                                   comm,
                                   &flagEnviaCliente,
                                   MPI_STATUS_IGNORE);
                        MPI_Iprobe(i,
                                   TAG_CANCELAR,
                                   comm,
                                   &flagCancelaEnvio,
                                   MPI_STATUS_IGNORE);
                        if (flagEnviaCliente) {
                            MPI_Recv(cRecibido,
                                     3,
                                     MPI_INT,
                                     i,
                                     TAG_CLIENTE,
                                     comm,
                                     MPI_STATUS_IGNORE);
                            estaOcupada[i] = 1;
                            if (cRecibido[0] != -1) {
                                cliente_t *cRecibidoED = (cliente_t *) malloc(sizeof(cliente_t));
                                cRecibidoED->id           = cRecibido[0];
                                cRecibidoED->prioritario  = cRecibido[1];
                                cRecibidoED->tiempoEspera = cRecibido[2];
                                fprintf(stderr, "%d: He recibido el cliente %d de %d\n", rank,
                                        cRecibidoED->id, i);
                                push(cola, cRecibidoED);
                                clientesEnCola++;
                                fprintf(stderr, "%d: Ahora tengo %d clientes. P:%dNP:%d\n", rank, clientesEnCola, cola->prioritarios->n, cola->no_prioritarios->n);
                                estaOcupada[rank] = 1;
                                seHaOfrecidoAyuda[i] = 0;
                            } else {
                                fprintf(stderr, "%d: He recibido un cliente NULL !!!!!!!\n", rank);
                                seHaOfrecidoAyuda[i] = 0;
                            }
                        } else if (flagCancelaEnvio) {
                            fprintf(stderr, "%d: %d Ha cancelado el envio\n", rank, i);
                            seHaOfrecidoAyuda[i] = 0;
                            estaOcupada[i] = 0;
                        }
                    }
                }
            }
        }

        /* 3. Enviar clientes si estamos ocupados */
        /* 3a. Preguntar si alguna caja esta libre */
        if (clientesEnCola > MUCHOS_CLIENTES) {
            fprintf(stderr, "%d: Ayuda, tengo %d clientes. P:%dNP:%d\n", rank, clientesEnCola, cola->prioritarios->n, cola->no_prioritarios->n);
            for (i = 0; i < nCajas; i++) {
                if (i != rank) {
                    MPI_Iprobe(i,
                               TAG_AYUDA,
                               comm,
                               &flagAyudaOfrecida,
                               MPI_STATUS_IGNORE);
                    /* 3b. Esperar que una caja nos ayude y enviarle un cliente */
                    if (flagAyudaOfrecida) {
                        int procLibre;
                        MPI_Recv(&procLibre,
                                 1,
                                 MPI_INT,
                                 i,
                                 TAG_AYUDA,
                                 comm,
                                 MPI_STATUS_IGNORE);
                        if (clientesEnCola > 0) {
                            cliente_t *cEnviadoED = pop(cola);
                            clientesEnCola--;
                            int cEnviado[3] = {
                                cEnviadoED->id,
                                cEnviadoED->prioritario,
                                cEnviadoED->tiempoEspera
                            };
                            fprintf(stderr, "%d: Voy a enviar el cliente %d a %d\n", rank, cEnviadoED->id, procLibre);
                            fprintf(stderr, "%d: Tengo %d clientes. P:%dNP:%d\n", rank, clientesEnCola, cola->prioritarios->n, cola->no_prioritarios->n);
                            MPI_Send(cEnviado,
                                     3,
                                     MPI_INT,
                                     procLibre,
                                     TAG_CLIENTE,
                                     comm);
                        } else {
                            fprintf(stderr, "%d: Falsa alarma %d, no necesito ayuda\n", rank, procLibre);
                            int cEnviado[3] = {-1, -1, -1};
                            MPI_Send(cEnviado,
                                     3,
                                     MPI_INT,
                                     procLibre,
                                     TAG_CANCELAR,
                                     comm);
                        }
                    }
                }
            }
        }

        /* 4. Comprobar si se ha terminado */
        /* 4a. Comprobar si otras cajas han atendido clientes */
        for (i = 0; i < nCajas; i++) {
            if (i != rank) {
                MPI_Iprobe(i,
                           TAG_INFORME,
                           comm,
                           &flagInformeRecibido,
                           MPI_STATUS_IGNORE);
                if (flagInformeRecibido) {
                    int procInforme;
                    MPI_Recv(&procInforme,
                             1,
                             MPI_INT,
                             i,
                             TAG_INFORME,
                             comm,
                             MPI_STATUS_IGNORE);
                    nClientesTotal--;
                    fprintf(stderr, "%d: %d ha procesado un cliente. Quedan %d\n", rank, i, nClientesTotal);
                }
            }
        }
        /* 4b. Comprobar si no quedan clientes por atender */
        if (nClientesTotal == 0) {
            fprintf(stderr, "%d: Hemos acabado\n", rank);
            break;
        }

    }
    /* Se libera la memoria del proceso */
    free(estaOcupada);
    free(seHaOfrecidoAyuda);
    free(requestsAyudaOfr);
    liberarColaPrioridad(cola);

    MPI_Finalize();

    return 0;
}
