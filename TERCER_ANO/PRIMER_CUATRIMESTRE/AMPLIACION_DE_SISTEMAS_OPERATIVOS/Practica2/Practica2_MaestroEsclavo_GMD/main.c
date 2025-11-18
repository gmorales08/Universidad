#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include "cliente.h"
#include "cola.h"

#define TAG_TERMINAR   0 /* Tag para terminar la ejecucion cuando no hay trabajo */
#define TAG_CAJA_LIBRE 1 /* Senal de que la caja esta libre */
#define TAG_CLIENTE    2 /* Lo que se envia/recibe es un cliente */
#define TAG_CATENDIDO  3 /* Se ha terminado de atender a un cliente */

int main(int argc, char *argv[]) {
    srand(time(NULL)); /* Para que las variables aleatorias cambien
                          en cada ejecucion */
    int rank;
    int size;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
    /* MAESTRO */
        int i;
        /* Creacion de cajas */
        int nCajas = size - 1;
        int *estaAbierta    = (int *) malloc(nCajas * sizeof(int));
        int *estaOcupada    = (int *) malloc(nCajas * sizeof(int));
        int *estaTrabajando = (int *) malloc(nCajas * sizeof(int));
        int *seHaPreguntado = (int *) malloc(nCajas * sizeof(int));
        int *esPrioritaria  = (int *) malloc(nCajas * sizeof(int));
        for (i = 0; i < nCajas; i++) {
            estaAbierta[i]    = 0;
            estaOcupada[i]    = 0;
            estaTrabajando[i] = 0;
            seHaPreguntado[i] = 0;
            esPrioritaria[i]  = 0;
        }
        int cajasAbiertas = nCajas / 2;        /* Se abre el 50% de las cajas */
        int cPrioritarias = cajasAbiertas / 4; /* Un 25% de las abiertas son prioritarias */
        int cPrioritariasOcupadas = 0;
        for (i = 0; i < cajasAbiertas; i++) {
            if (i < cPrioritarias) {
                estaAbierta[i]   = 1;
                esPrioritaria[i] = 1;
                fprintf(stderr, "Se abre la caja %d como prioritaria\n", i + 1);
            } else {
                estaAbierta[i] = 1;
                fprintf(stderr, "Se abre la caja %d\n", i + 1);
            }
        }
        /* Creacion de la cola y clientes */
        int nClientes = atoi(argv[1]);
        colaPrioridad_t *cola = crearColaPrioridad();
        for (i = 0; i < nClientes; i++) {
            cliente_t *clienteCreado = crearCliente();
            clienteCreado->id = i;
            push(cola, clienteCreado);
        }
        fprintf(stderr, "Numero de clientes: %d\n", nClientes);
        fprintf(stderr, "Numero de cajas: %d\n", nCajas);
        fprintf(stderr, "  Abiertas: %d\n", cajasAbiertas);
        fprintf(stderr, "  Prioritarias: %d\n", cPrioritarias);
        /* Bucle principal maestro */
        int clientesEnCola = nClientes;
        int esclavoPreparado;
        int clienteTerminado;
        /* Variables para saber que tipo de cliente recibe cada caja */
        // int CLP_CAP   = 0; /* Cl. prioritario que va a caja prioritaria */
        // int CLP_CANP  = 0; /* Cl. prioritario que va a caja no prioritaria */
        // int CLNP_CANP = 0; /* Cl. no prioritario a caja no prioritaria */
        /* Requests para saber cuando una caja esta dispuesta a aceptar clientes */
        MPI_Request *requestsCajaLibre = (MPI_Request *) malloc(nCajas * sizeof(MPI_Request));
        for (i = 0; i < nCajas; i++) requestsCajaLibre[i] = MPI_REQUEST_NULL;
        /* Requests para saber cuando una caja ha terminado de atender a un cliente */
        MPI_Request *requestsCAtendido = (MPI_Request *) malloc(nCajas * sizeof(MPI_Request));
        for (i = 0; i < nCajas; i++) requestsCAtendido[i] = MPI_REQUEST_NULL;

        while (clientesEnCola > 0) {
            /* 0. Se ajusta el numero de cajas abiertas */
            /* Se abre 1 caja si hay el doble de cola que de cajas */
            if (clientesEnCola >= cajasAbiertas * 2 && cajasAbiertas < nCajas) {
                for (i = 0; i < nCajas; i++) {
                    if (!estaAbierta[i] && !estaOcupada[i] && !estaTrabajando[i]) {
                        estaAbierta[i] = 1;
                        cajasAbiertas++;
                        /* Comprobamos si se necesitan cajas prioritarias */
                        if (cPrioritarias < cajasAbiertas / 4) {
                            esPrioritaria[i] = 1;
                            cPrioritarias++;
                            fprintf(stderr, "%d: Hay %d clientes en cola, abro la caja %d como vip\n",
                                rank, clientesEnCola, i + 1);
                        } else {
                            esPrioritaria[i] = 0;
                            fprintf(stderr, "%d: Hay %d clientes en cola, abro la caja %d\n",
                                rank, clientesEnCola, i + 1);
                        }
                        break; /* Abro solo una de momento */
                    }
                }
            /* Si hay mas cajas abiertas que clientes en cola se cierran cajas */
            } else if (clientesEnCola < cajasAbiertas && cajasAbiertas > 1)  {
                i = 0;
                while (clientesEnCola < cajasAbiertas) {
                    if (i >= nCajas - 1) break;
                    if (estaAbierta[i] && !estaOcupada[i] && !estaTrabajando[i]) {
                        if ((esPrioritaria[i] && (cPrioritarias > cajasAbiertas / 4)) ||
                            (!esPrioritaria[i] && (cPrioritarias <= cajasAbiertas / 4))) {
                            estaAbierta[i] = 0;
                            cajasAbiertas--;
                            if (esPrioritaria[i]) cPrioritarias--;
                            fprintf(stderr, "%d: Hay %d clientes en cola, cierro la caja %d (vip=%d)\n",
                                    rank, clientesEnCola, i + 1, esPrioritaria[i]);
                        }
                    }
                    i++;
                }
            }
            /* 1. Se comprueba si hay alguna caja libre */
            for (i = 1; i < nCajas + 1; i++) {
                if (estaAbierta[i - 1] && !estaOcupada[i - 1] && !estaTrabajando[i - 1]) {
                    /* Solo se envía el request si sabemos que va a ser aceptado */
                    int seVaAEnviar = 0;
                    if (esPrioritaria[i - 1] && cola->prioritarios->n > 0) {
                        seVaAEnviar = 1;
                        fprintf(stderr, "%d: La caja %d (vip=%d) está libre\n", rank, i, esPrioritaria[i - 1]);
                    } else if (!esPrioritaria[i - 1] && (cola->no_prioritarios->n > 0 ||
                    (cola->prioritarios->n > 0 && cPrioritarias - cPrioritariasOcupadas <= 0))) {
                        seVaAEnviar = 1;
                        fprintf(stderr, "%d: La caja %d (vip=%d) está libre\n", rank, i, esPrioritaria[i - 1]);
                    }

                    /* Solo se envía el request si sabemos que va a ser aceptado */
                    if (seVaAEnviar) {
                        estaOcupada[i - 1] = 1;
                        MPI_Irecv(&esclavoPreparado, 1, MPI_INT, i, TAG_CAJA_LIBRE, comm, &requestsCajaLibre[i - 1]);
                    }
                }
            }

            /* Se espera a que una caja quede libre */
            int cajaLibre;
            int flag;
            MPI_Testany(nCajas,
                        requestsCajaLibre,
                        &cajaLibre,
                        &flag,
                        MPI_STATUS_IGNORE);
            if (flag && cajaLibre >= 0 && estaAbierta[cajaLibre] && estaOcupada[cajaLibre]) {
                /* 2. Se envia un cliente a la caja */
                if (clientesEnCola > 0) {
                    /* Comprobamos si la caja libre es prioritaria */
                    fprintf(stderr, "Cola: P:%d NP:%d\n", cola->prioritarios->n, cola->no_prioritarios->n);
                    if (esPrioritaria[cajaLibre] && cola->prioritarios->n > 0) {
                        cliente_t *clienteActual = pop(cola);
                        /* En vez de enviar un cliente_t se envia solo su id y su
                        * tiempo de espera. Y si es vip o no */
                        int clienteEnviar[3] = {clienteActual->id,
                                                clienteActual->prioritario,
                                                clienteActual->tiempoEspera};
                        fprintf(stderr, "CLP_CAP Cliente %d (vip=%d) se enviara a caja %d\n",
                                clienteActual->id, clienteActual->prioritario, cajaLibre + 1);
                        clientesEnCola--;
                        cPrioritariasOcupadas++;
                        // CLP_CAP--;
                        MPI_Request request;
                        MPI_Isend(clienteEnviar,
                                3,
                                MPI_INT,
                                cajaLibre + 1,
                                TAG_CLIENTE,
                                comm,
                                &request);
                        estaTrabajando[cajaLibre] = 1;
                    } else if (!esPrioritaria[cajaLibre]) {
                        /* Si el cliente es prioritario se comprueba si hay
                           cajas prioritarias libres */
                        if (cola->prioritarios->n > 0 && (cPrioritarias - cPrioritariasOcupadas <= 0)) {
                            // int cPrioLibre = 0;
                            // for (i = 0; i < nCajas; i++) {
                            //     if (!estaTrabajando[i] && esPrioritaria[i]) {
                            //         cPrioLibre = 1;
                            //     }
                            // }
                            // if (!cPrioLibre) {
                                cliente_t *clienteActual = pop(cola);
                                /* En vez de enviar un cliente_t se envia solo su id y su
                                * tiempo de espera. Y si es vip o no */
                                int clienteEnviar[3] = {clienteActual->id,
                                                        clienteActual->prioritario,
                                                        clienteActual->tiempoEspera};
                                fprintf(stderr, "CLP_CANP Cliente %d (vip=%d) se enviara a caja %d\n",
                                        clienteActual->id, clienteActual->prioritario, cajaLibre + 1);
                                clientesEnCola--;
                                // CLP_CANP--;
                                MPI_Request request;
                                MPI_Isend(clienteEnviar,
                                        3,
                                        MPI_INT,
                                        cajaLibre + 1,
                                        TAG_CLIENTE,
                                        comm,
                                        &request);
                                estaTrabajando[cajaLibre] = 1;
                            /*}*//*
                             else if (requestsCajaLibre[cajaLibre] != MPI_REQUEST_NULL) {

                                estaOcupada[cajaLibre] = 0;
                                MPI_Cancel(&requestsCajaLibre[cajaLibre]);
                            }
                            */
                        } else if (cola->no_prioritarios->n > 0) {
                            cliente_t *clienteActual = popNP(cola);
                            /* En vez de enviar un cliente_t se envia solo su id y su
                            * tiempo de espera. Y si es vip o no */
                            int clienteEnviar[3] = {clienteActual->id,
                                                    clienteActual->prioritario,
                                                    clienteActual->tiempoEspera};
                            fprintf(stderr, "CLNP_CANP Cliente %d (vip=%d) se enviara a caja %d\n",
                                    clienteActual->id, clienteActual->prioritario, cajaLibre + 1);
                            clientesEnCola--;
                            // CLNP_CANP--;
                            MPI_Request request;
                            MPI_Isend(clienteEnviar,
                                    3,
                                    MPI_INT,
                                    cajaLibre + 1,
                                    TAG_CLIENTE,
                                    comm,
                                    &request);
                            estaTrabajando[cajaLibre] = 1;
                        } else {
                            /* Se sigue esperando a que una caja adecuada quede libre */
                        }
                    }
                }
            } else sleep(1);
            /* 3. Se comprueba si alguna caja ha terminado de atender */
            for (i = 1; i < nCajas + 1; i++) {
                if (estaOcupada[i - 1] && estaTrabajando[i - 1]) {
                    /* Se pregunta si ha terminado */
                    if (!seHaPreguntado[i - 1]) {
                        // fprintf(stderr, "%d: Compruebo si caja %d ha terminado\n",
                        //         rank, i);
                        MPI_Irecv(&clienteTerminado,
                              1,
                              MPI_INT,
                              i,
                              TAG_CATENDIDO,
                              comm,
                              &requestsCAtendido[i - 1]);
                        seHaPreguntado[i - 1] = 1;
                    } else {
                        int haTerminado;
                        MPI_Test(&requestsCAtendido[i - 1],
                                 &haTerminado,
                                 MPI_STATUS_IGNORE);
                        if (haTerminado) {
                            estaOcupada[i - 1]    = 0;
                            estaTrabajando[i - 1] = 0;
                            seHaPreguntado[i - 1] = 0;
                            if (esPrioritaria[i - 1]) cPrioritariasOcupadas--;
                            cliente_t *cRenacido = crearCliente();
                            cRenacido->id = clienteTerminado;
                            push(cola, cRenacido);
                            fprintf(stderr, "El cliente %d vuelve a la cola (vip=%d)\n",
                                    cRenacido->id, cRenacido->prioritario);
                            clientesEnCola++;
                        }
                    }
                }
            }
        }
        /* Se libera la memoria */
        free(requestsCajaLibre);
        free(requestsCAtendido);
        free(estaOcupada);
        free(estaTrabajando);
        free(seHaPreguntado);
        free(esPrioritaria);
        liberarColaPrioridad(cola);

        /* Se confirma que todos los nodos estan esperando */
        int procListo;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&procListo,
                     1,
                     MPI_INT,
                     i,
                     TAG_CAJA_LIBRE,
                     comm,
                     MPI_STATUS_IGNORE);
        }
        /* Se envia una señal de terminacion a los nodos */
        int terminar = 1;
        for (int i = 1; i < size; i++) {
            MPI_Send(&terminar,
                    1,
                    MPI_INT,
                    i,
                    TAG_TERMINAR,
                    comm);
        }
        fprintf(stderr, "%d: No hay clientes en cola\n", rank);
        MPI_Barrier(comm); /* Se espera a que la comunicacion haya terminado */

    } else {
    /* ESCLAVO */
        /*cliente_t *clienteComm = (cliente_t *) malloc(sizeof(cliente_t));*/
        int clienteRecibido[3] = {-1, -1, -1};
        MPI_Request request;
        MPI_Status status;
        while (1) {
            /* 1. Avisar de que estamos libres */
            // fprintf(stderr, "%d: Estoy libre\n", rank);
            MPI_Send(&rank, /* Se podria enviar cualquier int */
                      1,
                      MPI_INT,
                      0,
                      TAG_CAJA_LIBRE,
                      comm);
            /* 2. Recibir un cliente del maestro */
            MPI_Recv(clienteRecibido,
                     3,
                     MPI_INT,
                     0,
                     MPI_ANY_TAG,
                     comm,
                     &status);
            /* Se comprueba si se ha recibido un cliente */
            if (status.MPI_TAG == TAG_CLIENTE) {
                fprintf(stderr, "%d: El cliente %d (vip=%d) esperara %d seg.\n",
                    rank, clienteRecibido[0], clienteRecibido[1], clienteRecibido[2]);
                sleep(clienteRecibido[2]);
                fprintf(stderr, "%d: El cliente %d ha terminado.\n",
                        rank, clienteRecibido[0]);
            } else if (status.MPI_TAG == TAG_TERMINAR) {
                fprintf(stderr, "%d: Se cierra la caja por orden del maestro\n",
                                rank);
                MPI_Barrier(comm); /* Se espera a que la comunicacion haya terminado */
                break;
            }
            /* 3. Avisar de que el cliente ha terminado */
            // fprintf(stderr, "%d: He terminado\n", rank);
            MPI_Send(&clienteRecibido[0],
                     1,
                     MPI_INT,
                     0,
                     TAG_CATENDIDO,
                     comm);
        }
    }
    MPI_Finalize();

    return 0;
}

