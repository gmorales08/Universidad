#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "cliente.h"

cliente_t *crearCliente() {
    cliente_t *cliente = (cliente_t *) malloc(sizeof(cliente_t));
    cliente->id = -1;
    cliente->prioritario = (rand() % 4 == 0); /* 25% de ser vip */
    if (cliente->prioritario) cliente->tiempoEspera = rand() % 11 + 10;
    else cliente->tiempoEspera = rand() % 6 + 5;
    cliente->siguiente = NULL;

    return cliente;
}

int esperar(cliente_t *cliente) {
    fprintf(stderr, "El cliente %d esperara %d seg.\n",
            cliente->id, cliente->tiempoEspera);
    sleep(cliente->tiempoEspera);
    fprintf(stderr, "El cliente %d ha terminado de esperar\n",
            cliente->id);
    return 0;
}
