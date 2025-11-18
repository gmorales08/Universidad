#include <stdlib.h>
#include "cola.h"

cola_t *crearCola() {
    cola_t *cola = (cola_t *) malloc(sizeof(cola_t));
    cola->n = 0;
    cola->primero = NULL;
    cola->ultimo  = NULL;

    return cola;
}

colaPrioridad_t *crearColaPrioridad() {
    colaPrioridad_t *colaP = (colaPrioridad_t *) malloc(sizeof(colaPrioridad_t));
    colaP->prioritarios    = crearCola();
    colaP->no_prioritarios = crearCola();

    return colaP;
}

cliente_t *pop(colaPrioridad_t *cola) {
    if (cola->prioritarios->n > 0) {
        cola->prioritarios->n--;
        cliente_t *clienteASacar;
        clienteASacar = cola->prioritarios->primero;
        cola->prioritarios->primero =
            cola->prioritarios->primero->siguiente;

        return clienteASacar;
    } else if (cola->no_prioritarios->n > 0) {
        cola->no_prioritarios->n--;
        cliente_t *clienteASacar;
        clienteASacar = cola->no_prioritarios->primero;
        cola->no_prioritarios->primero =
            cola->no_prioritarios->primero->siguiente;

        return clienteASacar;
    } else return NULL;
}

cliente_t *popNP(colaPrioridad_t *cola) {
    if (cola->no_prioritarios->n > 0) {
        cola->no_prioritarios->n--;
        cliente_t *clienteASacar;
        clienteASacar = cola->no_prioritarios->primero;
        cola->no_prioritarios->primero =
            cola->no_prioritarios->primero->siguiente;

        return clienteASacar;
    } else return NULL;
}

int push(colaPrioridad_t *cola, cliente_t *cliente) {
    cola_t *colaDest;
    if (cliente->prioritario) {
        colaDest = cola->prioritarios;
    } else {
        colaDest = cola->no_prioritarios;
    }

    if (colaDest->n == 0) {
        colaDest->primero = cliente;
        colaDest->ultimo = cliente;
    } else {
        colaDest->ultimo->siguiente = cliente;
        colaDest->ultimo = cliente;
    }
    cliente->siguiente = NULL; /* Por si el cliente a vuelto a la cola */
    colaDest->n++;

    return 0;
}

int liberarColaPrioridad(colaPrioridad_t *cola) {
    if (cola == NULL) return 1;
    cliente_t *cliente;
    cliente_t *aux;
    cliente = cola->prioritarios->primero;
        while (cliente != NULL) {
        aux = cliente;
        cliente = cliente->siguiente;
        free(aux);
    }
    free(cola->prioritarios);
    cliente = cola->no_prioritarios->primero;
        while (cliente != NULL) {
        aux = cliente;
        cliente = cliente->siguiente;
        free(aux);
    }
    free(cola->no_prioritarios);
    free(cola);

    return 0;
}
