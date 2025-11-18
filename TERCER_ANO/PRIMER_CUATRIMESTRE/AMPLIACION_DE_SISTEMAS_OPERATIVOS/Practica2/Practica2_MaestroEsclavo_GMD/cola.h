#ifndef COLA_H
#define COLA_H

#include "cliente.h"

typedef struct cola {
    int n;
    cliente_t *primero;
    cliente_t *ultimo;
} cola_t;

typedef struct colaPrioridad {
    cola_t *prioritarios;
    cola_t *no_prioritarios;
} colaPrioridad_t;

cola_t *crearCola();
colaPrioridad_t *crearColaPrioridad();
cliente_t *pop(colaPrioridad_t *cola);
cliente_t *popNP(colaPrioridad_t *cola);
int push(colaPrioridad_t *cola, cliente_t *cliente);
int liberarColaPrioridad(colaPrioridad_t *cola);

#endif /* COLA_H */
