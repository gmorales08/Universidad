#ifndef CLIENTE_H
#define CLIENTE_H

typedef struct cliente {
    int id;
    int prioritario;
    int tiempoEspera;
    struct cliente *siguiente;
} cliente_t;

cliente_t *crearCliente();
int esperar(cliente_t *cliente);

#endif /* CLIENTE_H */
