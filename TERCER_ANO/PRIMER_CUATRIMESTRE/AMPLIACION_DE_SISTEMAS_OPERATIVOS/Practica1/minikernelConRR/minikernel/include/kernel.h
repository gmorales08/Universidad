/*
 *  minikernel/include/kernel.h
 *
 *  Minikernel. Versión 1.0
 *
 *  Fernando Pérez Costoya
 *
 */

/*
 *
 * Fichero de cabecera que contiene definiciones usadas por kernel.c
 *
 *      SE DEBE MODIFICAR PARA INCLUIR NUEVA FUNCIONALIDAD
 *
 */

#ifndef _KERNEL_H
#define _KERNEL_H

#include "const.h"
#include "HAL.h"
#include "llamsis.h"

#define NO_RECURSIVO 0
#define RECURSIVO 1
/*
 *
 * Definicion del tipo que corresponde con el BCP.
 * Se va a modificar al incluir la funcionalidad pedida.
 *
 */
typedef struct BCP_t *BCPptr;

typedef struct BCP_t {
        int id;				/* ident. del proceso */
        int estado;			/* TERMINADO|LISTO|EJECUCION|BLOQUEADO*/
        int segundos_dormir; /* Segs. que debe dormir en BLOQUEADO */
        contexto_t contexto_regs;	/* copia de regs. de UCP */
        void * pila;			/* dir. inicial de la pila */
	BCPptr siguiente;		/* puntero a otro BCP */
	void *info_mem;			/* descriptor del mapa de memoria */
    /* Para mutex */
    int descriptores[NUM_MUT_PROC]; /* fd de los mutex en uso */
    int num_descriptores;
} BCP;

/*
 *
 * Definicion del tipo que corresponde con la cabecera de una lista
 * de BCPs. Este tipo se puede usar para diversas listas (procesos listos,
 * procesos bloqueados en semáforo, etc.).
 *
 */

typedef struct{
	BCP *primero;
	BCP *ultimo;
} lista_BCPs;


/*
 * Variable global que identifica el proceso actual
 */

BCP * p_proc_actual= (BCP *) NULL;

/*
 * Variable global que representa la tabla de procesos
 */

BCP tabla_procs[MAX_PROC];

/*
 * Variable global que representa la cola de procesos listos
 */
lista_BCPs lista_listos= {(BCP *) NULL, (BCP *) NULL};

/*
 *
 * Definición del tipo que corresponde con una entrada en la tabla de
 * llamadas al sistema.
 *
 */
typedef struct{
	int (*fservicio)();
} servicio;

/* Definicion del tipo mutex */
typedef struct mutex_t {
    char nombre[MAX_NOM_MUT];
    int ocupado; /* 0 si esta libre, 1 si esta ocupado */
    int recursivo; /* NO_RECURSIVO/RECURSIVO (0/1) */
    BCPptr proc_utilizando; /* Proceso que esta usando el mutex */
    int num_proc_esperando;
    lista_BCPs lista_proc_esperando;
    int num_bloqueos; /* Numero total de bloqueos */
} mutex;



/*
 * Prototipos de las rutinas que realizan cada llamada al sistema
 */
int sis_crear_proceso();
int sis_terminar_proceso();
int sis_escribir();

int obtener_id_pr();
int dormir(unsigned int segundos);
int esperar_bloqueados();

int crear_mutex(char *nombre, int tipo);
/* Busca el mutex en la lista y devuelve su posicion, en caso que no */
/* exista devuelve -1 */
int existe_mutex(char *nombre);
/* Busca un hueco libre en la lista de mutex y devuelve su posicion */
int buscar_hueco_lista_mutex();
/* Busca un mutex por el nombre especificado y devuelve su descriptor */
int abrir_mutex(char *nombre);
int lock(unsigned int mutexid);
int unlock(unsigned int mutexid);
int cerrar_mutex(unsigned int mutexid);
int liberar_todos_mutex(mutex *m);

/*
 * Variable global que contiene las rutinas que realizan cada llamada
 */
servicio tabla_servicios[NSERVICIOS]={
                    {sis_crear_proceso},
					{sis_terminar_proceso},
					{sis_escribir},
                    {obtener_id_pr},
                    {dormir},
                    {crear_mutex},
                    {abrir_mutex},
                    {lock},
                    {unlock},
                    {cerrar_mutex}
};

/* Lista que contiene los procesos bloqueados */
lista_BCPs lista_bloqueados = {};


mutex *lista_mutex[NUM_MUT]; /* Lista que contiene los mutex creados */
lista_BCPs lista_bloq_mutex = {}; /* Lista de proc. bloqueados por un mutex */

/* ROUND ROBIN */
int ticksRestantes; /* Ticks que quedan para acabar la rodaja */
int round_robin();

#endif /* _KERNEL_H */

