/*
 *  kernel/kernel.c
 *
 *  Minikernel. Versión 1.0
 *
 *  Fernando Pérez Costoya
 *
 */

/*
 *
 * Fichero que contiene la funcionalidad del sistema operativo
 *
 */
#include <string.h>
#include "include/kernel.h"	/* Contiene defs. usadas por este modulo */

/*
 *
 * Funciones relacionadas con la tabla de procesos:
 *	iniciar_tabla_proc buscar_BCP_libre
 *
 */

/*
 * Función que inicia la tabla de procesos
 */
static void iniciar_tabla_proc(){
	int i;

	for (i=0; i<MAX_PROC; i++)
		tabla_procs[i].estado=NO_USADA;
}

/*
 * Función que busca una entrada libre en la tabla de procesos
 */
static int buscar_BCP_libre(){
	int i;

	for (i=0; i<MAX_PROC; i++)
		if (tabla_procs[i].estado==NO_USADA)
			return i;
	return -1;
}

/*
 *
 * Funciones que facilitan el manejo de las listas de BCPs
 *	insertar_ultimo eliminar_primero eliminar_elem
 *
 * NOTA: PRIMERO SE DEBE LLAMAR A eliminar Y LUEGO A insertar
 */

/*
 * Inserta un BCP al final de la lista.
 */
static void insertar_ultimo(lista_BCPs *lista, BCP * proc){
	if (lista->primero==NULL)
		lista->primero= proc;
	else
		lista->ultimo->siguiente=proc;
	lista->ultimo= proc;
	proc->siguiente=NULL;
}

/*
 * Elimina el primer BCP de la lista.
 */
static void eliminar_primero(lista_BCPs *lista){

	if (lista->ultimo==lista->primero)
		lista->ultimo=NULL;
	lista->primero=lista->primero->siguiente;
}

/*
 * Elimina un determinado BCP de la lista.
 */
static void eliminar_elem(lista_BCPs *lista, BCP * proc){
	BCP *paux=lista->primero;

	if (paux==proc)
		eliminar_primero(lista);
	else {
		for ( ; ((paux) && (paux->siguiente!=proc));
			paux=paux->siguiente);
		if (paux) {
			if (lista->ultimo==paux->siguiente)
				lista->ultimo=paux;
			paux->siguiente=paux->siguiente->siguiente;
		}
	}
}

/*
 *
 * Funciones relacionadas con la planificacion
 *	espera_int planificador
 */

/*
 * Espera a que se produzca una interrupcion
 */
static void espera_int(){
	int nivel;

	printk("-> NO HAY LISTOS. ESPERA INT\n");

	/* Baja al mínimo el nivel de interrupción mientras espera */
	nivel=fijar_nivel_int(NIVEL_1);
	halt();
	fijar_nivel_int(nivel);
}

/*
 * Función de planificacion que implementa un algoritmo FIFO.
 */
static BCP * planificador(){
	while (lista_listos.primero==NULL)
		espera_int();		/* No hay nada que hacer */
	return lista_listos.primero;
}

/*
 *
 * Funcion auxiliar que termina proceso actual liberando sus recursos.
 * Usada por llamada terminar_proceso y por rutinas que tratan excepciones
 *
 */
static void liberar_proceso(){
	BCP * p_proc_anterior;

	liberar_imagen(p_proc_actual->info_mem); /* liberar mapa */

	p_proc_actual->estado=TERMINADO;
	eliminar_primero(&lista_listos); /* proc. fuera de listos */

	/* Realizar cambio de contexto */
	p_proc_anterior=p_proc_actual;
	p_proc_actual=planificador();

	printk("-> C.CONTEXTO POR FIN: de %d a %d\n",
			p_proc_anterior->id, p_proc_actual->id);

	liberar_pila(p_proc_anterior->pila);
	cambio_contexto(NULL, &(p_proc_actual->contexto_regs));
        return; /* no debería llegar aqui */
}

/*
 *
 * Funciones relacionadas con el tratamiento de interrupciones
 *	excepciones: exc_arit exc_mem
 *	interrupciones de reloj: int_reloj
 *	interrupciones del terminal: int_terminal
 *	llamadas al sistemas: llam_sis
 *	interrupciones SW: int_sw
 *
 */

/*
 * Tratamiento de excepciones aritmeticas
 */
static void exc_arit(){

	if (!viene_de_modo_usuario())
		panico("excepcion aritmetica cuando estaba dentro del kernel");


	printk("-> EXCEPCION ARITMETICA EN PROC %d\n", p_proc_actual->id);
	liberar_proceso();

        return; /* no debería llegar aqui */
}

/*
 * Tratamiento de excepciones en el acceso a memoria
 */
static void exc_mem(){

	if (!viene_de_modo_usuario())
		panico("excepcion de memoria cuando estaba dentro del kernel");


	printk("-> EXCEPCION DE MEMORIA EN PROC %d\n", p_proc_actual->id);
	liberar_proceso();

        return; /* no debería llegar aqui */
}

/*
 * Tratamiento de interrupciones de terminal
 */
static void int_terminal(){
	char car;

	car = leer_puerto(DIR_TERMINAL);
	printk("-> TRATANDO INT. DE TERMINAL %c\n", car);

        return;
}

/*
 * Tratamiento de interrupciones de reloj
 */
static void int_reloj(){

	printk("-> TRATANDO INT. DE RELOJ\n");
    esperar_bloqueados(); /* Contar tiempo de espera de bloqueados */
    round_robin();

        return;
}

/*
 * Tratamiento de llamadas al sistema
 */
static void tratar_llamsis(){
	int nserv, res;

	nserv=leer_registro(0);
	if (nserv<NSERVICIOS)
		res=(tabla_servicios[nserv].fservicio)();
	else
		res=-1;		/* servicio no existente */
	escribir_registro(0,res);
	return;
}

/*
 * Tratamiento de interrupciuones software
 * Gabriel: modificado para Round Robin
 */
static void int_sw(){
    int nivelInterrupcion;
    BCPptr actual;

	printk("-> TRATANDO INT. SW\n");

    nivelInterrupcion = fijar_nivel_int(NIVEL_3);
    actual = p_proc_actual;
    eliminar_primero(&lista_listos);
    insertar_ultimo(&lista_listos, actual);
    p_proc_actual = planificador();
    fijar_nivel_int(nivelInterrupcion);
    cambio_contexto(&(actual->contexto_regs), &(p_proc_actual->contexto_regs));

	return;
}

/*
 *
 * Funcion auxiliar que crea un proceso reservando sus recursos.
 * Usada por llamada crear_proceso.
 *
 */
static int crear_tarea(char *prog){
	void * imagen, *pc_inicial;
	int error=0;
	int proc;
	BCP *p_proc;

	proc=buscar_BCP_libre();
	if (proc==-1)
		return -1;	/* no hay entrada libre */

	/* A rellenar el BCP ... */
	p_proc=&(tabla_procs[proc]);

	/* crea la imagen de memoria leyendo ejecutable */
	imagen=crear_imagen(prog, &pc_inicial);
	if (imagen)
	{
		p_proc->info_mem=imagen;
		p_proc->pila=crear_pila(TAM_PILA);
		fijar_contexto_ini(p_proc->info_mem, p_proc->pila, TAM_PILA,
			pc_inicial,
			&(p_proc->contexto_regs));
		p_proc->id=proc;
		p_proc->estado=LISTO;

		/* lo inserta al final de cola de listos */
		insertar_ultimo(&lista_listos, p_proc);
		error= 0;
	}
	else
		error= -1; /* fallo al crear imagen */

	return error;
}

/*
 *
 * Rutinas que llevan a cabo las llamadas al sistema
 *	sis_crear_proceso sis_escribir
 *
 */

/*
 * Tratamiento de llamada al sistema crear_proceso. Llama a la
 * funcion auxiliar crear_tarea sis_terminar_proceso
 */
int sis_crear_proceso(){
	char *prog;
	int res;

	printk("-> PROC %d: CREAR PROCESO\n", p_proc_actual->id);
	prog=(char *)leer_registro(1);
	res=crear_tarea(prog);
	return res;
}

/*
 * Tratamiento de llamada al sistema escribir. Llama simplemente a la
 * funcion de apoyo escribir_ker
 */
int sis_escribir()
{
	char *texto;
	unsigned int longi;

	texto=(char *)leer_registro(1);
	longi=(unsigned int)leer_registro(2);

	escribir_ker(texto, longi);
	return 0;
}

/*
 * Tratamiento de llamada al sistema terminar_proceso. Llama a la
 * funcion auxiliar liberar_proceso
 */
int sis_terminar_proceso(){

	printk("-> FIN PROCESO %d\n", p_proc_actual->id);

	liberar_proceso();

        return 0; /* no debería llegar aqui */
}

/* OBTENER_ID_PR
 */
int obtener_id_pr() {
    return p_proc_actual->id;
}

/* DORMIR
 */
int dormir(unsigned int segundos) {
    int n_interr_previo = fijar_nivel_int(NIVEL_3);

    BCPptr dormido = p_proc_actual;

    p_proc_actual->estado = BLOQUEADO;
    p_proc_actual->segundos_dormir = segundos * TICK;

    eliminar_elem(&lista_listos, p_proc_actual);
    insertar_ultimo(&lista_bloqueados, p_proc_actual);

    p_proc_actual = planificador();

    /* Restarar el nivel de interrupcion y realizar el cambio de cntx */
    fijar_nivel_int(n_interr_previo);
    cambio_contexto(&(dormido->contexto_regs), &(p_proc_actual->contexto_regs));

    return 0;
}

/* ESPERAR_BLOQUEADOS
 * Actualiza el tiempo de espera de los procesos de la lista de espera
 */
int esperar_bloqueados() {
    BCPptr actual;
    BCPptr siguiente;
    actual = lista_bloqueados.primero;
    while (actual != NULL) {
        siguiente = actual->siguiente;

        actual->segundos_dormir--;
        if (actual->segundos_dormir <= 0) {
            actual->estado = LISTO;
            eliminar_elem(&lista_bloqueados, actual);
            insertar_ultimo(&lista_listos, actual);
        }

        actual = siguiente;
    }

    return 0;
}

/* CREAR_MUTEX
 * Crea el mutex y devuelve el descriptor de fichero para acceder a el
 */
int crear_mutex(char *nombre, int tipo) {
    mutex m;
    int   posicion; /* Para buscar el hueco libre */
    char *nombreMutex = (char *) leer_registro(1);
    int  tipoMutex    = (int) leer_registro(2);

    int nivelInterrupcion = fijar_nivel_int(NIVEL_1);

    /* Si el nombre es muy largo, se corta a las 7 letras */
    if (strlen(nombreMutex) >= MAX_NOM_MUT) {
        nombreMutex[MAX_NOM_MUT] = '\0';
    }

    if (existe_mutex(nombreMutex) == -1) {
        /* El mutex no existe y se puede crear */
        posicion = buscar_hueco_lista_mutex();
        if (posicion >= 0) { /* Hay un hueco libre */
            /* Se crea el mutex para meterlo en la lista */
            strcpy(m.nombre, nombreMutex);
            m.ocupado = 1;
            m.recursivo = tipoMutex;
            m.proc_utilizando = NULL;
            m.num_proc_esperando = 0;
            /*m.lista_proc_esperando*/
            m.num_bloqueos = 0;

            lista_mutex[posicion] = &m;

            int fd = abrir_mutex(nombreMutex);
            fijar_nivel_int(nivelInterrupcion);

            return fd;
        } else {
            /* No quedan mutex disponibles. El proceso se bloquea */
            BCPptr actual = p_proc_actual;

            actual->estado = BLOQUEADO;
            eliminar_elem(&lista_listos, actual);
            insertar_ultimo(&lista_bloq_mutex, actual);
            p_proc_actual = planificador();

            fijar_nivel_int(nivelInterrupcion);
            cambio_contexto(&(actual->contexto_regs),
                    &(p_proc_actual->contexto_regs));

            return -1;
        }
    } else {
        /* Ya existe un mutex con el mismo nombre */
        fijar_nivel_int(nivelInterrupcion);

        return -1;
    }

    return 0;
}

int existe_mutex(char *nombre) {
    int i;
    for (i = 0; i < NUM_MUT; i++) {
        if (!strcmp(lista_mutex[i]->nombre, nombre)) {
            return i;
        }
    }

    return -1;
}

int buscar_hueco_lista_mutex() {
    int i;
    for (i = 0; i < NUM_MUT; i++) {
        if (lista_mutex[i]->ocupado == 0) {
            return i;
        }
    }

    return -1;
}

int abrir_mutex(char *nombre) {
    int posicion;
    char *nombreMutex = (char *) leer_registro(1);

    int nivelInterrupcion = fijar_nivel_int(NIVEL_1);

    if (p_proc_actual->num_descriptores < NUM_MUT_PROC) {
        posicion = existe_mutex(nombreMutex);
        if (posicion >= 0) {
            p_proc_actual->descriptores[p_proc_actual->num_descriptores] =
                posicion;
            p_proc_actual->num_descriptores++;

            fijar_nivel_int(nivelInterrupcion);

            return posicion;
        } else {
            /* No existe el mutex */
            fijar_nivel_int(nivelInterrupcion);
            return -1;
        }
    } else {
        /* El proceso no tiene fd libres */
        fijar_nivel_int(nivelInterrupcion);
        return -1;
    }

    return 0;
}

int lock(unsigned int mutexid) {
    int i;
    int existe;
    unsigned int id;
    int nivelInterrupcion;
    int nivelInterrupcion2;
    BCPptr actual;

    id = (unsigned int) leer_registro(1);
    nivelInterrupcion = fijar_nivel_int(NIVEL_1);

    existe = 0;
    for (i = 0; i < NUM_MUT_PROC; i++) {
        if (p_proc_actual->descriptores[i] == id) {
            existe = 1;
        }
    }
    if (!existe) {
        /* El mutex no existe */
        fijar_nivel_int(nivelInterrupcion);

        return -1;
    }

    while (lista_mutex[id]->proc_utilizando != p_proc_actual &&
           lista_mutex[id]->proc_utilizando != NULL) {
        p_proc_actual->estado = BLOQUEADO;
        nivelInterrupcion2 = fijar_nivel_int(NIVEL_3);
        eliminar_elem(&lista_listos, p_proc_actual);
        insertar_ultimo(&lista_mutex[id]->lista_proc_esperando,
                p_proc_actual);
        lista_mutex[id]->num_proc_esperando++;
        fijar_nivel_int(nivelInterrupcion2);

        actual = p_proc_actual;
        p_proc_actual = planificador();
        cambio_contexto(&(actual->contexto_regs),
                &(p_proc_actual->contexto_regs));
    }
    lista_mutex[id]->proc_utilizando = p_proc_actual;

    if (lista_mutex[id]->recursivo == NO_RECURSIVO &&
        lista_mutex[id]->num_bloqueos == 1) {
        /* El mutex no es recursivo */
        return -1; /* TODO: return -2 */
    }
    /* Si es recursivo o tiene mas bloqueos, se anota */
    lista_mutex[id]->num_bloqueos++;
    fijar_nivel_int(nivelInterrupcion);

    return 0;
}

int unlock(unsigned int mutexid) {
    int i;
    int existe;
    unsigned int id;
    int nivelInterrupcion;
    int nivelInterrupcion2;
    BCPptr siguiente;


    id = leer_registro(1);
    nivelInterrupcion = fijar_nivel_int(NIVEL_1);

    existe = 0;
    for (i = 0; i < NUM_MUT_PROC; i++) {
        if (p_proc_actual->descriptores[i] == id) {
            existe = 1;
        }
    }

    if (!existe) {
        fijar_nivel_int(nivelInterrupcion);
        return -1;
    }

    if (lista_mutex[id]->proc_utilizando != p_proc_actual) {
        fijar_nivel_int(nivelInterrupcion);
        return -1; /* TODO: return -2 */
    }

    lista_mutex[id]->num_bloqueos--;
    if (lista_mutex[id]->num_bloqueos != 0) {
        /* El mutex es recursivo */
        fijar_nivel_int(nivelInterrupcion);
        return 0;
    }
    lista_mutex[id]->proc_utilizando = NULL;

    if (lista_mutex[id]->lista_proc_esperando.primero == NULL) {
        /* No quedan procesos esperando al mutex */
        fijar_nivel_int(nivelInterrupcion);
        return -1; /* TODO: return -3 */
    }

    siguiente = lista_mutex[id]->lista_proc_esperando.primero;
    siguiente->estado = LISTO;
    nivelInterrupcion2 = fijar_nivel_int(NIVEL_3);
    insertar_ultimo(&lista_listos, siguiente);
    fijar_nivel_int(nivelInterrupcion2);
    lista_mutex[id]->proc_utilizando = siguiente;
    fijar_nivel_int(nivelInterrupcion);

    return 0;
}

int cerrar_mutex(unsigned int mutexid) {
    unsigned int id;
    int nivelInterrupcion;
    int nivelInterrupcion2;
    int i;
    int existe;
    mutex *m;
    BCPptr proc;

    id = (unsigned int) leer_registro(1);
    nivelInterrupcion = fijar_nivel_int(NIVEL_1);

    existe = 0;
    for (i = 0; i < NUM_MUT_PROC; i++) {
        if (p_proc_actual->descriptores[i] == id) {
            existe = 1;
        }
    }
    if (!existe) {
        fijar_nivel_int(nivelInterrupcion);

        return -1;
    }

    m = lista_mutex[id];
    if (m->proc_utilizando == p_proc_actual) {
        m->proc_utilizando = NULL;
        m->num_proc_esperando = 0;
    }
    p_proc_actual->num_descriptores--;
    p_proc_actual->descriptores[i] = -1;
    m->ocupado = 0;

    if (m->recursivo == RECURSIVO) {
        /* Si es recursivo se liberan todos los procesos asociados */
        if (liberar_todos_mutex(m) == -1) {
            return -1;
        }
    }
    m->num_bloqueos = 0;
    if (lista_bloq_mutex.primero != NULL) {
        proc = lista_bloq_mutex.primero;
        nivelInterrupcion2 = fijar_nivel_int(NIVEL_3);
        eliminar_primero(&lista_bloq_mutex);
        insertar_ultimo(&lista_listos, proc);
        proc->estado = LISTO;
        fijar_nivel_int(nivelInterrupcion2);
    }
    fijar_nivel_int(nivelInterrupcion);

    return 0;
}


int liberar_todos_mutex(mutex *m) {
    BCPptr actual;
    BCPptr siguiente;
    int nivelInterrupcion;
    int ok;

    actual = m->lista_proc_esperando.primero;
    siguiente = NULL;
    while(m->lista_proc_esperando.primero != NULL) {
        siguiente = actual->siguiente;
        nivelInterrupcion = fijar_nivel_int(NIVEL_3);
        eliminar_primero(&m->lista_proc_esperando);
        insertar_ultimo(&lista_listos, siguiente);
        actual->estado = LISTO;
        fijar_nivel_int(nivelInterrupcion);
        actual = siguiente;
    }

    ok = 0;
    if (m->lista_proc_esperando.primero != NULL) {
        ok = -1;
    }

    return ok;
}

/* ROUND ROBIN */
int round_robin() {
    ticksRestantes--;
    if (ticksRestantes <= 0) {
        activar_int_SW();
    }

    return 0;
}

/*
 *
 * Rutina de inicialización invocada en arranque
 *
 */
int main(){
	/* se llega con las interrupciones prohibidas */

	instal_man_int(EXC_ARITM, exc_arit);
	instal_man_int(EXC_MEM, exc_mem);
	instal_man_int(INT_RELOJ, int_reloj);
	instal_man_int(INT_TERMINAL, int_terminal);
	instal_man_int(LLAM_SIS, tratar_llamsis);
	instal_man_int(INT_SW, int_sw);

	iniciar_cont_int();		/* inicia cont. interr. */
	iniciar_cont_reloj(TICK);	/* fija frecuencia del reloj */
	iniciar_cont_teclado();		/* inici cont. teclado */

	iniciar_tabla_proc();		/* inicia BCPs de tabla de procesos */

	/* crea proceso inicial */
	if (crear_tarea((void *)"init")<0)
		panico("no encontrado el proceso inicial");

	/* activa proceso inicial */
	p_proc_actual=planificador();
	cambio_contexto(NULL, &(p_proc_actual->contexto_regs));
	panico("S.O. reactivado inesperadamente");
	return 0;
}
