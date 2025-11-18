/******************************************************************************
 * SISTEMAS OPERATIVOS                                                        *
 * PRACTICA 2 - SCRIPTING                                                     *
 *                                                                            *
 * GABRIEL MORALES DATO                                                       *
 * ***************************************************************************/
/* ****************************************************************************
 * MINISHELL                                                                  *
 *                                                                            *
 * Objetivo:                                                                  *
 *   1. Leer de la entrada estandar uno o varios comandos separados por '|'   *
 *      - Se permiten redirecciones: <, >, >&                                 *
 *      - Se permite la ejecucion en background si el comando termina en '&'  *
 *        + Mostrar el PID del proceso por el que se espera y no bloquear     *
 *          la ejecucion.                                                     *
 *   2. Analizar los comandos con la libreria parser                          *
 *   3. Ejecutar los comandos                                                 *
 *      - Comunicarlos con pipes                                              *
 *   4. Esperar a que terminen los mandatos y repetir el proceso.             *
 *                                                                            *
 *  Anotaciones:                                                              *
 *      - Si no se introduce ningun comando o se ejecuta en background, se    *
 *        vuelve a mostrar el prompt.                                         *
 *      - Si el mandato no existe se indica: "mandato: No se encuentra el     *
 *        mandato"                                                            *
 *      - Si sucede algun error al abrir un fichero, se indica: “fichero:     *
 *        Error. Descripcion del error”                                       *
 *      - Ni minishell ni los procesos en background deben finalizar al       *
 *        recibir las senales desde teclado SIGINT (ctrl+c) y SIGQUIT (ctrl + *
 *        \), mientras que los procesos que se lancen deben actuar ante ellas *
 *****************************************************************************/

#define _POSIX_SOURCE /* Si no pongo esto, gcc da error con kill */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>    /* Para las senales */
#include <sys/types.h> /* Para pid_t */
#include <sys/wait.h>  /* Para wait */
#include <unistd.h>    /* Para execvp, access */
#include <sys/stat.h>  /* Para ficheros */
#include <fcntl.h>     /* Para ficheros */
#include <errno.h>     /* Para identificar errores */
#include "myshell.h"

/* Constantes */
#define TAMANO_BUFFER 80 /* Limite de caracteres que se leeran en stdin */
#define SALIDA 0         /* Extremos de los pipes. Los nombro de esta forma: */
#define ENTRADA 1        /* Proceso1 -> |==TUBERIA==| -> Proceso2 */
                         /*     ENTRADA-^    SALIDA-^          */

/* Variables globales */
tline   *entradaUsuario;   /* Entrada del usuario */
pid_t   *pids;             /* Lista de PIDs */
trabajo *cabeza;           /* Primer trabajo de la lista */
trabajo *cola;             /* Ultimo trabajo de la lista */
int     idbg;              /* Numero que identifica la posicion en la lista de bg */


int main(void) {
    cabeza = NULL;
    cola   = NULL;
    idbg   = 1;
    printf("msh> ");
    while (1) {

        /* Ignorar las senales SIGINT Y SIGQUIT por defecto */
        signal(SIGINT, SIG_IGN);
        signal(SIGQUIT, SIG_IGN);
        /* 1. Leer y validar la entrada del usuario */
        entradaUsuario = solicitarEntrada();
        /* 2. Ejecutar la secuencia de comandos */
        ejecutarLinea(entradaUsuario);
        printf("msh> ");
    }

    return 0;
}


tline *solicitarEntrada() {
    tline *linea;
    char bufferEntrada[TAMANO_BUFFER];
    int valido;
    int i;

    /* Se comprueba si el comando es valido, y si esta permitido por la shell*/
    valido = 1;
    do {
        if (valido == 0) { /* Para no imprimir varias veces el prompt */
            printf("msh> ");
        }
        valido = 0;
        if (fgets(bufferEntrada, TAMANO_BUFFER, stdin) != NULL) {
            linea = tokenize(bufferEntrada);
            /* Si el comando esta vacio, no se pasa a comprobarlo, para evitar
             * un segmentation fault */
            if (linea->ncommands > 0) {
                for (i = 0; i < linea->ncommands; i++) {
                    if (!strcmp(*(linea->commands->argv), "cd")) {
                        cd(linea->commands->argc, linea->commands->argv);
                        break;
                    } else if (!strcmp(*(linea->commands->argv), "jobs")) {
                        jobs(linea->commands->argc, cabeza);
                        break;
                    } else if (!strcmp(*(linea->commands->argv), "fg")) {
                        fg(linea->commands->argc);
                    } else {
                        valido = comprobarComando(linea->commands + i);
                    }
                    if (valido == 0) {
                        break;
                    }
                }
            }
        /* Modificacion para extraordinaria */
        /* Si se pulsa Ctrl+D fgets devuelve NULL y se sale de la terminal */
        } else {
            printf("\n");
            linea = NULL; /* TODO: he añadido esta linea para probar */
            exit(0);
        }
    } while (valido == 0);

    return linea;
}


int comprobarComando(tcommand *comando) {
    int valido;

    valido = 0;
    if (comando->argc > 0) { /* Se ha introducido al menos un comando */
        if (comando->filename != NULL) { /* El comando existe */
            valido = 1;
        } else {
            printf("msh> %s: No se encuentra el mandato\n",*(comando->argv));
        }
    }

    return valido;
}

void ejecutarLinea(tline *linea) {
    pid_t   pid;
    int     status;
    int     numeroComandos; /* Numero de comandos/hijos */
    int     i;
    int     j;
    int     **tub;          /* Vector de tuberias */
    int     fdEntrada;      /* Redireccion de entrada */
    int     fdSalida;       /* Redireccion de salida */
    int     fdError;        /* Redireccion de error */
    int     bg;             /* Para saber si la ejecucion es en background */
    trabajo *tb;            /* Para los procesos en background */

    numeroComandos = linea->ncommands;
    bg             = linea->background;
    pids           = (pid_t *) malloc((numeroComandos) * sizeof(pid_t));
    tb             = NULL;

    /* Creacion de las tuberias */
    tub            = (int **) malloc((numeroComandos - 1) * sizeof(int *));
    for (i = 0; i < numeroComandos - 1; i++) {
        tub[i] = malloc(2 * sizeof(int));
    }
    for (i = 0; i < numeroComandos - 1; i++) {
        if (pipe(tub[i]) == -1) {
            printf("Error al crear la tuberia %d\n", i);
            exit(1);
        }

    }

    /* Bucle de creacion de procesos */
    for (i = 0; i < numeroComandos; i++) {
        pid     = fork();
        pids[i] = pid;

        /* Si se ejecuta en segundo plano, se crea un struct de trabajo */
        if (linea->background == 1 && pid != 0) {
            tb            = (trabajo *) malloc(sizeof(trabajo));
            tb->pid       = pids[i];
            tb->bg        = bg;
            tb->comando   = (linea->commands + i);
            tb->siguiente = NULL;
            if (cabeza == NULL) {
                cabeza   = tb;
                cola     = tb;
                tb->idbg = idbg;
            } else {
                cola->siguiente = tb;
                cola            = tb;
                tb->idbg        = idbg++;
            }

            printf("[%d] %d\n", tb->idbg, tb->pid);
        }

        if (pid == -1) {
            printf("Error al crear el proceso %d\n", i);
            exit(1);
        } else if (pid == 0) {
            /* Proceso hijo */
            /* 1. Cerrar las tuberias no necesarias */
            /* Excepcion 1: es el primer proceso y hay mas de un comando */
            if (i == 0 && numeroComandos > 1) {
                for (j = 1; j < numeroComandos - 1; j++) {
                        close(tub[j][SALIDA]);
                        close(tub[j][ENTRADA]);
                }
            /*Excepcion 2: es el ultimo proceso */
            } else if (i == numeroComandos - 1) {
                for (j = 0; j < numeroComandos - 2; j++) {
                    close(tub[j][SALIDA]);
                    close(tub[j][ENTRADA]);
                }
            /* Caso normal: es un proceso intermedio y hay mas de un comando */
            } else if (numeroComandos > 1) {
                for (j = 0; j < numeroComandos - 1; j++) {
                    if (j != i && j != i - 1) {
                        close(tub[j][SALIDA]);
                        close(tub[j][ENTRADA]);
                    }
                }
            }

            /* 2. Conectar las tuberias y hacer redirecciones */
            /* Asignar las redirecciones */
            if (linea->redirect_input != 0 && i == 0) {
                fdEntrada = open(linea->redirect_input, O_RDONLY, 0777);
                if (fdEntrada == -1) {
                    printf("Error al abrir el fichero de entrada\n");
                    exit(1);
                } else {
                    dup2(fdEntrada, STDIN_FILENO);
                    close(fdEntrada);
                }
            }

            if (linea->redirect_output != 0 && i == numeroComandos - 1) {
                fdSalida = open(linea->redirect_output, O_WRONLY | O_CREAT | O_TRUNC, 0777);
                if (fdSalida == -1) {
                    printf("Error al abrir el fichero de salida\n");
                    exit(1);
                } else {
                    dup2(fdSalida, STDOUT_FILENO);
                    close(fdSalida);
                }
            }

            if (linea->redirect_error != 0 && i == numeroComandos - 1) {
                fdError = open(linea->redirect_error, O_WRONLY | O_CREAT | O_TRUNC, 0777);
                if (fdError == -1) {
                    printf("Error al abrir el fichero de error\n");
                    exit(1);
                } else {
                    dup2(fdError, STDERR_FILENO);
                    close(fdError);
                }
            }

            /* Excepcion 1: solo hay un proceso */
            if (numeroComandos == 1) {
                ejecutarComando(linea->commands + 0);
            /* Excepcion 2: hay dos comandos (una tuberia) */
            } else if (numeroComandos == 2) {
                if (i == 0) {
                    close(tub[0][SALIDA]);
                    if (dup2(tub[0][ENTRADA], STDOUT_FILENO) == -1) {
                        printf("Error al conectar la entrada de la tuberia %d\n", i);
                        exit(1);
                    }
                    close(tub[0][ENTRADA]);

                    ejecutarComando(linea->commands + i);
                } else {
                    close(tub[0][ENTRADA]);
                    if (dup2(tub[0][SALIDA], STDIN_FILENO) == -1) {
                        printf("Error al conectar la salida de la tuberia %d\n", i);
                        exit(1);
                    }
                    close(tub[0][SALIDA]);

                    ejecutarComando(linea->commands + i);
                }
            /* Caso normal: hay n > 2 comandos (n - 1 tuberias) */
            } else {
                /* Primer comando */
                if (i == 0) {
                    close(tub[0][SALIDA]);
                    if (dup2(tub[0][ENTRADA], STDOUT_FILENO) == -1) {
                        printf("Error al conectar la entrada de la tuberia %d\n", i);
                        exit(1);
                    }
                    close(tub[0][ENTRADA]);

                    ejecutarComando(linea->commands + i);
                /* Ultimo comando */
                } else if (i == numeroComandos - 1) {
                    close(tub[numeroComandos - 2][ENTRADA]);
                    if (dup2(tub[numeroComandos - 2][SALIDA], STDIN_FILENO) == -1) {
                        printf("Error al conectar la salida de la tuberia %d\n", i);
                        exit(1);
                    }
                    close(tub[numeroComandos - 2][SALIDA]);

                    ejecutarComando(linea->commands + i);
                /* Comando intermedio */
                } else {
                    close(tub[i - 1][ENTRADA]);
                    close(tub[i][SALIDA]);
                    dup2(tub[i - 1][SALIDA], STDIN_FILENO);
                    close(tub[i - 1][SALIDA]);
                    dup2(tub[i][ENTRADA], STDOUT_FILENO);
                    close(tub[i][ENTRADA]);

                    ejecutarComando(linea->commands + i);
                }
            }
        } else {
            /* Proceso padre */
            if (bg == 0) {
                signal(SIGINT, manejadorComando);
                signal(SIGQUIT, manejadorComando);
            }
        }
    }

    /* Cerrar todos los pipes en el padre */
    for (j = 0; j < numeroComandos - 1; j++) {
        close(tub[j][SALIDA]);
        close(tub[j][ENTRADA]);
    }
    /* Esperar a que terminen los procesos */
    for (i = 0; i < numeroComandos; i++) {
        if (linea->background == 0) {
            wait(NULL);
        } else {
            waitpid(pids[i], &status, WNOHANG);
        }
    }
}

int ejecutarComando(tcommand *comando) {
    return execvp(comando->filename, comando->argv);
}

void manejadorComando(int pid) {
    kill(pid, SIGKILL);
    printf("\n");
}

int cd(int argc, char *argv[]) {
    char *directorio;
    char *buf;
    int  bufsize;

    bufsize = 1024;
    buf     = (char *) malloc(sizeof(char) * bufsize);

    if (argc > 2) {
        printf("Cd: demasiados argumentos\n");

        return 1;
    } else if (argc == 1) { /* No se indican argumentos */
        directorio = getenv("HOME");
        if (directorio == NULL) {
            printf("Cd: No existe HOME\n");

            return 1;
        }
    } else {
            directorio = argv[1];
    }

    if (chdir(directorio) != 0) {
        printf("El directorio especificado no existe\n");

        return 1;
    } else {
        getcwd(buf, bufsize);
        printf("%s\n", buf);
    }

    return 0;
}

int jobs(int argc, trabajo *tb) {
    int num;

    if (argc == 1) {
        if (tb == NULL || tb->comando == NULL) {
            printf("jobs: No hay ningun trabajo en background\n");
            return 1;
        } else if (tb->comando == NULL || tb->comando->argv == NULL) {
            printf("No se ha podido acceder la informacion del comando\n");
        } else {
            num = 1;
            do {
                if (tb->bg == 1) {
                    printf("[%d] Ejecutando: %s\n", num, *(tb->comando->argv));
                    num++;
                }
                if (tb->siguiente != NULL)  {
                    tb = tb->siguiente;
                }
            } while (tb->siguiente != NULL);
        }
    } else {
        printf("Numero de argumentos invalido\n");

        return 1;
    }

    return 0;
}

int fg(int argc) {
    if (argc == 1) {

    } else {
        printf("Numero de argumentos invalido\n");
        return 1;
    }

    return 0;
}

void imprimirTrabajo(tcommand *trabajo) {
    printf("%s\n", trabajo->filename);
}
