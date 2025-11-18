#ifndef MYSHELL_H
#define MYSHELL_H

#include "parser.h"

/* SOLICITAR_ENTRADA
 *
 * Funcion que lee la entrada standard y devuelve un puntero a un tline
 * Tambien realiza comprobaciones de la cadena introducida
 */
tline *solicitarEntrada();

/* COMPROBAR_COMANDO
 *
 * Comprueba si el comando pasado por parametro es valido
 */
int comprobarComando(tcommand *comando);

/* EJECUTAR_LINEA
 *
 * Ejecuta la linea pasada por parametro.
 */
void ejecutarLinea(tline *linea);

/* EJECUTAR_COMANDO
 *
 * Ejecuta el comando pasado por parametro
 *
 * Sobrecarga: se le indica si despues viene un comando mas, para hacer
 * redirecciones. No la uso.
 */
int ejecutarComando(tcommand *comando);
/* int ejecutarComando(tcommand *comando, int sig);*/

/* MANEJADOR_COMANDO
 *
 * Manejador de los procesos que ejecutaran comandos
 */
void manejadorComando(int pid);

/* MANEJADOR_COMANDO_OFF
 *
 * Manejador para ignorar las senales
 */
/* void manejadorComandoOff(); */

/* CD
 *
 * Comando interno. Se le pasa la lista de argumentos y el numero de
 * argumentos
 */
int cd(int argc, char *argv[]);

/* TRABAJO
 *
 * Nodo de una lista enlazada que contiene todos los trabajos que se ejecutan
 * en segundo plano
*/
typedef struct trabajo {
    int             idbg;
    pid_t           pid;
    int             bg;
    tcommand        *comando;
    struct trabajo  *siguiente;
} trabajo;

/* JOBS
 *
 * Comando interno. muestra la lista de procesos que se est ́an ejecutando en
 * segundo plano en la minishell
 */
int jobs(int argc, struct trabajo *tb);

/* FG
 *
 * Comando interno. Reanuda la ejecucion del proceso en background
 * identificado por el numero obtenido en el mandato jobs, indicando el
 * mandato que se esta ejecutando
 */
int fg(int argc);

/* IMPRIMIR_TRABAJO
 *
 * Funcion auxiliar para impimir los trabajos en bg con formato
 */
void imprimirTrabajo(tcommand *trabajo);

#endif /* MYSHELL_H */
