#pragma once
#include "Nodo.h"

// �rbol binario de b�squeda. Se permiten duplicados, y �stos van hacia la izquierda
class ArbolBinarioDeBusqueda
{
	Nodo *raiz; // Raiz del �rbol, o NULL si el arbol est� vac�o
	int n; // Numero de nodos del arbol
	int orientacionSiguienteEliminacion; // Orientacion (-1 si es izquierdo, 1 si es derecho) que tendr� que tener la siguiente eliminaci�n a realizar

public:

	// Construye un �rbol binario de b�squeda vac�o
	// Complejidad temporal: O(1)
	ArbolBinarioDeBusqueda();

	// Inserta un elemento (siempre se insertar� como hoja)
	// Par�metro: nuevo elemento a insertar. Lo coloca en su sitio adecuado
	// Precondici�n: nuevoElemento no exist�a previamente en el �rbol
	// Complejidad temporal: O(lgn) con la mejor topolog�a, O(n) con la peor
	void insertar (int nuevoElemento);

	// Busca un elemento en el arbol binario de busqueda
	// Par�metro: elementoABuscar es la clave del nodo que queremos encontrar
	// Retorno: true si se encuentra en el �rbol, false si no
	// Complejidad temporal: O(lgn) con la mejor topolog�a, O(n) con la peor
	bool buscar(int elementoABuscar);
	
	// Elimina el primer nodo que se encuentre con un elemento dado
	// Par�metro: elemento a eliminar
	// Precondiciones: elementoAEliminar existe en el �rbol
	// Complejidad temporal: O(lgn) con la mejor topolog�a, O(n) con la peor
	void eliminar (int elementoAEliminar);

	// Imprime el �rbol en forma de esquema tabulado, indicando si cada nodo es hijo izquiero o derecho de su padre
	// Complejidad temporal: O(n), siendo n el n�mero de nodos del sub�rbol, tanto con la mejor topolog�a como con la peor
	void imprimir();

	// Destruye el �rbol, liberando la memoria de todos los nodos
	// Complejidad temporal: O(n), siendo n el n�mero de nodos del sub�rbol, tanto con la mejor topolog�a como con la peor
	~ArbolBinarioDeBusqueda();

private:

	// Busca recursivamente un elemento en el arbol binario de busqueda
	// Par�metros:
	// - raizSubarbol indica el subarbol en donde buscar
	// - elementoABuscar es la clave del nodo que queremos encontrar
	// Retorno: puntero al nodo que contiene el elementoABuscar si lo encuentra, o, 
	//          si no lo encuentra, puntero al nodo padre de donde se podr�a insertar
	// Precondici�n: raizSubarbol != NULL
	// Complejidad temporal: O(lgn) con la mejor topolog�a, O(n) con la peor
	Nodo* buscarRecursivo(Nodo* raizSubarbol, int elementoABuscar);
	
	// Imprime un subarbol por pantalla en forma de esquema, sangrando los hijos con una tabulaci�n. Esta pensado para ser recursivo
	// Par�metros:
	// - subarbol: nodo ra�z del subarbol que queremos imprimir
	// - numeroTabulaciones: numero de tabulaciones con la que imprimimos la raiz. Los hijos directos tendr�n una tabulaci�n m�s
	// - orientacion indica si subarbol (primer parametro) es un hijo izquierdo de su padre (-1) o es derecho (+1) o no tiene padre (0)
	// Precondiciones: 
	// - subarbol != NULL
	// - numeroTabulaciones>=0
	// - orientacion == 1 || orientacion == -1 || orientacion == 0
	// Complejidad temporal: O(n), tanto con la mejor topolog�a (T(n)=1+2T(n/2)) como con la peor (T(n)=1+T(n-1))
	void imprimirRecursivo(Nodo* subarbol, int numeroTabulaciones, int orientacion);

	// Elimina recursivamente los nodos de un subarbol
	// Par�metro: el nodo ra�z del subarbol a eliminar
	// Precondici�n: subarbol != NULL
	// Complejidad temporal: O(n), siendo n el n�mero de nodos del sub�rbol, tanto con la mejor topolog�a como con la peor
	void eliminarSubarbol(Nodo* raizSubarbol);

	// Elimina el nodo pasado como parametro. Lo sustituye por un descendiente suyo (recursivamente)
	// Par�metros:
	// - nodoParaEliminar: puntero al nodo que queremos eliminar
	// Complejidad temporal: O(lgn) con la mejor topolog�a, O(n) con la peor
	// (n es el n�mero de nodos del subarbol que empieza en nodoParaEliminar)
	void eliminarNodo(Nodo* nodoParaEliminar);
	
	// Buscar el maximo de un subarbol (ir por la rama derecha hasta no poder m�s)
	// Par�metro: raiz del subarbol en donde buscar
	// Retorno: puntero al nodo que contiene el m�ximo
	// Precondicion: raizSubarbol != NULL
	// Complejidad temporal: O(lgn) con la mejor topolog�a, O(n) con la peor
	Nodo *buscarMaximo (Nodo *raizSubarbol);

	// Buscar el minimo de un subarbol (ir por la rama izquierda hasta no poder m�s)
	// Par�metro: raiz del subarbol en donde buscar
	// Retorno: puntero al nodo que contiene el minimo
	// Precondicion: raizSubarbol != NULL
	// Complejidad temporal: O(lgn) con la mejor topolog�a, O(n) con la peor
	Nodo *buscarMinimo (Nodo *raizSubarbol);

};

