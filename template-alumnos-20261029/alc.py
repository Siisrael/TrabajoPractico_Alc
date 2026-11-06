import numpy as np
import os

#Funciones Auxiliares

PI = 3.141592653589793

def ceros(n, m=None):
    """
    Creo una matriz de NxM de ceros (float).
    Si no se pasa m, creo un vector de N ceros.
    """
    if m is None:
        return np.zeros(n)
    return np.zeros((n, m))

def identidad(n):
    """
    Creo la matriz identidad de NxN (float)
    """
    return np.eye(n)


_semilla = [12345]

def aleatorio(*forma):
    """
    Genera un array de numeros pseudoaleatorios en [0, 1) con la forma pedida,
    usando un generador congruencial lineal (reemplaza a np.random.rand).
    Args:
        forma: Dimensiones del array (ej: aleatorio(n) o aleatorio(n, m)).
    Returns:
        np.ndarray: Array con valores en [0, 1).
    """
    total = 1
    for d in forma:
        total *= d
    valores = []
    for _ in range(total):
        _semilla[0] = (1103515245 * _semilla[0] + 12345) % 2**31
        valores.append(_semilla[0] / 2**31)
    return np.array(valores).reshape(forma)


def signo(x):
    """
    Devuelve el signo de x (1, -1 o 0) (reemplaza a np.sign).
    """
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0


def _reducir_angulo(theta):
    """
    Lleva theta al intervalo [-pi, pi].
    """
    theta = float(theta)
    vueltas = int(theta / (2 * PI))
    theta = theta - vueltas * 2 * PI
    if theta > PI:
        theta -= 2 * PI
    elif theta < -PI:
        theta += 2 * PI
    return theta


def seno(theta, terminos=30):
    """
    Calcula sin(theta) con su serie de Taylor (reemplaza a np.sin).
    """
    x = _reducir_angulo(theta)
    termino = x
    suma = x
    for k in range(1, terminos):
        termino = -termino * x * x / ((2 * k) * (2 * k + 1))
        suma += termino
    return suma


def coseno(theta, terminos=30):
    """
    Calcula cos(theta) con su serie de Taylor (reemplaza a np.cos).
    """
    x = _reducir_angulo(theta)
    termino = 1.0
    suma = 1.0
    for k in range(1, terminos):
        termino = -termino * x * x / ((2 * k - 1) * (2 * k))
        suma += termino
    return suma


def producto_int(x, y):
    """
    Calcula el producto interno de dos vectores x e y.
    Args:
        x: Primer vector.
        y: Segundo vector.
    Returns:
        float: El resultado del producto interno.
    Raises:
        ValueError: Si los vectores no tienen la misma longitud.
    """
    if len(x) != len(y):
        raise ValueError("Los vectores deben tener la misma longitud para el producto interno.")
    res = 0.0
    for i in range(len(x)):
        res = res + x[i] * y[i]
    return res


def trasponer(A):
    """
    Calcula la transpuesta de una matriz A.
    Args:
        A: La matriz de entrada (m x n).
    Returns:
        np.ndarray: La matriz transpuesta A_t (n x m).
    """
    m, n = A.shape
    A_t = ceros(n, m)
    for i in range(m):
        for j in range(n):
            A_t[j, i] = A[i, j]
    return A_t

def mult_matricial(A, B):
    """
    Calcula la multiplicación de matrices C = A @ B.
    Args:
        A: Matriz (m x p).
        B: Matriz (p x n).
    Returns:
        np.ndarray: Matriz C (m x n).
    Raises:
        ValueError: Si las dimensiones internas no coinciden.
    """
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("Dimensiones internas no coinciden para la multiplicación")
    C = ceros(m, n)
    for i in range(m):
        for k in range(p):
            aik = A[i, k]
            for j in range(n):
                C[i, j] += aik * B[k, j]
    return C

def multMatVec(A,v):
    """
    Calcula la multiplicación de una Matriz A por un Vector v (Ax = b).
    Args:
        A: Matriz (n x m).
        v: Vector (m elementos).
    Returns:
        np.ndarray: Vector resultado (n elementos).
    Raises:
        ValueError: Si las dimensiones internas no coinciden (A.shape[1] != len(v)).
    """
    n, m = A.shape
    if m != len(v):
        raise ValueError(f"Dimensiones internas no coinciden: Matriz ({n},{m}) y Vector ({len(v)},)")
    res = ceros(n)
    for i in range(n):
        for j in range(m):
            res[i] += A[i][j]*v[j]
    return res

def f_A(A,v):
    """
    Calcula el paso normalizado del método de la potencia.
    Args:
        A: Matriz de entrada.
        v: Vector 1D.
    Returns:
        np.ndarray: Vector 1D normalizado.
    """
    tol = 1e-15
    Av = multMatVec(A,v)
    norma_Av = norma(Av,2)
    if np.abs(norma_Av) >= tol:
        Av = Av/norma_Av
    return Av

def esSimetrica(A, tol=1e-15):
    """
    Verifica si una matriz A es simétrica dentro de una tolerancia.
    Args:
        A: Matriz de entrada.
        tol: Tolerancia para la comparación.
    Returns:
        bool: True si A es simétrica, False en caso contrario.
    """
    res = True
    m, n = A.shape
    if m != n:
        return False
    for i in range(m):
        for j in range(i+1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                res = False
    return res


def producto_ext(v1,v2):
    """
    Calcula el producto externo de dos vectores v1 y v2 (A = v1 @ v2^T).
    Args:
        v1: El primer vector (vector columna, m elementos).
        v2: El segundo vector (vector fila, n elementos).
    Returns:
        np.ndarray: La matriz resultante A (m x n).
    """
    m = len(v1)
    n = len(v2)
    A = ceros(m,n)
    for i in range(m):
        for j in range(n):
            A[i,j] = v1[i]*v2[j]
    return A

# Laboratorio 1
def error(x,y):
    """
    Calcula el error absoluto e_abs = |x - y|.
    Args:
        x: El valor "real" o de referencia.
        y: El valor aproximado.
    Returns:
        float: El error absoluto.
    """
    e_abs = np.abs(float(x) - float(y))
    return e_abs


def error_relativo(x,y):
    """
    Calcula el error relativo e_rel = e_abs / |x|.
    Usa un epsilon de estabilidad para evitar la división por cero.
    Args:
        x: El valor "real" o de referencia.
        y: El valor aproximado.
    Returns:
        float: El error relativo.
    """
    e_abs = error(x,y)
    epsilon = 1e-15
    denominador = np.abs(float(x)) + epsilon
    e_rel = e_abs / denominador
    return e_rel


def matricesIguales(A,B):
    """
    Verifica si dos matrices A y B son iguales en dimensiones y valores
    (dentro de una tolerancia numérica).
    Args:
        A: Primera matriz.
        B: Segunda matriz.
    Returns:
        bool: True si son iguales, False en caso contrario.
    """
    tol = 1e-15
    res = True
    if A.shape != B.shape:
        res = False
    else:
        C = A - B
        m,n = C.shape
        for i in range(m):
            for j in range(n):
                if np.abs(C[i,j]) > tol:
                    res = False
    return res

# Laboratorio 2

def rota(theta):
    """
    Genera una matriz de rotación 2x2.
    Args:
        theta: El ángulo de rotación en radianes.
    Returns:
        np.ndarray: La matriz de rotación 2x2.
    """
    A_rot = np.array([
            [coseno(theta), -seno(theta)],
            [seno(theta), coseno(theta)]])
    return A_rot


def escala(s):
    """
    Genera una matriz de escalado diagonal n x n.
    Args:
        s: Una lista de n factores de escalado.
    Returns:
        np.ndarray: La matriz de escalado diagonal n x n.
    """
    n = len(s)
    A_esc = ceros(n, n)
    for i in range(n):
        A_esc[i,i] = s[i]
    return A_esc


def rota_y_escala(theta,s):
    """
    Combina una rotación y un escalado en una única matriz 2x2.
    Args:
        theta: El ángulo de rotación en radianes.
        s: Una lista de factores de escalado.
    Returns:
        np.ndarray: La matriz de transformación 2x2 combinada.
    Raises:
        ValueError: Si la lista 's' no tiene 2 elementos.
    """
    if len(s) != 2:
        raise ValueError("s debe tener 2 elementos")
    A_esc = escala(s)
    A_rot = rota(theta)
    A = mult_matricial(A_esc,A_rot)
    return A


def afin(theta,s,b):
    """
    Genera una matriz afín 3x3 (rotación, escalado y traslación).
    Args:
        theta: Ángulo de rotación (radianes).
        s: Factores de escalado.
        b: Vector de traslación.
    Returns:
        np.ndarray: Matriz de transformación afín 3x3.
    """
    A = rota_y_escala(theta,s)
    M_afin = ceros(3,3)
    for i in range(2):
        for j in range(2):
            M_afin[i,j] = A[i,j]
    for i in range(2):
        M_afin[i,2] = b[i]
    M_afin[2,2] = 1.0
    return M_afin


def trans_afin(v,theta,s,b):
    """
    Aplica una transformación afín 2D (rotación, escalado, traslación) a un vector v.
    Args:
        v: Vector 2D de entrada.
        theta: Ángulo de rotación.
        s: Factores de escalado.
        b: Vector de traslación.
    Returns:
        np.ndarray: El vector 2D transformado.
    """
    if len(v) != 2:
        raise ValueError("El vector 'v' debe tener 2 elementos.")
    M_afin = afin(theta,s,b)
    # Convierto v a coordenadas homogéneas (vector columna 3x1)
    v_aux = ceros(3,1)
    for i in range(2):
        v_aux[i,0] = v[i]
    v_aux[2,0] = 1.0
    # Aplico la transformación
    w = mult_matricial(M_afin, v_aux)
    epsilon = 1e-15
    if np.abs(w[2,0]) < epsilon:
        w[2,0] = epsilon
    w = w/w[2,0]
    # Resultado 2D
    w = np.array([w[0,0],w[1,0]])
    return w

# Laboratorio 3

def norma(x,p):
    """
    Devuelve la norma p del vector x.
    Args:
        x: El vector de entrada (si es columna n x 1 también funciona).
        p: El orden de la norma.
    Returns:
        float: El valor de la norma-p.
    """
    x = np.array(x, dtype=float).reshape(-1)
    suma = 0.0
    if p == 'inf':
        suma = np.max(np.abs(x))
    else:
        for elemento in x:
            suma += np.abs(elemento)**p
        suma = suma**(1/p)
    return suma


def normaliza(X,p):
    """
    Recibe X, una lista de vectores no vacíos, y un escalar p.
    Devuelve una lista donde cada elemento corresponde a normalizar
    los elementos de X con la norma p.
    Args:
        X: Una lista de vectores.
        p: El orden de la norma.
    Returns:
        np.ndarray: Una lista de los vectores normalizados.
    """
    Y = []
    tol = 1e-15
    for v in X:
        v = np.array(v, dtype=float)
        norm_v = norma(v,p)
        if np.abs(norm_v) < tol:
            Y.append(v*0)
        else:
            Y.append(v/norm_v)
    return np.array(Y)


def normaMatMC(A,q,p,Np):
    """
    Devuelve la norma ||A|| \\ {q,p} y el vector x en el cual se alcanza el maximo.
    Args:
        A: Matriz de entrada.
        q: Orden de la norma para el espacio de llegada.
        p: Orden de la norma para el espacio de salida.
        Np: Número de iteraciones.
    Returns:
        tuple (float, np.ndarray): (max_norma, w_max)
            - max_norma: El valor de la norma aproximada.
            - w_max: El vector que genera dicha norma.
    """
    epsilon = 1e-15
    m,n = A.shape
    w = ceros(n,1)
    maximo = 0.0
    for i in range(Np):
        # Vector al azar con entradas en [-1, 1]
        v = 2 * aleatorio(n,1) - 1
        v_norma = norma(v,p)
        if np.abs(v_norma) < epsilon:
            v_norma += epsilon
        v = v / v_norma
        norma_vector = norma(mult_matricial(A,v),q)
        if maximo <= norma_vector:
            maximo = norma_vector
            w = np.array(v)
    return maximo, w


def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A.
    Si p es 1 o 'inf', devuelve solo esa norma.
    Args:
        A: Matriz de entrada.
        p: 1, 'inf', o una lista con esos valores (por defecto [1, 'inf']).
    Returns:
        float o list: La norma pedida, o la lista de normas si p es una lista.
    Raises:
        ValueError: Si 'p' no es 1 o 'inf'.
    """
    if isinstance(p, list):
        return [normaExacta(A, q) for q in p]
    maximo = 0.0
    m, n = A.shape
    if p == 1:
        # Calculo norma 1
        for j in range(n):
            suma_col = 0.0
            for i in range(m):
                suma_col += np.abs(A[i, j])
            maximo = max(maximo, suma_col)
    elif p == 'inf':
        # Calculo norma infinito
        for i in range(m):
            suma_fila = 0.0
            for j in range(n):
                suma_fila += np.abs(A[i, j])
            maximo = max(maximo, suma_fila)
    else:
        raise ValueError(f"Parámetro 'p' no válido: {p}. Solo se acepta 1 o 'inf'.")
    return maximo


def condMC(A,p):
    """
    Devuelve el numero de condicion de A usando la norma inducida p
    (aproximado por Monte Carlo).
    Args:
        A: Matriz de entrada.
        p: El orden de la norma.
    Returns:
        float or None: El número de condición aproximado, o None si A no es invertible.
    """
    A_inv = inversa(A)
    if A_inv is None:
        return None
    Np = 10000 # Cantidad de vectores al azar para Monte Carlo
    norma_A, _ = normaMatMC(A,p,p,Np)
    norma_inv, _ = normaMatMC(A_inv,p,p,Np)
    return norma_A * norma_inv


def condExacto(A,p):
    """
    Devuelve el numero de condicion de A usando la norma-p exacta.
    Args:
        A: Matriz de entrada.
        p: El orden de la norma (solo soporta 1 o 'inf').
    Returns:
        float or None: El número de condición exacto, o None si A no es invertible.
    """
    A_inv = inversa(A)
    if A_inv is None:
        return None
    norma_A = normaExacta(A, p)
    norma_inv = normaExacta(A_inv, p)
    return norma_A * norma_inv

# Laboratorio 4
def calculaLU(A):
    """
    Calcula la factorizacion LU de la matriz A (sin pivoteo) y
    retorna L, U, y el número de operaciones.
    Args:
        A: La matriz cuadrada de entrada.
    Returns:
        tuple (L, U, cant_op) o (None, None, 0) si falla.
            L (np.ndarray): Matriz triangular inferior (con 1s en diagonal).
            U (np.ndarray): Matriz triangular superior.
            cant_op (int): Conteo de operaciones (sumas, restas, mult, div).
    """
    m, n = A.shape
    if m != n:
        return None, None, 0 # No es cuadrada
    A_copia = np.array(A, dtype=float)
    cant_op = 0
    tol = 1e-15
    for j in range(n):
        if np.abs(A_copia[j, j]) < tol:
            return None, None, 0 # Pivote cero
        # Eliminación (para filas i > j)
        for i in range(j + 1, n):
            mult_fila = A_copia[i, j] / A_copia[j, j]
            cant_op += 1
            A_copia[i, j] = mult_fila
            # Actualizo el resto de la fila i (la parte U)
            for k in range(j + 1, n):
                A_copia[i, k] = A_copia[i, k] - mult_fila * A_copia[j, k]
                cant_op += 2
    #Extracción de L y U de A_copia
    L = identidad(n)
    U = ceros(n, n)
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = A_copia[i, j]
            else:
                U[i, j] = A_copia[i, j]
    return L, U, cant_op


def res_tri(L, b, inferior = True):
    """
    Resuelve el sistema Lx=b, donde L es triangular.
    Args:
        L: Matriz triangular (superior o inferior).
        b: Vector de términos independientes, o matriz (n x k) para
           resolver k sistemas a la vez (uno por columna).
        inferior: True si L es triangular inferior. False si es superior.
    Returns:
        np.ndarray: El vector solución 'x' (1D array), o matriz X (n x k).
    Raises:
        ValueError: Si un pivote en la diagonal es numéricamente cero.
    """
    tol = 1e-15
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)
    n = L.shape[0]
    # Caso b matriz (n x k): resuelvo columna por columna
    if b.ndim == 2:
        k = b.shape[1]
        X = ceros(n, k)
        for j in range(k):
            X[:, j] = res_tri(L, b[:, j], inferior)
        return X
    # Caso b vector
    res = ceros(n)
    if inferior:
        for i in range(n):
            suma = 0.0
            for j in range(i):
                suma += L[i,j] * res[j]
            denominador = L[i,i]
            if np.abs(denominador) < tol:
                raise ValueError(f"Pivote numéricamente cero en L[{i},{i}]")
            res[i] = (b[i] - suma) / denominador
    else:
        for i in reversed(range(n)):
            suma = 0.0
            for j in range(i + 1, n):
                suma += L[i,j] * res[j]
            denominador = L[i, i]
            if np.abs(denominador) < tol:
                raise ValueError(f"Pivote numéricamente cero en L[{i},{i}]")
            res[i] = (b[i] - suma) / denominador
    return res


def inversa(A):
    """
    Calcula la inversa de A empleando la factorizacion LU
    y las funciones que resuelven sistemas triangulares.
    Args:
        A: Matriz de entrada (debe ser cuadrada e invertible).
    Returns:
        np.ndarray or None: La matriz inversa, o None si A es singular.
    """
    L,U,_ = calculaLU(A)
    if L is None:
        return None
    m = A.shape[0]
    I = identidad(m)
    A_inv = ceros(m, m)
    for i in range(m):
        y = res_tri(L,I[:,i]) # Sustitución hacia adelante
        x_i = res_tri(U,y,False) # Sustitución hacia atrás
        A_inv[:,i] = x_i
    return A_inv


def calculaLDV(A):
    """
    Calcula la factorizacion LDV de la matriz A.
    Args:
        A: Matriz de entrada.
    Returns:
        tuple (L, D, V) o (None, None, None) si falla.
            L (np.ndarray): Triangular inferior (1s en diagonal).
            D (np.ndarray): Diagonal.
            V (np.ndarray): Triangular superior (1s en diagonal).
    """
    tol = 1e-15
    L, U, _ = calculaLU(A)
    # Chequeo si LU falló
    if L is None:
        return None, None, None
    m,n = U.shape
    D = ceros(m,n)
    V = ceros(m,n)
    for i in range(n):
        pivote = U[i, i]
        if np.abs(pivote) < tol:
            return None, None, None # Matriz U es singular
        D[i, i] = pivote
        for j in range(i, n):
            V[i, j] = U[i, j] / pivote
    return L, D, V


def esSPD(A, atol=1e-8):
    """
    Checkea si la matriz A es simetrica definida positiva (SDP) usando
    la factorizacion LDV.
    Args:
        A: Matriz de entrada.
        atol: Tolerancia para los chequeos.
    Returns:
        bool: True si A es SDP, False en caso contrario.
    """
    res = True
    m, n = A.shape
    if m != n:
        return False # No es cuadrada
    # Chequeo simetría
    for i in range(m):
        for j in range(i + 1, m):
            if np.abs(A[i, j] - A[j, i]) > atol:
                return False # No es simétrica
    L, D, V = calculaLDV(A)
    # Chequeo si LDV falló
    if L is None:
        return False
    # Chequeo la diagonal de D
    for i in range(n):
        if D[i, i] <= atol:
            res = False
    return res

# Laboratorio 5
def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    Calcula la factorización QR de A usando Gram-Schmidt.
    Args:
        A: Matriz de entrada (m x n).
        tol: Tolerancia para la norma.
        retorna_nops: Si es True, devuelve (Q, R, n_ops).
    Returns:
        tuple: (Q, R) o (Q, R, n_ops)
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = ceros(m, n)
    R = ceros(n, n)
    n_ops = 0
    norma_v = norma(A[:, 0], 2)
    n_ops += m*2
    if np.abs(norma_v) < tol:
        Q[:, 0] = 0.0
    else:
        Q[:, 0] = A[:, 0] / norma_v
        n_ops += m
    R[0, 0] = norma_v
    # Bucle principal (Columnas 1 a n-1)
    for j in range(1, n):
        q_j_monio = np.array(A[:, j], dtype=float)
        for k in range(j):
            # Proyección
            R[k, j] = producto_int(Q[:, k], q_j_monio)
            n_ops += (m * 2)
            # Ortogonalización
            q_j_monio = q_j_monio - R[k, j] * Q[:, k]
            n_ops += m + m
        R[j, j] = norma(q_j_monio, 2)
        n_ops += m * 2
        if abs(R[j, j]) < tol:
            Q[:, j] = 0.0
        else:
            Q[:, j] = q_j_monio / R[j, j]
            n_ops += m
    if retorna_nops:
        return Q, R, n_ops
    else:
        return Q, R


def QR_con_HH(A, tol=1e-12, extras=False):
    """
    Calcula la factorización QR de A usando reflectores de Householder.
    Args:
        A: Matriz de entrada (m x n), con m >= n.
        tol: Tolerancia con la que se filtran elementos nulos en R.
        extras: Si es True, devuelve informacion extra sobre el proceso
                de factorizacion (para poder graficarlo).
    Returns:
        Q, R (si extras es False)
        Q, R, extra_info (si extras es True)
        None si A no cumple m >= n.
            Q (np.ndarray): Matriz con columnas ortonormales (m x n).
            R (np.ndarray): Matriz triangular superior (n x n).
            extra_info (dict): Diccionario con las claves
                'R_matrices': lista de las matrices R en cada paso.
                'Q_matrices': lista de las matrices Q en cada paso.
                (La posicion 0 es el estado inicial: R = A, Q = I.)
    """
    m, n = A.shape
    if m < n:
        return None

    Q = identidad(m)
    R = np.array(A, dtype=float)

    R_matrices = [np.array(R)]
    Q_matrices = [np.array(Q)]

    for k in range(n):
        # Calculo el reflector u
        x = R[k:m, k]
        norm_x = norma(x, 2)

        a = -signo(x[0]) * norm_x
        if a == 0 and norm_x > tol:
            a = -norm_x

        u = x - a * identidad(m - k)[0]
        norm_u = norma(u, 2)

        if norm_u > tol:
            u = u / norm_u
            u_fila = u.reshape(1, -1)
            u_col = u.reshape(-1, 1)

            # R <- H_k R (solo cambia el bloque R[k:m, k:n])
            R_sub = R[k:m, k:n]
            vT = mult_matricial(u_fila, R_sub)
            R[k:m, k:n] = R_sub - 2 * mult_matricial(u_col, vT)

            # Q <- Q H_k (solo cambian las columnas k:m)
            Q_sub = Q[:, k:m]
            v_col = multMatVec(Q_sub, u).reshape(-1, 1)
            Q[:, k:m] = Q_sub - 2 * mult_matricial(v_col, u_fila)

        # Filtro elementos nulos en R
        for i in range(m):
            for j in range(n):
                if np.abs(R[i, j]) < tol:
                    R[i, j] = 0.0

        R_matrices.append(np.array(R))
        Q_matrices.append(np.array(Q))

    #Redimensionamos
    Q_thin = Q[:, :n]
    R_thin = R[:n, :n]

    if extras:
        extra_info = {'R_matrices': R_matrices, 'Q_matrices': Q_matrices}
        return Q_thin, R_thin, extra_info
    return Q_thin, R_thin


def calculaQR(A, metodo='RH', tol=1e-12, retorna_nops=False):
    """
    Calcula la factorización QR de A usando reflectores de Householder (RH)
    o Gram-Schmidt (GS).
    Args:
        A: Matriz de entrada (n x n).
        metodo: 'RH' o 'GS'.
        tol: Tolerancia con la que se filtran elementos nulos en R.
        retorna_nops: Si es True y metodo='GS', devuelve tambien el
                      numero de operaciones como tercer elemento.
    Returns:
        Q, R  (o Q, R, n_ops si metodo='GS' y retorna_nops=True)
        None si el metodo no esta entre las opciones.
    """
    if metodo == 'GS':
        return QR_con_GS(A, tol, retorna_nops)
    elif metodo == 'RH':
        return QR_con_HH(A, tol)
    else:
        return None

# Alias con el nombre anterior (por compatibilidad)
calcula_QR = calculaQR


# Laboratorio 6


def metpot2k(A, tol=1e-15, K=1000):
    """
    Calcula el autovector dominante y autovalor asociado usando el
    método de la potencia (variante 2k).
    Args:
        A: Matriz de entrada (n x n).
        tol: Tolerancia para la convergencia.
        K: Número máximo de iteraciones.
    Returns:
        tuple (v_monio, lam, k, epsilon):
            v_monio (np.ndarray): Autovector dominante (normalizado, 1D).
            lam (float): Autovalor dominante asociado.
            k (int): Número de iteraciones realizadas.
            epsilon (float): Error final de convergencia.
    """
    n = A.shape[0]
    v = aleatorio(n)
    v = v / norma(v, 2)
    v_monio = f_A(A,f_A(A,v))
    e = producto_int(v_monio,v)
    k = 0
    while abs(e-1) > tol and k < K:
        v = v_monio
        v_monio = f_A(A,f_A(A,v))
        e = producto_int(v_monio,v)
        k += 1
    lam = producto_int(v_monio, multMatVec(A,v_monio))
    epsilon = e - 1
    return v_monio,lam,k,epsilon

def diagRH(A, tol=1e-15, K=1000):
    """
    Calcula la diagonalización A = S D S^T para una matriz simétrica.
    Utiliza el método recursivo de Householder.
    Args:
        A: Matriz simétrica (n x n).
        tol: Tolerancia.
        K: Máximo de iteraciones para metpot2k.
    Returns:
        tuple (S, D) o (None, None):
            S (np.ndarray): Matriz ortogonal de autovectores.
            D (np.ndarray): Matriz diagonal de autovalores.
    """
    if not esSimetrica(A, tol):
        return None, None
    n = A.shape[0]
    # Caso trivial (n=1)
    if n == 1:
        return identidad(1), np.array(A, dtype=float)
    # Obtengo autovalor/vector dominante
    v1, lam1, _, _ = metpot2k(A,tol,K)
    # Construyo Reflector H_v1
    l = len(v1)
    I_l = identidad(l)
    e1 = ceros(l)
    e1[0] = 1
    if producto_int(e1-v1, e1-v1) > tol:
        H_v1 = I_l - 2*(producto_ext(e1-v1, e1-v1)/producto_int(e1-v1, e1-v1))
    else:
        H_v1 = I_l
    # Caso Base (n=2)
    if n == 2:
        S = H_v1
        D = mult_matricial(mult_matricial(H_v1,A), trasponer(H_v1))
    # Paso Recursivo (n > 2)
    else:
        B = mult_matricial(mult_matricial(H_v1,A), trasponer(H_v1))
        A_monio = B[1:n, 1:n]
        # Simetrizo para eliminar errores de redondeo
        A_monio = (A_monio + trasponer(A_monio)) / 2
        S_monio, D_monio = diagRH(A_monio,tol,K)
        if D_monio is None:
            return None, None
        D = ceros(D_monio.shape[0]+1,D_monio.shape[1]+1)
        D[0, 0] = lam1
        D[1:n, 1:n] = D_monio
        S_aux = ceros(S_monio.shape[0]+1,S_monio.shape[1]+1)
        S_aux[0, 0] = 1
        S_aux[1:n, 1:n] = S_monio
        S = mult_matricial(H_v1, S_aux)
    return S, D


# Laboratorio 07

def transiciones_al_azar_continuas(n):
    """
    Retorna matriz T de nxn normalizada por columnas, y con entradas
    al azar en el intervalo [0, 1].
    Args:
        n: Dimensión de la matriz (n x n).
    Returns:
        np.ndarray: La matriz T (n x n) de transiciones.
    """
    tol = 1e-15
    A = aleatorio(n,n)
    for j in range(A.shape[1]):
        columna = A[:, j]
        norma1 = norma(columna, 1)
        if np.abs(norma1) < tol:
            A[:, j] = 0.0
        else:
            A[:, j] = columna / norma1
    return A

def transiciones_al_azar_uniformes(n,thres):
    """
    Retorna matriz T de nxn normalizada por columnas, con 0s y 1s
    basados en un umbral 'thres'.
    Args:
        n: Dimensión de la matriz (n x n).
        thres: Umbral de probabilidad [0, 1].
    Returns:
        np.ndarray: La matriz T (n x n) de transiciones.
    """
    tol = 1e-15
    V_aplanado = aleatorio(n*n)
    for i in range(len(V_aplanado)):
        if V_aplanado[i] > thres:
            V_aplanado[i] = 0.0
        else:
            V_aplanado[i] = 1.0
    V = V_aplanado.reshape(n, n)
    for j in range(V.shape[1]):
        columna = V[:, j]
        norma1 = norma(columna, 1)
        if np.abs(norma1) < tol:
            V[:, j] = 0.0
        else:
            V[:, j] = columna / norma1
    return V


def nucleo(A,tol = 1e-15):
    """
    Calcula el nucleo de la matriz A diagonalizando A^T A.
    Args:
        A: Matriz de entrada (m x n).
        tol: Tolerancia para el autovalor cero.
    Returns:
        np.ndarray: Matriz (n x k) cuyas columnas son la base del núcleo
                    (n x 0 si el núcleo es trivial), o None si la
                    diagonalización falla.
    """
    B = mult_matricial(trasponer(A),A)
    S,D = diagRH(B,tol)
    # Chequeo si diagRH falló
    if S is None:
        return None
    res = []
    for i in range(D.shape[0]):
        if np.abs(D[i,i]) < tol:
            res.append(S[:,i])
    if len(res) == 0:
        res = ceros(A.shape[1],0)
    else:
        res = np.array(res)
        res = trasponer(res)
    return res


def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Crea una matriz rala en formato de diccionario.
    Args:
        listado: Lista de 3 listas: [filas, columnas, valores].
        m_filas: Número total de filas de la matriz.
        n_columnas: Número total de columnas de la matriz.
        tol: Tolerancia para descartar valores.
    Returns:
        tuple: (rala, dimensiones)
            rala (dict): Diccionario {(i, j): Aij} con elementos > tol.
            dimensiones: (m_filas, n_columnas).
    """
    dimensiones = (m_filas, n_columnas)
    rala = dict()
    if len(listado) == 0:
        return rala, dimensiones
    for i in range(len(listado[0])):
        if abs(listado[2][i]) > tol:
            rala[(listado[0][i],listado[1][i])] = listado[2][i]
    return rala, dimensiones


def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala (formato dict de crea_rala) y un vector v.
    Retorna un vector w = A @ v.
    Args:
        A: La matriz rala (diccionario, dimensiones).
        v: El vector de entrada.
    Returns:
        np.ndarray: El vector resultado.
    Raises:
        ValueError: Si las dimensiones de A y v no son compatibles.
    """
    diccionario, dimensiones = A
    n,m = dimensiones
    if len(v) != m:
        raise ValueError(f"Dimensiones no compatibles: Matriz ({n}x{m}) y Vector ({len(v)},)")
    w = ceros(n)
    for clave,valor in diccionario.items():
        w[clave[0]] += valor * v[clave[1]]
    return w



# Laboratorio 08

def svd_reducida(A, k="max", tol=1e-15):
    """
    Calcula la SVD reducida (U, S, V) de A.
    Args:
        A: Matriz de entrada (m x n).
        k: Número de valores singulares a retener.
        tol: Tolerancia para el valor singular cero.
    Returns:
        tuple (hatU, hatSig, hatV) o (None, None, None) si falla.
            hatU (np.ndarray): Matriz (m x k) de vectores singulares izquierdos.
            hatSig (np.ndarray): Vector (k,) de valores singulares.
            hatV (np.ndarray): Matriz (n x k) de vectores singulares derechos.
    """
    m, n = A.shape
    switch = True
    if m >= n:
        C = mult_matricial(trasponer(A),A)
    else:
        C = mult_matricial(A,trasponer(A))
        switch = False
    S, D = diagRH(C,tol)
    # Chequeo si diagRH falló
    if S is None:
        return None, None, None
    E = []
    r = 0
    # ordeno D y S
    for i in range(D.shape[0]):
        if D[i][i] > tol:
            r += 1
            E.append((np.sqrt(D[i][i]),i))
    if k != "max":
        r = min(int(k),r)
    E = sorted(E,reverse=True)
    E = E[:r]
    hatSig = []
    hatV = ceros(S.shape[0],r)
    for i in range(r):
        hatSig.append(E[i][0])
        indice = E[i][1]
        hatV[:,i] = S[:,indice]
    hatSig = np.array(hatSig)
    if switch:
        U = mult_matricial(A,hatV)
    else:
        U = mult_matricial(trasponer(A),hatV)
    for col in range(U.shape[1]):
        sigma_i = E[col][0]
        if np.abs(sigma_i) > tol:
            U[:, col] = (1.0 / sigma_i) * U[:, col]
        else:
            U[:, col] = 0.0
    hatU = U
    if not switch:
        hatU, hatV = hatV, hatU
    return hatU, hatSig, hatV


# ====================================================================================================
# Funciones del TP - Calibración y Pose de la Cámara
# Son las mismas que están en la notebook; acá llaman directo a las funciones de este módulo (sin "alc.").
# ====================================================================================================


# ----------------------------------------------------------------------------------------------------
# Ejercicio 2 - Calibración con puntos 3D (DLT)
# ----------------------------------------------------------------------------------------------------


def obtenerQ3d(imgPts, objPts):
    '''
    Arma la matriz Q de la ecuación 6 del enunciado, cuyo núcleo contiene a la matriz de proyección
    vectorizada m = (m11, m12, m13, m14, m21, ..., m34).
    Parámetros:
        imgPts: array de n x 2 con las coordenadas (u, v) de los puntos en la imagen, en píxeles.
        objPts: array de n x 3 con las coordenadas (U, V, W) de los mismos puntos en el mundo.
    Devuelve:
        Q: matriz de 2n x 12 (dos filas por punto).
    '''
    n = imgPts.shape[0]
    Q = np.zeros((2*n,12))

    for i in range(n): 

        U, V, W = objPts[i,0], objPts[i,1], objPts[i,2]
        u, v = imgPts[i]

        ecuacionRespecto_u = 2*i
        ecuacionRespecto_v = 2*i + 1 

        Q[ecuacionRespecto_u] = [ U, V, W, 1, 
                                  0, 0, 0, 0, 
                               -u*U, -u*V, -u*W, -u ]

        Q[ecuacionRespecto_v] = [ 0, 0, 0, 0,
                                  U, V, W, 1, 
                               -v*U, -v*V, -v*W, -v ]

    return Q


def resolverDLT3D(Q):
    '''
    Encuentra la matriz de proyección M resolviendo min ||Q m|| con ||m|| = 1 (ecuación 8).
    Por el Ejercicio 1 b), la solución es el autovector de Q^T Q asociado al menor autovalor.
    Parámetros:
        Q: matriz de 2n x 12 armada con obtenerQ3d.
    Devuelve:
        M: matriz de proyección de 3 x 4, definida a menos de un factor de escala (sale con norma 1).
    '''

    Q_traspuesta = trasponer(Q)
    A = mult_matricial(Q_traspuesta, Q)
    
    V, D = diagRH(A)

    autoval_min = 0
    for j in range(1, D.shape[0]):
        if D[j, j] < D[autoval_min, autoval_min]:
            autoval_min = j

    m = V[:, autoval_min]
    M = m.reshape(3, 4)

    return M


def determinante3x3(A):
    '''
    Calcula el determinante de una matriz de 3 x 3 desarrollando por la primera fila.
    Parámetros:
        A: matriz de 3 x 3.
    Devuelve:
        el determinante de A (un número).
    '''

    determinante = ( A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
                    -A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
                    +A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]) )
    return determinante


def devolverMatricesCalibracion3d(M, focos_iguales=False, sin_sesgo=False):
    '''
    Descompone la matriz de proyección M = s K [R | t] en los intrínsecos K, la rotación R y la traslación t.
    Aplica la factorización QR del módulo a P A^T P (P invierte el orden de filas y columnas), para escribir
    las primeras tres columnas de M como triangular superior (K) por ortogonal (R). Así R sale ortonormal.
    Parámetros:
        M: matriz de proyección de 3 x 4, a menos de un factor de escala (por ejemplo, la de resolverDLT3D).
        focos_iguales: si es True, impone alpha_u = alpha_v (las reemplaza por su promedio).
        sin_sesgo: si es True, impone K12 = 0 (sesgo nulo).
    Devuelve:
        R: matriz de rotación de 3 x 3 (ortonormal, con det(R) = 1).
        t: vector de traslación de largo 3.
        K: matriz de intrínsecos de 3 x 3, triangular superior, con diagonal positiva y K[2, 2] = 1.
    '''
    # M = s K [R | t]: las primeras 3 columnas son s K R y la última es s K t
    A = M[:, :3]
    b = M[:, 3]

    # P da vuelta el orden de las filas y columnas (P P = I). La usamos para poder sacar K y R con la QR
    P = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    # Necesitamos A = (triangular superior) * (ortogonal), pero la QR del módulo da al revés
    # (ortogonal * triangular superior). 
    B = mult_matricial(P, trasponer(A))
    B = mult_matricial(B, P)

    Q_qr, R_qr = calculaQR(B, metodo="RH")

    # Como P A^T P = Q_qr R_qr, despejando queda A = (P R_qr^T P)(P Q_qr^T P)
    # P R_qr^T P es triangular superior -> es K (todavía sin arreglar signos ni escala)
    K0 = mult_matricial(P, trasponer(R_qr))
    K0 = mult_matricial(K0, P)

    # P Q_qr^T P es ortogonal -> es R
    R0 = mult_matricial(P, trasponer(Q_qr))
    R0 = mult_matricial(R0, P)

    # La diagonal de K tiene las focales, que tienen que ser positivas. Con D = diag(+-1) arreglamos los signos:
    # como D D = I, K0 D D R0 = K0 R0, así que el producto no cambia
    D = np.zeros((3, 3))
    for i in range(3):
        D[i, i] = signo(K0[i, i])

    K1 = mult_matricial(K0, D)
    R1 = mult_matricial(D, R0)

    # R tiene que ser una rotación (det = 1). Si da -1 es porque M vino con el signo cambiado
    # (m y -m son las dos solución del DLT), así que le cambiamos el signo a R (y más abajo a b)
    signo_det = signo(determinante3x3(R1))
    R = signo_det * R1

    # M está definida a menos de escala: normalizamos K para que K[2, 2] = 1 (como en la sección 2.2 g)
    c = K1[2, 2]
    K = K1 / c

    # Inciso d): restricciones opcionales sobre K
    if focos_iguales:
        # alpha_u = alpha_v: reemplazamos las dos focales por su promedio
        alpha = (K[0, 0] + K[1, 1]) / 2
        K[0, 0] = alpha
        K[1, 1] = alpha

    if sin_sesgo:
        # K12 = 0: sacamos el sesgo
        K[0, 1] = 0

    # La última columna de M es s K t: le sacamos el signo y la escala, y despejamos t de K t = b
    # (K es triangular superior, así que alcanza con sustitución hacia atrás)
    b_normalizada = signo_det * b / c

    t = res_tri(K, b_normalizada, inferior=False)

    return R, t, K


# ----------------------------------------------------------------------------------------------------
# Ejercicio 3 - Calibración con puntos coplanares (Zhang)
# Auxiliares: autovectorMinimo, matrizNormalizacion, aplicarTransformacion, estimarHomografia, vectorV, cholesky,
# productoVectorial y ortonormalizarRotacion (las usan las funciones pedidas)
# ----------------------------------------------------------------------------------------------------


def autovectorMinimo(A):
    '''
    Calcula el autovector asociado al menor autovalor de una matriz simétrica,
    que por el Ejercicio 1 b) es la solución de min ||Qx|| con ||x|| = 1 cuando A = Q^T Q.
    Parámetros:
        A: matriz simétrica de n x n.
    Devuelve:
        vector de largo n, de norma 1.
    '''
    autovectores, autovalores = diagRH(A)

    # Buscamos la posición del menor autovalor en la diagonal
    autoval_min = 0
    for j in range(1, autovalores.shape[0]):
        if autovalores[j, j] < autovalores[autoval_min, autoval_min]:
            autoval_min = j

    return autovectores[:, autoval_min]


def matrizNormalizacion(puntos):
    '''
    Arma la matriz de normalización isotrópica de Hartley (sección 2.3 del enunciado).
    Parámetros:
        puntos: array de n x 2 con coordenadas cartesianas.
    Devuelve:
        T: matriz de 3 x 3 que, en coordenadas homogéneas, lleva el centroide de los puntos
           al origen y los escala para que su distancia media al origen sea sqrt(2).
    '''
    centroide = np.mean(puntos, axis=0)
    distancia_media = np.mean(np.sqrt(np.sum((puntos - centroide)**2, axis=1)))
    escala = np.sqrt(2) / distancia_media
    return np.array([[escala, 0, -escala*centroide[0]],
                     [0, escala, -escala*centroide[1]],
                     [0, 0, 1]])


def aplicarTransformacion(T, puntos):
    '''
    Aplica una transformación proyectiva a un conjunto de puntos del plano.
    Parámetros:
        T: matriz de 3 x 3 (normalización u homografía).
        puntos: array de n x 2 con coordenadas cartesianas.
    Devuelve:
        array de n x 2 con los puntos transformados, en coordenadas cartesianas.
    '''
    # Pasamos a coordenadas homogéneas agregando una columna de unos
    puntos_homogeneos = np.hstack([puntos, np.ones((puntos.shape[0], 1))])

    # Aplicamos T a cada punto (como columnas) y volvemos a dejar un punto por fila
    transformados = trasponer(mult_matricial(T, trasponer(puntos_homogeneos)))

    # Dividimos por la tercera coordenada para volver a coordenadas cartesianas
    return transformados[:, :2] / transformados[:, 2:]


def obtenerQ2d(imgPts, objPts):
    '''
    Arma la matriz Q de la ecuación 11 del enunciado, cuyo núcleo contiene a la homografía vectorizada
    h = (h1, h2, h3), donde h1^T, h2^T, h3^T son las filas de H.
    Parámetros:
        imgPts: array de n x 2 con las coordenadas (u, v) de los puntos en la imagen.
        objPts: array de n x 2 con las coordenadas (U, V) de los mismos puntos en el plano del tablero (W = 0).
    Devuelve:
        Q: matriz de 2n x 9.
    '''
    n = imgPts.shape[0]
    Q = np.zeros((2*n, 9))
    cero = np.zeros(3)

    for i in range(n):
        # Punto del tablero en coordenadas homogéneas: P_i = (U_i, V_i, 1)
        P_i = np.array([objPts[i, 0], objPts[i, 1], 1])
        u_i, v_i = imgPts[i]

        # Cada correspondencia aporta dos filas, con los bloques de la ecuación 11:
        #   [ P_i^T   0^T    -u_i P_i^T ]
        #   [ 0^T     P_i^T  -v_i P_i^T ]
        Q[2*i]     = np.hstack([P_i, cero, -u_i*P_i])
        Q[2*i + 1] = np.hstack([cero, P_i, -v_i*P_i])

    return Q


def estimarHomografia(imgPts, objPts):
    '''
    Estima la homografía H que lleva los puntos del tablero a la imagen, usando DLT con normalización isotrópica.
    Parámetros:
        imgPts: array de n x 2 con las coordenadas (u, v) en la imagen.
        objPts: array de n x 2 con las coordenadas (U, V) en el tablero.
    Devuelve:
        H: matriz de 3 x 3 con H[2, 2] = 1.
    '''
    # Normalizamos ambos conjuntos de puntos
    T_img = matrizNormalizacion(imgPts)
    T_world = matrizNormalizacion(objPts)
    imgPts_norm = aplicarTransformacion(T_img, imgPts)
    objPts_norm = aplicarTransformacion(T_world, objPts)

    # DLT sobre los puntos normalizados: h es el autovector de Q^T Q con menor autovalor
    Q = obtenerQ2d(imgPts_norm, objPts_norm)
    h = autovectorMinimo(mult_matricial(trasponer(Q), Q))
    H_norm = h.reshape(3, 3)

    # Deshacemos la normalización (ecuación 14): H = T_img^-1 H_norm T_world
    H = mult_matricial(mult_matricial(inversa(T_img), H_norm), T_world)
    return H / H[2, 2]


def vectorV(H, i, j):
    '''
    Arma el vector v_ij que cumple c_i^T B c_j = v_ij^T b, siendo c_i y c_j columnas de H.
    Parámetros:
        H: homografía de 3 x 3.
        i, j: índices (0, 1 o 2) de las columnas de H.
    Devuelve:
        vector de largo 6.
    '''
    ci, cj = H[:, i], H[:, j]
    return np.array([ci[0]*cj[0],
                     ci[0]*cj[1] + ci[1]*cj[0],
                     ci[1]*cj[1],
                     ci[2]*cj[0] + ci[0]*cj[2],
                     ci[2]*cj[1] + ci[1]*cj[2],
                     ci[2]*cj[2]])


def armarA(Hlst):
    '''
    Arma la matriz A del sistema A b = 0, con dos filas por cada homografía.
    Parámetros:
        Hlst: lista de N homografías de 3 x 3, una por imagen.
    Devuelve:
        A: matriz de 2N x 6.
    '''
    filas = []
    for H in Hlst:
        # Llevamos H a norma 1 para que todas las imágenes pesen lo mismo en el sistema
        H = H / norma(H.reshape(9), 2)
        filas.append(vectorV(H, 0, 1))                    # r1^T r2 = 0
        filas.append(vectorV(H, 0, 0) - vectorV(H, 1, 1))  # ||r1|| = ||r2||
    return np.array(filas)


def cholesky(B):
    '''
    Calcula la factorización de Cholesky B = L L^T a partir de la factorización LDV del módulo ALC.
    Parámetros:
        B: matriz simétrica definida positiva de n x n.
    Devuelve:
        L: matriz triangular inferior con diagonal positiva.
    '''
    # Como B es simétrica, B = L' D L'^T, con L' triangular inferior con unos en la diagonal
    L_unos, D, _ = calculaLDV(B)

    # Como B es definida positiva, D tiene diagonal positiva y L = L' sqrt(D)
    raiz_D = np.zeros(D.shape)
    for i in range(D.shape[0]):
        raiz_D[i, i] = np.sqrt(D[i, i])
    return mult_matricial(L_unos, raiz_D)


def calcularMatrizK2D(A):
    '''
    Calcula la matriz de parámetros intrínsecos K a partir de la matriz A del método de Zhang.
    Parámetros:
        A: matriz de 2N x 6 armada con armarA.
    Devuelve:
        K: matriz de 3 x 3 triangular superior con K[2, 2] = 1.
    '''
    # b es el autovector de A^T A con menor autovalor (Ejercicio 1 b)
    B11, B12, B22, B13, B23, B33 = autovectorMinimo(mult_matricial(trasponer(A), A))
    B = np.array([[B11, B12, B13],
                  [B12, B22, B23],
                  [B13, B23, B33]])

    # El signo de b es arbitrario: elegimos el que hace a B definida positiva
    if B[0, 0] < 0:
        B = -B

    # B = L L^T con L = K^-T, entonces K = (L^T)^-1
    L = cholesky(B)
    K = inversa(trasponer(L))

    # B se conoce a menos de una escala: normalizamos para que K[2, 2] = 1
    return K / K[2, 2]


def productoVectorial(a, b):
    '''
    Calcula el producto vectorial entre dos vectores de R^3.
    Parámetros:
        a, b: vectores de largo 3.
    Devuelve:
        vector de largo 3 perpendicular a a y a b.
    '''
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])


def ortonormalizarRotacion(r1, r2):
    '''
    Arma una matriz de rotación a partir de r1 y r2 aproximados, usando Gram-Schmidt.
    Por el ruido, r1 y r2 no salen exactamente perpendiculares ni de norma 1.
    Parámetros:
        r1, r2: vectores de largo 3 (las dos primeras columnas de R, con ruido).
    Devuelve:
        R: matriz de rotación de 3 x 3 (R^T R = I y det(R) = 1).
    '''
    # Normalizamos r1
    q1 = r1 / norma(r1, 2)

    # A r2 le sacamos la parte que tiene en la dirección de q1 (así queda perpendicular) y lo normalizamos
    q2 = r2 - producto_int(r2, q1) * q1
    q2 = q2 / norma(q2, 2)

    # La tercera columna con el producto vectorial: es perpendicular a q1 y q2 y deja det(R) = 1
    q3 = productoVectorial(q1, q2)

    R = np.zeros((3, 3))
    R[:, 0], R[:, 1], R[:, 2] = q1, q2, q3
    return R


def obtenerPose2D(Hj, K):
    '''
    Calcula la pose (R, t) de una vista a partir de su homografía y de la matriz de intrínsecos.
    Parámetros:
        Hj: homografía de 3 x 3 que lleva los puntos (U, V, 1) del tablero a la imagen.
        K: matriz de parámetros intrínsecos de 3 x 3.
    Devuelve:
        R: matriz de rotación de 3 x 3.
        t: vector de traslación de largo 3, en las mismas unidades que las coordenadas del tablero.
    '''
    K_inv = inversa(K)

    # Columnas de H llevadas al sistema de la cámara: K^-1 c_k = s r_1, s r_2, s t
    K_inv_c1 = multMatVec(K_inv, Hj[:, 0])
    K_inv_c2 = multMatVec(K_inv, Hj[:, 1])
    K_inv_c3 = multMatVec(K_inv, Hj[:, 2])

    # La escala sale de que r_1 tiene norma 1
    escala = 1 / norma(K_inv_c1, 2)

    # Elegimos el signo que deja el tablero delante de la cámara (t_z > 0)
    if K_inv_c3[2] < 0:
        escala = -escala

    r1 = escala * K_inv_c1
    r2 = escala * K_inv_c2
    t = escala * K_inv_c3

    # Por el ruido, r1 y r2 no salen exactamente ortonormales: armamos R con Gram-Schmidt y r3 = r1 x r2
    # (restricción de rotación, sección 2.2 h)
    R = ortonormalizarRotacion(r1, r2)

    return R, t
