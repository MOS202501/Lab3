import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para la gráfica 3D

# =============================================================================
# PARTE A: Función de Rosenbrock (3D)
# =============================================================================

# 1. Calcular analíticamente el gradiente y la matriz Hessiana de f(x,y)
# La función de Rosenbrock está definida como:
#   f(x,y) = (x - 1)^2 + 100*(y - x^2)^2


def rosenbrock(x, y):
    return (x - 1)**2 + 100*(y - x**2)**2

# Gradiente analítico:
#   df/dx = 2*(x - 1) - 400*x*(y - x^2)
#   df/dy = 200*(y - x^2)


def grad_rosenbrock(x, y):
    df_dx = 2 * (x - 1) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

# Matriz Hessiana analítica:
#   f_xx = 2 - 400*y + 1200*x^2
#   f_xy = f_yx = -400*x
#   f_yy = 200


def hess_rosenbrock(x, y):
    f_xx = 2 - 400 * y + 1200 * x**2
    f_xy = -400 * x
    f_yy = 200
    return np.array([[f_xx, f_xy],
                     [f_xy, f_yy]])

# 2. Implementar el método de Newton-Raphson para funciones bidimensionales


def newton_raphson_rosenbrock(initial_point, alpha=0.25, tol=1e-6, max_iter=1000):
    """
    Aplica el método de Newton-Raphson para minimizar la función de Rosenbrock.

    Parámetros:
      initial_point : Punto inicial (x0, y0).
      alpha         : Factor de convergencia (tamaño del paso).
      tol           : Tolerancia para la norma del gradiente.
      max_iter      : Número máximo de iteraciones.

    Retorna:
      - La solución aproximada.
      - Lista de puntos iterativos (trayectoria) para graficar.
      - Número de iteraciones realizadas.
    """
    # Inicializar el punto
    xk = np.array(initial_point, dtype=float)
    # Almacenar la trayectoria para análisis posterior
    path = [xk.copy()]

    # Iterar hasta que se cumpla el criterio de parada o se alcance el máximo de iteraciones
    for i in range(max_iter):
        # Calcular el gradiente en el punto actual
        grad = grad_rosenbrock(xk[0], xk[1])

        # 4. Criterio de parada: si la norma del gradiente es menor que la tolerancia, se detiene
        if np.linalg.norm(grad) < tol:
            break

        # Calcular la matriz Hessiana en el punto actual
        H = hess_rosenbrock(xk[0], xk[1])

        # Verificar que la Hessiana sea invertible
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("La Hessiana es singular en la iteración", i)
            break

        # 2. Actualización según Newton-Raphson:
        #   x_{k+1} = x_k - alpha * H_inv * grad
        xk = xk - alpha * H_inv.dot(grad)
        # Almacenar el punto actualizado
        path.append(xk.copy())

    return xk, path, i+1


# 3. Utilizar como punto inicial (x0, y0) = (0, 10)
initial_point_rosen = (0, 10)
solution_rosen, path_rosen, iter_rosen = newton_raphson_rosenbrock(
    initial_point_rosen)

# Imprimir resultados de la parte a
print("=== PARTE A: Función de Rosenbrock (3D) ===")
print("Punto inicial:", initial_point_rosen)
print("Solución encontrada:", solution_rosen)
print("Número de iteraciones:", iter_rosen)
print("Error en x respecto a 1:", abs(solution_rosen[0]-1))
print("Error en y respecto a 1:", abs(solution_rosen[1]-1))

# 4. Graficar la superficie z = f(x,y) en el espacio tridimensional
# 5. Representar los puntos iterativos sobre la superficie y destacar el mínimo final en color rojo
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Crear una malla de puntos para graficar la superficie
X = np.linspace(-1, 2, 100)
Y = np.linspace(-1, 12, 100)
X, Y = np.meshgrid(X, Y)
Z = rosenbrock(X, Y)

# Graficar la superficie con un mapa de colores
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Convertir la trayectoria iterativa a un arreglo NumPy para facilitar la graficación
path_rosen = np.array(path_rosen)
# Trazar la trayectoria iterativa en color azul
ax.plot(path_rosen[:, 0], path_rosen[:, 1], rosenbrock(path_rosen[:, 0], path_rosen[:, 1]),
        color='blue', marker='o', markersize=5, linewidth=2, label='Trayectoria iterativa')

# 5. Destacar únicamente el mínimo final en color rojo
ax.scatter(solution_rosen[0], solution_rosen[1],
           rosenbrock(solution_rosen[0], solution_rosen[1]),
           color='red', s=100, label='Mínimo final')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('Superficie de Rosenbrock y trayectoria del método Newton-Raphson')
ax.legend()
plt.savefig('plots/Superficie_problema_3.png')

# 6. Analizar la convergencia hacia el mínimo conocido (1,1)
# Se imprime el error final respecto al mínimo teórico (1,1)
print("\nAnálisis de convergencia para la función de Rosenbrock:")
print("Error en x:", abs(solution_rosen[0]-1))
print("Error en y:", abs(solution_rosen[1]-1))


# =============================================================================
# PARTE B: Función en 4D
# =============================================================================

# 1. Formular matemáticamente el algoritmo de Newton-Raphson en R^4:
#    Para una función f: R^4 -> R, el algoritmo se formula como:
#       x_{k+1} = x_k - H(x_k)^{-1} * ∇f(x_k)
#
# 2. Calcular analíticamente el gradiente y la matriz Hessiana de la función propuesta.
#    Consideramos la función en R^4:
#       f(x, y, z, w) = (x-1)^2 + (y-2)^2 + (z-3)^2 + (w-4)^2
#    Su gradiente es:
#       ∇f = [2(x-1), 2(y-2), 2(z-3), 2(w-4)]
#    Y su Hessiana es:
#       H = 2 * I_4 (matriz identidad 4x4 multiplicada por 2)


# Definición de la función f(x,y,z)

def f_xyz(x):
    # x es un vector de 3 componentes: [x, y, z]
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-3)**2

# Cálculo del gradiente analítico de f(x,y,z)


def grad_f_xyz(x):
    # ∇f = [2(x-1), 2(y-2), 2(z-3)]
    return 2 * np.array([x[0]-1, x[1]-2, x[2]-3])

# Cálculo de la matriz Hessiana de f(x,y,z)


def hess_f_xyz(x):
    # La Hessiana es constante: H = 2*I (matriz identidad 3x3)
    return 2 * np.eye(3)

# Implementación del método de Newton-Raphson para funciones de R^3


def newton_raphson_3d(initial_point, alpha=1.0, tol=1e-6, max_iter=100):
    """
    Aplica el método de Newton-Raphson para minimizar f(x,y,z).

    Parámetros:
      initial_point: Vector inicial [x0, y0, z0].
      alpha        : Factor de convergencia.
      tol          : Tolerancia basada en la norma del gradiente.
      max_iter     : Número máximo de iteraciones.

    Retorna:
      - La solución aproximada.
      - Lista de puntos iterativos (trayectoria).
      - Número de iteraciones realizadas.
    """
    xk = np.array(initial_point, dtype=float)
    path = [xk.copy()]  # Almacena la trayectoria iterativa
    for i in range(max_iter):
        grad = grad_f_xyz(xk)
        # Criterio de parada: si la norma del gradiente es menor que la tolerancia, se detiene
        if np.linalg.norm(grad) < tol:
            break
        H = hess_f_xyz(xk)
        # Dado que H es constante e igual a 2*I, su inversa es (1/2)*I
        H_inv = np.linalg.inv(H)
        # Actualización de Newton-Raphson
        xk = xk - alpha * H_inv.dot(grad)
        path.append(xk.copy())
    return xk, path, i+1


# Selección de un punto inicial en R^3, por ejemplo [0,0,0]
initial_point_3d = [0, 0, 0]
solution_3d, path_3d, iter_3d = newton_raphson_3d(initial_point_3d)

# Imprimir los resultados
print("Newton-Raphson para f(x,y,z) = (x-1)^2+(y-2)^2+(z-3)^2")
print("Punto inicial:", initial_point_3d)
print("Solución encontrada:", solution_3d)
print("Número de iteraciones:", iter_3d)


# 5. Representar la convergencia mediante la gráfica de la norma del gradiente vs iteración
grad_norms = [np.linalg.norm(grad_f_xyz(point)) for point in path_3d]
plt.figure(figsize=(8, 6))
plt.plot(range(len(grad_norms)), grad_norms, marker='o', linestyle='-')
plt.xlabel('Iteración')
plt.ylabel('Norma del gradiente')
plt.title('Convergencia del método Newton-Raphson para f(x,y,z)')
plt.grid(True)
plt.savefig('plots/convergencia_problema_3.png')
plt.show()

# 6. Discusión de dificultades computacionales en alta dimensión:
"""
En este ejemplo la función es muy simple (la Hessiana es constante y diagonal),
lo que permite una implementación directa. En problemas reales en alta dimensión,
el cálculo y la inversión de la Hessiana puede resultar computacionalmente costoso,
además de posibles problemas de inestabilidad numérica y manejo de matrices mal condicionadas."
"""
