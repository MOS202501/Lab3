import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PARTE A: Implementación de Gradiente descendente en 3-D
# =============================================================================

# 1. Calcular analíticamente el gradiente de la función de pérdida
# La función de pérdida está definida como:
#   L(x,y) = (x - 2)^2 + (y + 1)^2



# Definimos la función de pérdida y su gradiente
def loss_function(x, y):
    """Calcula el valor de la función de pérdida en un punto (x, y)."""
    return (x - 2)**2 + (y + 1)**2

# 2. Calcular el gradiente de la función de pérdida
# El gradiente es el vector de derivadas parciales:
# dL/dx = 2*(x - 2)
# dL/dy = 2*(y + 1)

def gradient(x, y):
    """Calcula el gradiente de la función de pérdida en (x, y)."""
    dL_dx = 2 * (x - 2)
    dL_dy = 2 * (y + 1)
    return np.array([dL_dx, dL_dy])

# 3.Implementar el algoritmo de Gradiente Descendente
def gradient_descent(initial_point, alpha=0.25, tol=1e-6, max_iter=1000):

    """
    Aplica el método de Descenso del Gradiente para minimizar la función de pérdida.

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
        grad = gradient(xk[0], xk[1])

        #Criterio de parada: si la norma del gradiente es menor que la tolerancia, se detiene
        if np.linalg.norm(grad) < tol:
            break

        #Actualización según Descenso del gradiente:
        #   x_{k+1} = x_k - alpha * grad
        xk = xk - alpha *grad
        # Almacenar el punto actualizado
        path.append(xk.copy())
    # Retornar la solución final, la trayectoria y el número de iteraciones
    return xk, np.array(path), i+1

# =============================================================================
# PASO 3: Experimentar con diferentes valores de α
# =============================================================================
# Parámetros iniciales
plt.figure(figsize=(10, 6))
initial_point = (0, 0)  
alpha_values = [0.1, 0.2,0.3,0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1.0]  # Diferentes tasas de aprendizaje
results = {}

for alpha in alpha_values:
    solution, path, iterations = gradient_descent(initial_point, alpha)
    results[alpha] = (solution, path, iterations)
    #4. Graficar la trayectoria de los parámetros durante la optimización.
    plt.plot(path[:,0], path[:,1], 'o-', label=f"α={alpha} ({iterations} it.)")
    
    # Mostrar el valor final obtenido
    print(f"α={alpha}: solución obtenida = {solution}, iteraciones = {iterations}")



# Encontrar el mejor resultado basado en la menor cantidad de iteraciones y proximidad al óptimo
optimal_alpha = min(results.keys(), key=lambda a: (results[a][2], np.linalg.norm(results[a][0] - np.array([2, -1]))))
optimal_solution, optimal_path, optimal_iterations = results[optimal_alpha]

# =============================================================================
# PASO 5 y 6: Destacar el valor óptimo final y analizar sensibilidad a α
# =============================================================================

# Imprimir el resultado óptimo encontrado
print(f"Mejor alpha: {optimal_alpha}")
print(f"Solución óptima encontrada: {optimal_solution}")
print(f"Número de iteraciones: {optimal_iterations}")
print(f"Distancia al valor analítico (2,-1): {np.linalg.norm(optimal_solution - np.array([2, -1]))}")
print("\nAnálisis de sensibilidad:")
print("- Valores lejanos al α optimo (>α+0.3) hacen que la convergencia sea lenta.")
print("- Valores moderados de α (α+0.1 a α+0.2) permiten una convergencia rápida y estable.")
print("- Valores grandes de α (α>1.0) pueden causar oscilaciones o divergencia. Nunca converge")
print("Para seleccionar optimamente el α podemos experimentar con diferentes valores y escoger el que nos de la menor distancia al valor analítico en menor número de iteraciones")

# Graficar el mínimo global en (2, -1)
plt.scatter(2, -1, c='black', marker='x', s=100, label="Mínimo (2, -1)")

# Configuración del gráfico
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trayectorias del Gradiente Descendente para Diferentes α")
plt.legend()
plt.grid()
plt.savefig('plots/trayectorias_problema4.png')
plt.show()
plt.close()


# =============================================================================
# PARTE B: Comparación entre Newton-Raphson y Gradiente Descendente
# =============================================================================

# Definición de la función f(x,y,z)

def f_xy(x,y):
    return (x - 2)**2 * (y + 2)**2 + (x + 1)**2 + (y - 1)**2

# Calculamos el gradiente analítico
def grad_f(x, y):
    df_dx = 2*(x - 2)*(y + 2)**2 + 2*(x + 1)
    df_dy = 2*(x - 2)**2*(y + 2) + 2*(y - 1)
    return np.array([df_dx, df_dy])

# Calculamos la Hessiana analítica
def hess_f(x, y):
    d2f_dx2 = 2*(y + 2)**2 + 2
    d2f_dy2 = 2*(x - 2)**2 + 2
    d2f_dxdy = 4*(x - 2)*(y + 2)
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

# Implementamos el método de Gradiente Descendente
def gradient_descent2(initial_point, alpha, tol=1e-6, max_iter=1000):

    """
    Aplica el método de Descenso del Gradiente para minimizar la función de pérdida.

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
        grad = grad_f(xk[0], xk[1])

        # Verificar si el gradiente es NaN o infinito
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            print(f"Iteración {i}: Gradiente inválido, deteniendo iteraciones.")
            break

        #Criterio de parada: si la norma del gradiente es menor que la tolerancia, se detiene
        if np.linalg.norm(grad) < tol:
            break

        # Evitar que los valores exploten
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e6:  # Límite arbitrario para detectar inestabilidad
            break

        #Actualización según Descenso del gradiente:
        #   x_{k+1} = x_k - alpha * grad
        xk = xk - alpha *grad
        # Almacenar el punto actualizado
        path.append(xk.copy())
    # Retornar la solución final, la trayectoria y el número de iteraciones
    return xk, np.array(path), i+1


# Implementamos el método de Newton-Raphson
def newton_raphson(initial_point, alpha, tol=1e-6, max_iter=1000):
    """
    Aplica el método de Newton-Raphson 

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
        grad = grad_f(xk[0], xk[1])

        # 4. Criterio de parada: si la norma del gradiente es menor que la tolerancia, se detiene
        if np.linalg.norm(grad) < tol:
            break

        # Calcular la matriz Hessiana en el punto actual
        H = hess_f(xk[0], xk[1])

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

# Parámetros iniciales
x0 = [-2, -3]

alpha_values = [0.1, 0.2,0.3,0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1.0]  # Diferentes tasas de aprendizaje
results2 = {}
results3 = {}

for alpha in alpha_values:
    solution2, path2, iterations2 = gradient_descent2(x0, alpha)
    solution_rosen, path_rosen, iter_rosen = newton_raphson(
    x0,alpha)

    results2[alpha] = (solution2, path2, iterations2)
    results3[alpha] = (solution_rosen,path_rosen,iter_rosen)

    # Mostrar el valor final obtenido
    print(f"α={alpha}: solución obtenida para descenso del gradiente = {solution2}, iteraciones = {iterations2}")
    print(f"α={alpha}: solución obtenida para Newton-Raphson = {solution_rosen}, iteraciones = {iter_rosen}")


print("En este caso, no hay un valor optimo. Dependiendo del algoritmo que usemos y el alpha, va a cambiar nuestro resultado."
"Esto se debe a que nuestra funcion presenta un punto de silla. Además, en descenso del gradiente para alphas mayores a 0.1, el gradiente"
" es demasiado grande y en cada iteración sigue creciendo, no convergiendo a nada. ")

x0 = [-2, -3]
alpha_gd = 0.1
alpha_nr = 1.0

_, path_gd, _ = gradient_descent(x0, alpha=alpha_gd)
_, path_nr, _ = newton_raphson(x0, alpha=alpha_nr)

path_nr = np.array(path_nr)  # Convertir la lista en un array de NumPy
path_gd = np.array(path_gd)

# Crear malla para graficar
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f_xy(X, Y)

# Figura 3D con trayectorias
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
ax.plot(path_gd[:, 0], path_gd[:, 1], f_xy(path_gd[:, 0], path_gd[:, 1]), 'ro-', label='Gradiente Descendente')
ax.plot(path_nr[:, 0], path_nr[:, 1], f_xy(path_nr[:, 0], path_nr[:, 1]), 'bo-', label='Newton-Raphson')
ax.scatter(x0[0], x0[1], f_xy(x0[0], x0[1]), color='black', marker='s', s=100, label='Inicio')
ax.scatter(path_gd[-1, 0], path_gd[-1, 1], f_xy(path_gd[-1, 0], path_gd[-1, 1]), color='yellow', marker='*', s=150, label='Final GD')
ax.scatter(path_nr[-1, 0], path_nr[-1, 1], f_xy(path_nr[-1, 0], path_nr[-1, 1]), color='green', marker='*', s=150, label='Final NR')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("F(X, Y)")
ax.set_title("Superficie y Trayectorias de Optimización")
ax.legend()

# Convergencia del error en escala logarítmica
ax2 = fig.add_subplot(122)
optimum = path_nr[-1]
error_gd = np.linalg.norm(path_gd - optimum, axis=1)
error_nr = np.linalg.norm(path_nr - optimum, axis=1)
ax2.plot(range(len(error_gd)), error_gd, 'r-', label='Gradiente Descendente')
ax2.plot(range(len(error_nr)), error_nr, 'b-', label='Newton-Raphson')
ax2.set_yscale("log")
ax2.set_xlabel("Iteraciones")
ax2.set_ylabel("Error (distancia al óptimo)")
ax2.set_title("Convergencia del Error")
ax2.legend()
ax2.grid()
plt.tight_layout()
plt.savefig('plots/convergencia_problema_4.png')
plt.show()

