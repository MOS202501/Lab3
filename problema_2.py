import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Definición de la función y sus derivadas
# =============================================================================


def f(x):
    """
    Función original: f(x) = x^5 - 8x^3 + 10x + 6
    """
    return x**5 - 8*x**3 + 10*x + 6


def f_prime(x):
    """
    Primera derivada de f(x): f'(x) = 5x^4 - 24x^2 + 10
    """
    return 5*x**4 - 24*x**2 + 10


def f_double_prime(x):
    """
    Segunda derivada de f(x): f''(x) = 20x^3 - 48x
    """
    return 20*x**3 - 48*x

# =============================================================================
# Implementación del método de Newton-Raphson
# =============================================================================


def newton_raphson(x0, alpha=0.6, tol=1e-6, max_iter=10000):
    x = x0  # valor inicial
    for i in range(max_iter):
        deriv = f_prime(x)  # calcular f'(x)
        deriv2 = f_double_prime(x)  # calcular f''(x)
        # Si la derivada es muy pequeña, consideramos convergencia
        if abs(deriv) < tol:
            break
        # Evitar división por cero en la segunda derivada
        if deriv2 == 0:
            print("La segunda derivada es cero. No se puede continuar la iteración.")
            break
        x = x - alpha * (deriv / deriv2)  # actualizar x
    return x, i  # retornar el extremo y el número de iteraciones

# =============================================================================
# Función para graficar la función, puntos críticos y extremos
# =============================================================================


def graficar_resultados():
    """
    Realiza lo siguiente:
    - Aplica el método de Newton-Raphson para distintos puntos iniciales.
    - Identifica los puntos críticos y los clasifica según la segunda derivada.
    - Considera también los extremos del intervalo [-3, 3] para determinar
      los extremos globales.
    - Genera y guarda una gráfica de f(x) con:
        • La curva de la función.
        • Los puntos críticos (mínimos y máximos locales).
        • Los extremos del intervalo.
        • La resaltes del mínimo y máximo global.
    """
    # Selección de 13 puntos iniciales equidistantes en el intervalo [-3, 3]
    initial_points = np.linspace(-3, 3, 13)
    results = []

    print("Resultados de Newton-Raphson para diferentes x0:")
    # Aplicación del método para cada valor inicial
    for x0 in initial_points:
        x_star, iterations = newton_raphson(x0, alpha=1.0)
        results.append((x_star, f(x_star), iterations))
        print(
            f"x0 = {x0:5.2f} -> x* = {x_star:8.6f}, f(x*) = {f(x_star):8.6f}, iter = {iterations}")

    # Eliminar duplicados: diferentes x0 pueden converger al mismo punto crítico
    unique_points = {}
    for x_val, fx_val, iters in results:
        key = round(x_val, 6)
        unique_points[key] = (x_val, fx_val, iters)
    unique_points = list(unique_points.values())

    # Clasificar cada punto crítico utilizando f''(x)
    critical_points = []
    for x_val, fx_val, iters in unique_points:
        second_deriv = f_double_prime(x_val)
        if second_deriv > 0:
            tipo = "mínimo local"
        elif second_deriv < 0:
            tipo = "máximo local"
        else:
            continue
        critical_points.append((x_val, fx_val, tipo))

    # Considerar los extremos del intervalo [-3, 3] como candidatos a extremos globales
    endpoints = [(-3, f(-3)), (3, f(3))]

    # Combinar los puntos críticos y los extremos para evaluar los extremos globales
    candidates = critical_points + [(x, fx, 'endpoint')
                                    for (x, fx) in endpoints]

    # Determinar el mínimo y el máximo global entre los candidatos
    global_min = min(candidates, key=lambda item: item[1])
    global_max = max(candidates, key=lambda item: item[1])

    # Mostrar los resultados obtenidos en la consola
    print("\nPuntos críticos encontrados:")
    for x_val, fx_val, t in critical_points:
        print(f"x = {x_val:8.6f}, f(x) = {fx_val:8.6f} -> {t}")

    print("\nPuntos extremos del intervalo [-3, 3]:")
    for x_val, fx_val in endpoints:
        print(f"x = {x_val:8.6f}, f(x) = {fx_val:8.6f} -> endpoint")

    print(
        f"\nEl mínimo global es: x = {global_min[0]:8.6f}, f(x) = {global_min[1]:8.6f}")
    print(
        f"El máximo global es: x = {global_max[0]:8.6f}, f(x) = {global_max[1]:8.6f}")

    # =============================================================================
    # Generación de la gráfica
    # =============================================================================

    # Creación de un conjunto denso de puntos para graficar f(x)
    x_vals = np.linspace(-3, 3, 400)
    y_vals = f(x_vals)

    # Crear la figura con un tamaño personalizado
    plt.figure(figsize=(10, 6))
    # Graficar la función f(x)
    plt.plot(x_vals, y_vals, label="f(x)", color="blue")

    # Graficar los puntos críticos (mínimos y máximos locales) con marcador circular negro
    for (x_val, fx_val, tipo) in critical_points:
        plt.plot(x_val, fx_val, 'ko', markersize=8)
        plt.text(x_val, fx_val, f" {tipo}\n({x_val:.2f}, {fx_val:.2f})",
                 fontsize=9, verticalalignment='bottom')

    # Graficar los extremos del intervalo con marcador cuadrado negro
    for (x_val, fx_val) in endpoints:
        plt.plot(x_val, fx_val, 'ks', markersize=8)
        plt.text(x_val, fx_val, f" endpoint\n({x_val:.2f}, {fx_val:.2f})",
                 fontsize=9, verticalalignment='top')

    # Resaltar el mínimo y el máximo global con marcadores rojos
    plt.plot(global_min[0], global_min[1], 'ro',
             markersize=10, label="Mínimo Global")
    plt.plot(global_max[0], global_max[1], 'ro',
             markersize=10, label="Máximo Global")

    # Configuración de etiquetas, título y leyenda de la gráfica
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Extremos locales y globales de f(x) = $x^5 - 8x^3 + 10x + 6$")
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica en un archivo PNG y mostrarla
    plt.savefig('plots/problema_2.png')
    plt.show()


def main():
    """
    Función principal que ejecuta el algoritmo:
    1. Aplica Newton-Raphson para distintos puntos iniciales.
    2. Determina los puntos críticos y extremos globales en el intervalo [-3, 3].
    3. Genera la gráfica de f(x) con la información obtenida.

    Análisis breve de los resultados:
    - Se identifican dos mínimos locales y dos máximos locales obtenidos mediante Newton-Raphson.
    - Al incluir los extremos del intervalo [-3, 3] se observa que:
        • El mínimo global se encuentra en x = -3 con f(x) = -51.
        • El máximo global se encuentra en x = 3 con f(x) = 63.
    - Esto resalta la importancia de considerar los extremos del intervalo en problemas de optimización.
    """
    graficar_resultados()


# Punto de entrada del programa
if __name__ == "__main__":
    main()


"""
Resultados de Newton-Raphson para diferentes x0:
    x0 = -3.00 -> x* = -2.083044, f(x*) = 18.258776, iter = 6
    x0 = -2.50 -> x* = -2.083044, f(x*) = 18.258776, iter = 5
    x0 = -2.00 -> x* = -2.083044, f(x*) = 18.258776, iter = 4
    x0 = -1.50 -> x* = 2.083044, f(x*) = -6.258776, iter = 6
    x0 = -1.00 -> x* = -0.678917, f(x*) = 1.570047, iter = 3
    x0 = -0.50 -> x* = -0.678917, f(x*) = 1.570047, iter = 3
    La segunda derivada es cero. No se puede continuar la iteración.
    x0 =  0.00 -> x* = 0.000000, f(x*) = 6.000000, iter = 0
    x0 =  0.50 -> x* = 0.678917, f(x*) = 10.429953, iter = 3
    x0 =  1.00 -> x* = 0.678917, f(x*) = 10.429953, iter = 3
    x0 =  1.50 -> x* = -2.083044, f(x*) = 18.258776, iter = 6
    x0 =  2.00 -> x* = 2.083044, f(x*) = -6.258776, iter = 4
    x0 =  2.50 -> x* = 2.083044, f(x*) = -6.258776, iter = 5
    x0 =  3.00 -> x* = 2.083044, f(x*) = -6.258776, iter = 6

Puntos críticos encontrados:
    x = -2.083044, f(x) = 18.258776 -> máximo local
    x = 2.083044, f(x) = -6.258776 -> mínimo local
    x = -0.678917, f(x) = 1.570047 -> mínimo local
    x = 0.678917, f(x) = 10.429953 -> máximo local

Puntos extremos del intervalo [-3, 3]:
    x = -3.000000, f(x) = -51.000000 -> endpoint
    x = 3.000000, f(x) = 63.000000 -> endpoint

    El mínimo global es: x = -3.000000, f(x) = -51.000000
    El máximo global es: x = 3.000000, f(x) = 63.000000
"""
