import numpy as np
import matplotlib.pyplot as plt

# Función original


def f(x):
    return 3*x**3 - 10*x**2 - 56*x + 50  # calcular f(x)

# Derivada de f(x)


def f_prime(x):
    return 9*x**2 - 20*x - 56  # calcular f'(x) [hecho a mano]

# Segunda derivada de f(x)


def f_double_prime(x):
    return 18*x - 20  # calcular f''(x) [hecho a mano]

# Método de Newton-Raphson para hallar extremos


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


def plot_extrema(extremos_unicos):
    x_vals = np.linspace(-6, 6, 400)  # generar valores de x para graficar
    y_vals = f(x_vals)  # calcular f(x) para esos valores
    plt.figure(figsize=(10, 6))  # configurar tamaño de la figura
    plt.plot(x_vals, y_vals, label="f(x)", color="black")  # graficar f(x)

    # definir colores según el tipo de extremo
    colores = {"mínimo": "red", "máximo": "blue", "indeterminado": "green"}

    # Graficar cada extremo encontrado
    for x_val, y_val, tipo, iter_count in extremos_unicos:
        plt.plot(x_val, y_val, 'o', color=colores.get(
            tipo, "black"), markersize=8)  # marcar el extremo
        plt.text(x_val, y_val, f" {tipo}\n({x_val:.2f}, {y_val:.2f})", color=colores.get(tipo, "black"),
                 fontsize=9, verticalalignment='bottom')  # anotar el extremo

    plt.xlabel("x")  # etiquetar eje x
    plt.ylabel("f(x)")  # etiquetar eje y
    # título del gráfico
    plt.title("Newton-Raphson: Extremos de f(x) = 3x³ - 10x² - 56x + 50")
    plt.legend()  # mostrar leyenda
    plt.grid(True)  # mostrar cuadrícula
    plt.savefig("plots/problema_1.png")  # guardar gráfico en un archivo


if __name__ == "__main__":
    # puntos iniciales para el método
    initial_points = [-6, -4, -2, 0, 2, 4, 6]
    extrema = []  # lista para almacenar resultados

    print("Resultados de Newton-Raphson:")
    # Iterar sobre cada punto inicial
    for x0 in initial_points:
        # obtener extremo y número de iteraciones
        x_star, iterations = newton_raphson(x0)
        # calcular segunda derivada en x*
        second_deriv = f_double_prime(x_star)
        # Clasificar el extremo según el signo de la segunda derivada
        if second_deriv > 0:
            tipo = "mínimo"
        elif second_deriv < 0:
            tipo = "máximo"
        else:
            tipo = "indeterminado"
        # agregar resultado a la lista
        extrema.append((x_star, f(x_star), tipo, iterations))
        print(
            f"x0 = {x0:>3}  -->  x* = {x_star:>8.6f} , f(x*) = {f(x_star):>8.6f}  -->  {tipo} (iteraciones: {iterations})")

    # Filtrar para eliminar duplicados (varios x0 pueden converger al mismo extremo)
    extremos_unicos = {}
    for x_val, y_val, tipo, iter_count in extrema:
        key = round(x_val, 6)  # redondear para identificar duplicados
        extremos_unicos[key] = (x_val, y_val, tipo, iter_count)
    extremos_unicos = list(extremos_unicos.values())

    plot_extrema(extremos_unicos)  # llamar función para graficar

    """
    Resultados de Newton-Raphson:
        x0 =  -6  -->  x* = -1.619601 , f(x*) = 101.721420  -->  máximo (iteraciones: 23)
        x0 =  -4  -->  x* = -1.619601 , f(x*) = 101.721420  -->  máximo (iteraciones: 22)
        x0 =  -2  -->  x* = -1.619601 , f(x*) = 101.721420  -->  máximo (iteraciones: 19)
        x0 =   0  -->  x* = -1.619601 , f(x*) = 101.721420  -->  máximo (iteraciones: 18)
        x0 =   2  -->  x* = 3.841824 , f(x*) = -142.626770  -->  mínimo (iteraciones: 20)
        x0 =   4  -->  x* = 3.841824 , f(x*) = -142.626770  -->  mínimo (iteraciones: 18)
        x0 =   6  -->  x* = 3.841824 , f(x*) = -142.626770  -->  mínimo (iteraciones: 21)

    Con un alpha muy chiquito (0.002 en vez de 0.6) da:
    Resultados de Newton-Raphson:
        x0 =  -6  -->  x* = -1.619601 , f(x*) = 101.721420  -->  máximo (iteraciones: 9879)
        x0 =  -4  -->  x* = -1.619601 , f(x*) = 101.721420  -->  máximo (iteraciones: 9461)
        x0 =  -2  -->  x* = -1.619601 , f(x*) = 101.721420  -->  máximo (iteraciones: 8398)
        x0 =   0  -->  x* = -1.619601 , f(x*) = 101.721420  -->  máximo (iteraciones: 8912)
        x0 =   2  -->  x* = 3.841823 , f(x*) = -142.626770  -->  mínimo (iteraciones: 8946)
        x0 =   4  -->  x* = 3.841824 , f(x*) = -142.626770  -->  mínimo (iteraciones: 7940)
        x0 =   6  -->  x* = 3.841824 , f(x*) = -142.626770  -->  mínimo (iteraciones: 9398)

    Llega a los mismos resultados, pero tarda mucho más en converger.
    """
