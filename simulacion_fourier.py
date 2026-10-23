import numpy as np
import matplotlib.pyplot as plt

# ACTIVIDAD FORMATIVA 2
# Simulacion y analisis de senales con la Transformada de Fourier

# Crear el dominio del tiempo
N = 1000
t = np.linspace(-1, 1, N, endpoint=False)
dt = t[1] - t[0]
frecuencia_muestreo = 1 / dt

# Definir tres senales elementales
pulso = np.where(np.abs(t) < 0.2, 1.0, 0.0)
escalon = np.where(t >= 0, 1.0, 0.0)
senoidal = np.sin(2 * np.pi * 5 * t)

senales = [
    ("Pulso rectangular", pulso),
    ("Funcion escalon", escalon),
    ("Senal senoidal", senoidal)
]

# Graficar cada senal y su Transformada de Fourier
fig, ejes = plt.subplots(3, 3, figsize=(14, 10))

for fila, (nombre, x) in enumerate(senales):
    # Transformada rapida de Fourier
    X = np.fft.fft(x)
    frecuencias = np.fft.fftfreq(N, d=dt)

    # Centrar el espectro en frecuencia cero
    X_centrado = np.fft.fftshift(X)
    f_centrada = np.fft.fftshift(frecuencias)

    # Normalizar la magnitud y calcular la fase
    magnitud = np.abs(X_centrado) / N
    fase = np.angle(X_centrado)

    # Grafica en el dominio del tiempo
    ejes[fila, 0].plot(t, x)
    ejes[fila, 0].set_title(nombre)
    ejes[fila, 0].set_xlabel("Tiempo (s)")
    ejes[fila, 0].set_ylabel("Amplitud")
    ejes[fila, 0].grid(True)

    # Espectro de magnitud
    ejes[fila, 1].plot(f_centrada, magnitud)
    ejes[fila, 1].set_title("Magnitud de Fourier")
    ejes[fila, 1].set_xlabel("Frecuencia (Hz)")
    ejes[fila, 1].set_ylabel("Magnitud")
    ejes[fila, 1].set_xlim(-30, 30)
    ejes[fila, 1].grid(True)

    # Espectro de fase
    ejes[fila, 2].plot(f_centrada, fase)
    ejes[fila, 2].set_title("Fase de Fourier")
    ejes[fila, 2].set_xlabel("Frecuencia (Hz)")
    ejes[fila, 2].set_ylabel("Fase (rad)")
    ejes[fila, 2].set_xlim(-30, 30)
    ejes[fila, 2].grid(True)

plt.tight_layout()
plt.savefig("resultados_fourier.png", dpi=150)
plt.show()

# Verificar la propiedad de linealidad
x1 = np.sin(2 * np.pi * 5 * t)
x2 = np.cos(2 * np.pi * 10 * t)
a = 2
b = 3

transformada_combinada = np.fft.fft(a * x1 + b * x2)
combinacion_transformadas = a * np.fft.fft(x1) + b * np.fft.fft(x2)

print("¿Se cumple la linealidad?")
print(np.allclose(transformada_combinada, combinacion_transformadas))

# Analizar el desplazamiento temporal
desplazamiento = 0.1
x_desplazada = np.sin(2 * np.pi * 5 * (t - desplazamiento))

plt.figure(figsize=(10, 4))
plt.plot(t, x1, label="Senal original")
plt.plot(t, x_desplazada, "--", label="Senal desplazada")
plt.title("Propiedad de desplazamiento en el tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("desplazamiento_temporal.png", dpi=150)
plt.show()

print("Frecuencia de muestreo:", frecuencia_muestreo, "Hz")
print("Se guardaron las graficas de resultados.")