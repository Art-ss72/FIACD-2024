#Salcedo Salas Arturo Enrique

import segyio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft, ifft

# Definir la frecuencia de muestreo en Hz y el intervalo de muestreo (dt) en segundos
#dt = 0.000133  # Intervalo de muestreo en segundos (1/8000 Hz)
#dt = 0.00025
dt = 0.000125
frecuencia_muestreo = 1 / dt  # Frecuencia de muestreo en Hz
nyquist = frecuencia_muestreo / 2  # Frecuencia de Nyquist, la máxima frecuencia que se puede captar
visualizacion_trz = 0.2 # Duración de la visualización en segundos
num_muestras_corte = int(visualizacion_trz / dt)  # Número de muestras a cortar para la visualización

# Cargar datos reales desde un archivo .sgy sin usar la geometría

#archivo_sgy = r"C:\Users\jakem\OneDrive\Documentos\Servicio Social\Datos2\SGY\file_1.sgy"  #dt=0.000133
#archivo_sgy = r"C:\Users\jakem\OneDrive\Documentos\Servicio Social\PrimerforoIA_CD\Datos sintéticos\500.sgy"  #dt=0.00025
archivo_sgy = r"C:\Users\jakem\OneDrive\Documentos\Servicio Social\PrimerforoIA_CD\DatosSGY\1-01.sgy"  #dt=0.000125, traza nula 22, traza ruidosa 7

with segyio.open(archivo_sgy, "r", ignore_geometry=True) as archivo:  # Abrir el archivo .sgy
    datos = np.stack([archivo.trace[i][:] for i in range(len(archivo.trace))]).T  # Leer las trazas y apilarlas en un array

print("Trazas reales cargadas: ", datos.shape)  # Imprimir la forma del array de datos

# Normalizar los datos y cortar a 1 segundo
def procesar_datos(datos, num_muestras_corte):
    datos_cortados = datos[:num_muestras_corte, :]  # Cortar los datos a la longitud especificada
    scaler = StandardScaler()  # Crear un objeto StandardScaler para normalizar los datos
    datos_normalizados = scaler.fit_transform(datos_cortados)  # Normalizar los datos
    return datos_normalizados  # Devolver los datos normalizados

datos_procesados = procesar_datos(datos, num_muestras_corte)  # Procesar los datos

# Aplicar Isolation Forest a una traza con ventanas
def aplicar_isolation_forest(traza, tamano_ventana, contaminacion):
    etiquetas = np.zeros(len(traza), dtype=int)  # Inicializar un array de etiquetas con ceros
    modelo_if = IsolationForest(contamination=contaminacion, random_state=42)  # Crear el modelo Isolation Forest
    
    # Iterar sobre la traza en ventanas del tamaño especificado
    for i in range(0, len(traza) - tamano_ventana + 1, tamano_ventana):
        ventana = traza[i:i + tamano_ventana].reshape(-1, 1)  # Extraer la ventana de datos y reformatearla
        etiquetas[i:i + tamano_ventana] = modelo_if.fit_predict(ventana).flatten()  # Ajustar el modelo y predecir etiquetas
    
    return etiquetas  # Devolver las etiquetas

# Graficar serie temporal con colores
def graficar_serie_temporal(traza, etiquetas, titulo):
    tiempo = np.arange(len(traza)) * dt  # Crear un array de tiempo basado en el intervalo de muestreo
    plt.figure(figsize=(15, 5))  # Crear una figura para la gráfica
    plt.plot(tiempo, traza, label='Traza', color='blue')  # Graficar la traza en azul
    plt.scatter(tiempo[etiquetas == -1], traza[etiquetas == -1], color='red', label='Anomalías', s=10)  # Resaltar anomalías en rojo
    plt.title(titulo)  # Título de la gráfica
    plt.xlabel('Tiempo (s)')  # Etiqueta del eje x
    plt.ylabel('Amplitud')  # Etiqueta del eje y
    plt.legend()  # Mostrar leyenda
    plt.grid(True)  # Mostrar cuadrícula
    plt.show()  # Mostrar la gráfica

# Función para aplicar FFT y graficar el espectro de frecuencias
def graficar_fft_superpuesta(datos_normales, datos_anomalos, fs, datos_originales):
    plt.figure(figsize=(15, 5))  # Crear una figura para la gráfica
    
    # Graficar FFT de los datos originales
    if len(datos_originales) > 0:
        N_original = len(datos_originales)  # Número de muestras en los datos originales
        fft_original = fft(datos_originales)  # Aplicar FFT a los datos originales
        frecuencias_originales = np.fft.fftfreq(N_original, d=1/fs)  # Calcular las frecuencias correspondientes
        plt.plot(frecuencias_originales[:N_original // 2], np.abs(fft_original)[:N_original // 2], label='Datos Originales', color='black', linestyle='--')  # Graficar FFT de datos originales

    # Graficar FFT de los datos normales
    if len(datos_normales) > 0:
        N = len(datos_normales)  # Número de muestras en los datos normales
        fft_result = fft(datos_normales)  # Aplicar FFT a los datos normales
        frecuencias = np.fft.fftfreq(N, d=1/fs)  # Calcular las frecuencias correspondientes
        plt.plot(frecuencias[:N // 2], np.abs(fft_result)[:N // 2], label='Datos Normales', color='blue')  # Graficar FFT de datos normales

    # Graficar FFT de los datos anómalos
    if len(datos_anomalos) > 0:
        N = len(datos_anomalos)  # Número de muestras en los datos anómalos
        fft_result = fft(datos_anomalos)  # Aplicar FFT a los datos anómalos
        frecuencias = np.fft.fftfreq(N, d=1/fs)  # Calcular las frecuencias correspondientes
        plt.plot(frecuencias[:N // 2], np.abs(fft_result)[:N // 2], label='Datos Anómalos', color='red')  # Graficar FFT de datos anómalos
    
    plt.xscale('log')  # Establecer el eje x en escala logarítmica
    plt.title('FFT de Datos Originales, Normales y Anómalos')  # Título de la gráfica
    plt.xlabel('Frecuencia (Hz)')  # Etiqueta del eje x
    plt.ylabel('Amplitud')  # Etiqueta del eje y
    plt.legend()  # Mostrar leyenda
    plt.grid(True)  # Mostrar cuadrícula
    plt.show()  # Mostrar la gráfica

# Procesar y graficar FFT e IFFT
def graficar_fft_ifft(traza, etiquetas, fs_original):
    tiempo = np.arange(len(traza)) * dt  # Crear un array de tiempo basado en el intervalo de muestreo
    
    # Separar datos normales y anómalos
    datos_normales = traza[etiquetas == 1]
    datos_anomalos = traza[etiquetas == -1]
    
    # Recalcular el nuevo dt para cada segmento
    if len(datos_normales) > 0:
        dt_normales = len(tiempo) * dt / len(datos_normales)  # Intervalo de muestreo para datos normales
        fs_normales = 1 / dt_normales  # Frecuencia de muestreo para datos normales
    else:
        fs_normales = None
    
    if len(datos_anomalos) > 0:
        dt_anomalos = len(tiempo) * dt / len(datos_anomalos)  # Intervalo de muestreo para datos anómalos
        fs_anomalos = 1 / dt_anomalos  # Frecuencia de muestreo para datos anómalos
    else:
        fs_anomalos = None
    
    def fft_ifft(data):
        if len(data) == 0:
            return np.array([]), np.array([])  # Si no hay datos, devolver arrays vacíos
        data_fft = fft(data)  # Aplicar FFT
        data_ifft = ifft(data_fft).real  # Aplicar IFFT y tomar la parte real
        return data_fft, data_ifft
    
    # Aplicar FFT e IFFT
    fft_normales, ifft_normales = fft_ifft(datos_normales)
    fft_anomalos, ifft_anomalos = fft_ifft(datos_anomalos)
    
    # Ajustar el tiempo a la longitud de los datos
    t_normales = np.arange(len(ifft_normales)) * dt_normales if fs_normales else np.array([])
    t_anomalos = np.arange(len(ifft_anomalos)) * dt_anomalos if fs_anomalos else np.array([])

    # Graficar los resultados en ventanas separadas
    
    
    # # Graficar datos normales después de la IFFT
    # plt.figure(figsize=(15, 5))
    # if len(ifft_normales) > 0:
    #     plt.plot(t_normales[:len(ifft_normales)], ifft_normales[:len(ifft_normales)], color='blue')
    # plt.title('Datos Normales después de la IFFT')  # Título de la gráfica
    # plt.xlabel('Tiempo (s)')  # Etiqueta del eje x
    # plt.ylabel('Amplitud')  # Etiqueta del eje y
    # plt.grid(True)  # Mostrar cuadrícula
    # plt.show()  # Mostrar la gráfica
        
    # # Graficar datos anómalos después de la IFFT
    # plt.figure(figsize=(15, 5))
    # if len(ifft_anomalos) > 0:
    #     plt.plot(t_anomalos[:len(ifft_anomalos)], ifft_anomalos[:len(ifft_anomalos)], color='red')
    # plt.title('Datos Anómalos después de la IFFT')  # Título de la gráfica
    # plt.xlabel('Tiempo (s)')  # Etiqueta del eje x
    # plt.xlim(0,0.2)
    # plt.ylabel('Amplitud')  # Etiqueta del eje y
    # plt.grid(True)  # Mostrar cuadrícula
    # plt.show()  # Mostrar la gráfica
    

    # Crear una figura y un conjunto de ejes
    plt.figure(figsize=(15, 5))
    
    # Graficar datos normales después de la IFFT
    if len(ifft_normales) > 0:
        plt.plot(t_normales[:len(ifft_normales)], ifft_normales[:len(ifft_normales)], color='blue', label='Datos Normales')
    
    # Graficar datos anómalos después de la IFFT
    if len(ifft_anomalos) > 0:
        plt.plot(t_anomalos[:len(ifft_anomalos)], ifft_anomalos[:len(ifft_anomalos)], color='red', label='Datos Anómalos')
    
    
    plt.plot(tiempo, traza, label='Traza', color='green')  # Graficar la traza en azul
    
    # Configuración de la gráfica
    plt.title('Datos después de la IFFT')  # Título de la gráfica
    plt.xlabel('Tiempo (s)')  # Etiqueta del eje x
    plt.ylabel('Amplitud')  # Etiqueta del eje y
    plt.grid(True)  # Mostrar cuadrícula
    plt.legend()  # Mostrar leyenda para diferenciar las líneas
    plt.xlim(0, 0.2)  # Ajustar el límite del eje x
    
    # Mostrar la gráfica
    plt.show()

    
    # Graficar FFT de los datos normales, anómalos y originales sobrepuestos
    graficar_fft_superpuesta(datos_normales, datos_anomalos, fs_original, traza)

# Seleccionar una traza para procesar
while True:
    try:
        indice_traza = int(input(f"Introduce el número de traza (1-{datos_procesados.shape[1]}): "))  # Solicitar al usuario el número de traza
        if 1 <= indice_traza <= datos_procesados.shape[1]:  # Verificar que el número esté en el rango válido
            traza_seleccionada = datos_procesados[:, indice_traza - 1]  # Seleccionar la traza correspondiente
            break
    except ValueError:
        print(f"Por favor, ingrese un número válido entre 1 y {datos_procesados.shape[1]}.")  # Mensaje de error en caso de entrada inválida

# Parámetros para el aislamiento
tamano_ventana = 30  # Tamaño de la ventana para Isolation Forest
contaminacion = 0.1  # Proporción de datos que se considera anómala

# Aplicar el Isolation Forest y graficar la traza
etiquetas = aplicar_isolation_forest(traza_seleccionada, tamano_ventana, contaminacion)  # Aplicar el modelo de Isolation Forest
titulo_grafica = f'Traza {indice_traza} - Detección de Anomalías con Isolation Forest (Contaminación={contaminacion})'  # Crear título de la gráfica
graficar_serie_temporal(traza_seleccionada, etiquetas, titulo_grafica)  # Graficar la serie temporal con anomalías destacadas

# Graficar FFT e IFFT
graficar_fft_ifft(traza_seleccionada, etiquetas, frecuencia_muestreo)  # Graficar los resultados de FFT e IFFT
