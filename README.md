# Diffusion Models – Proyecto AAIII

Este repositorio contiene el código desarrollado para el proyecto **Sistema Generativo basado en Modelos de Difusión**, realizado para la asignatura *Aprendizaje Automático III* del Grado en Ciencia e Ingeniería de Datos de la Universidad Autónoma de Madrid.

El objetivo del proyecto es implementar y comparar modelos generativos basados en difusión continua para generación de imágenes, incluyendo modelos **Variance Exploding (VE)** basados en movimiento browniano y modelos **Variance Preserving (VP)** basados en procesos de Ornstein--Uhlenbeck.

## Contenido del proyecto

El sistema permite:

- Entrenar modelos de score mediante *denoising score matching*.
- Generar imágenes con distintos métodos de muestreo:
  - Euler--Maruyama.
  - Predictor--Corrector.
  - Probability Flow ODE.
- Comparar modelos VE y VP.
- Evaluar resultados mediante FID, IS y BPD.
- Probar distintos *noise schedules*: `linear`, `cosine` y `sigmoid`.
- Realizar generación condicional por clase y color.
- Realizar imputación de regiones ocultas en imágenes.
- Explorar la generación de imágenes RGB naturales con AFHQ.

## Estructura del repositorio

```text
.
├── src/                 # Código fuente principal
├── train/               # Notebooks o scripts de entrenamiento
├── samplers/            # Notebooks o scripts de generación
├── measures/            # Evaluación cuantitativa: FID, IS y BPD
├── checkpoints/         # Modelos entrenados
├── samples/             # Imágenes generadas guardadas en formato .pt
├── data/                # Datasets utilizados
├── figures/             # Figuras empleadas en la memoria
└── README.md
```

## Modelos implementados

### Modelo VE

El modelo VE se basa en un proceso de movimiento browniano con coeficiente de difusión dependiente del tiempo. En este caso, la media permanece centrada en la muestra original mientras la varianza aumenta progresivamente.

### Modelo VP

El modelo VP se basa en un proceso de Ornstein--Uhlenbeck con coeficientes dependientes del tiempo. La presencia de un término de deriva permite mantener la varianza acotada, favoreciendo una dinámica más estable.

## Datasets utilizados

### MNIST

Se utiliza principalmente MNIST restringido al dígito 3 como entorno experimental controlado para comparar modelos, samplers y schedules.

### CMNIST

Se construye una versión coloreada de MNIST para estudiar generación condicional por clase y color.

### AFHQ

Se realiza una exploración preliminar sobre imágenes RGB de gatos del dataset AFHQ, redimensionadas de 512 x 512 a 64 x 64 píxeles. Este experimento se interpreta como prueba de escalabilidad del pipeline, no como resultado competitivo.

## Instalación

Se recomienda crear un entorno virtual antes de instalar las dependencias.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

En Google Colab, basta con instalar las dependencias necesarias y montar Google Drive si los datasets, checkpoints o muestras están almacenados allí.

## Uso general

El flujo de trabajo seguido en el proyecto es:

1. Preparar el dataset y aplicar las transformaciones correspondientes.
2. Definir el proceso de difusión VE o VP.
3. Entrenar la red de score.
4. Guardar checkpoints durante el entrenamiento.
5. Generar imágenes con los distintos samplers.
6. Calcular métricas FID, IS y BPD.
7. Comparar configuraciones y seleccionar el mejor modelo.

## Evaluación

Las métricas utilizadas son:

- **FID**: compara distribuciones de características entre imágenes reales y generadas.
- **IS**: evalúa calidad y diversidad a partir de un clasificador preentrenado.
- **BPD**: estima la calidad probabilística del modelo mediante la log-verosimilitud calculada con la probability flow ODE.

Para el cálculo de BPD se aplica dequantización uniforme y una corrección de cambio de variable para obtener valores comparables con la definición estándar en bits por dimensión.

## Resultados principales

En los experimentos sobre MNIST, el modelo **VP con schedule cosine y sampler predictor--corrector** obtuvo el mejor comportamiento global, tanto en términos de FID como de calidad visual.

El modelo VE permitió validar correctamente el pipeline, pero mostró un comportamiento menos estable y valores de FID superiores. En AFHQ, el sistema logró generar estructuras compatibles con imágenes de gatos, aunque con limitaciones asociadas a la resolución, la complejidad del dataset y la capacidad de la arquitectura.

## Reproducibilidad

Para mejorar la reproducibilidad:

- Se fijan semillas aleatorias en los experimentos principales.
- Los checkpoints se almacenan por época.
- Las muestras generadas se guardan en formato `.pt`.
- Las métricas se calculan posteriormente a partir de las mismas muestras generadas.

## Autoras

Proyecto realizado por:

- Eva Blázquez
- Gabriela Damas

Asignatura: Aprendizaje Automático III  
Universidad Autónoma de Madrid  
Curso 2025/2026
