
# Clasificación de Tumores Cerebrales en MRI con ResNet-50, ResNet-101 y ViT-B/16

Este repositorio contiene el código desarrollado para la clasificación de tumores cerebrales a partir de imágenes de resonancia magnética (MRI) que se encuentran en el repositorio de [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data) mediante modelos de aprendizaje profundo. El proyecto implementa y compara arquitecturas convolucionales (ResNet-50, ResNet-101) y un modelo basado en atención (Vision Transformer ViT-B/16), siguiendo un pipeline reproducible que incluye la preparación del conjunto de datos, el análisis exploratorio y la ejecución de experimentos con CNN y Transformers.

El propósito del repositorio es documentar el flujo completo de trabajo y proporcionar la secuencia adecuada de ejecución de los notebooks, garantizando que los experimentos puedan reproducirse de manera consistente.

---

## Objetivos del proyecto

- Preparar un dataset organizado y estratificado para tareas de clasificación.
- Implementar modelos basados en CNN y Transformers para el análisis de imágenes MRI.
- Proveer notebooks modulares para ejecutar cada etapa del proyecto.
- Mantener un flujo claro y reproducible de experimentación.

---

## Estructura del repositorio

```bash
project/
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_1_cnn_baseline.ipynb
│ ├── 02_cnn_baseline.ipynb
│ ├── 03_vit_experiments.ipynb
| ├── prepare_dataset.py # Script para preparar y particionar el dataset
│
└── README.md
```

---

## Orden de ejecución

Para reproducir el proyecto, los archivos deben ejecutarse en el siguiente orden:

### 1. `prepare_dataset.py`
Script que prepara la estructura del dataset:

- carga de imágenes,
- redimensionado y preprocesamiento básico,
- división estratificada de las clases,
- creación de las carpetas `train/`, `val/` y `test/`.

Ejecutar desde consola:

```bash
python prepare_dataset.py
```

### 2. 2.01_data_exploration.ipynb

Notebook de exploración inicial:

- visualización de imágenes,
-   revisión de clases,
- análisis básico de distribución y características del dataset.

### 3. 02_1_cnn_baseline.ipynb

Primer experimento con CNN:

- implementación del baseline con ResNet-50,

- entrenamiento inicial sin aumentación,

- análisis preliminar del comportamiento del modelo.

### 4. 02_cnn_baseline.ipynb

Extensión del experimento anterior:

- entrenamiento completo de ResNet-50 y ResNet-101,

- configuraciones con y sin aumentación,

- análisis comparativo entre ambas arquitecturas.

5. 03_vit_experiments.ipynb

Experimentos con Vision Transformer (ViT-B/16):

- implementación del modelo,

- entrenamiento bajo diferentes configuraciones,

- comparación contra los modelos CNN.
