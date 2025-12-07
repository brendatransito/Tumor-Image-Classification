
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
| ├── 04_ResNet50_noise.ipynb
| ├── 05_Evaluar_ruido_GradCAM.ipynb
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

### 2. 01_data_exploration.ipynb

Notebook de exploración inicial:

- visualización de imágenes,
-   revisión de clases,
- análisis básico de distribución y características del dataset.

### 3. 02_cnn_baseline.ipynb

Entrenamiento de ResNet-50, con dos configuraciones:
- Con augmentación: rotaciones, flips, jitter, etc.
- Sin augmentación: baseline puro
Incluye:
- Entrenamiento
- Curvas de pérdida y exactitud
- Reportes de clasificación
- Matrices de confusión
### 4. 02_1_cnn_baseline.ipynb
Extiende el experimento CNN utilizando ResNet-101, una arquitectura más profunda.
Se comparan nuevamente los escenarios:
- Augmentation vs NoAug
- Métricas finales y desempeño por clase

### 5. 03_vit_experiments.ipynb
Implementación del modelo Transformer para imágenes, ViT-B/16:
- Fine-tuning completo
- Training con AdamW
- Dropout y Stochastic Depth
- Comparación Aug vs NoAug
- Matrices de confusión
- Reportes de clasificación

### 6. 04_ResNet50_noise.ipynb
Este notebook introduce ruido gaussiano en el entrenamiento:
- Entrenamiento con augment + ruido
- Entrenamiento solo ruido
- Comparación con modelos base
- Métricas en test limpio
Es fundamental para evaluar robustez.

### 7. 05_Evaluar_ruido_GradCam.ipynb
Evalúa modelos bajo distintos niveles de ruido gaussiano:
- std = 0.0, 0.02, 0.05, 0.10, 0.20
Modelos evaluados:
- ResNet-50 (normal)
- ResNet-101
- ViT-B/16
- ResNet-50 Noise-Trained
Incluye:
- Gráficas de degradación del Accuracy
- Comparaciones entre modelos
- Grad-CAM aplicado con ruido (0, 0.05, 0.2)
Permite observar:
- Qué modelos mantienen activaciones consistentes bajo condiciones adversas
- Qué arquitectura resulta más robusta
