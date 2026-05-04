# Proyecto 1: Sistema Híbrido Bioinspirado (REINFORCE + STDP)

Este repositorio contiene la implementación de un sistema de aprendizaje automático híbrido desarrollado para la asignatura **APAU BIO** (Aprendizaje Automático Bioinspirado).

El sistema acopla el algoritmo de Reinforcement Learning clásico **REINFORCE** con una regla local de aprendizaje neuroinspirada (**STDP / Hebbiana de 3 factores**).

## Estructura del Repositorio

*   `src/`: Contiene el código fuente en Python.
    *   `agent.py`: Implementación de las clases `ReinforceAgent` (baseline) e `HybridReinforceAgent` (propuesta).
    *   `train.py`: Script principal para entrenar a ambos agentes durante 1000 episodios y guardar los resultados.
    *   `evaluate.py`: Script para cargar los resultados del entrenamiento y generar la gráfica comparativa.
    *   `test_hybrid.py`: (Opcional) Script de pruebas utilizado durante el desarrollo inicial.
*   `results/`: Contiene los registros de los experimentos.
    *   `training_rewards.csv`: Datos crudos del proceso de entrenamiento.
    *   `learning_curves.png`: Gráfica comparativa que muestra el estudio de ablación.
*   `informe/`: Contiene el código fuente en LaTeX (`informe.tex`) con la justificación teórica y la estructura del informe.
*   `Propuesta_Proyecto.md`: El diseño original aceptado para este proyecto.

## Requisitos

Para ejecutar el código de este proyecto, asegúrate de tener Python 3.8+ instalado y ejecuta el siguiente comando para instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Reproducción de Resultados (Scripts de ejecución)

Para reproducir el estudio de ablación (entrenamiento de ambos agentes) y la posterior visualización de los resultados:

1.  **Entrenar los agentes:**
    ```bash
    python src/train.py
    ```
    *Esto puede tardar un par de minutos, dependiendo de tu ordenador. Generará el archivo `training_rewards.csv` en la carpeta `results/`.*

2.  **Generar Gráficos:**
    ```bash
    python src/evaluate.py
    ```
    *Este script leerá el archivo CSV y guardará `learning_curves.png` en la carpeta `results/`.*

## Estudio de Ablación (Validez)

Este repositorio cumple con las tres reglas de validez requeridas en la asignatura:
1.  **Prueba de ablación pura:** Si ejecutamos el modelo base (`ReinforceAgent`), el sistema vuelve a utilizar Backpropagation clásico de PyTorch en lugar de la traza STDP.
2.  **Dinámica temporal:** Las trazas STDP evolucionan con decaimiento en cada paso de simulación.
3.  **Flujo de datos bidireccional:** REINFORCE calcula el error de TD o Retorno Global ($G_t$) a nivel episódico y la envía al módulo bioinspirado, que modula localmente las trazas para la actualización final de los pesos.

## Video

