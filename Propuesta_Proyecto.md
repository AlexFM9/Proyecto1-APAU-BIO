# Propuesta de Proyecto 1: Sistema Híbrido Bioinspirado con Aprendizaje por Refuerzo

## 1. Título Provisional
**Control de Políticas mediante REINFORCE Modulado por Trazas de Elegibilidad STDP (Regla de Tres Factores)**

## 2. Componentes del Sistema Híbrido
* **Algoritmo de Aprendizaje por Refuerzo (RL):** REINFORCE (Policy Gradient).
* **Componente Bioinspirado:** Trazas de elegibilidad locales basadas en STDP (Plasticidad Dependiente del Tiempo de la Espiga) o regla Hebbiana de tres factores.
* **Entorno (Environment):** `CartPole-v1` (con posible extensión a `LunarLander-v2` si se domina rápido). CartPole es ideal para probar REINFORCE y permite diagnosticar muy bien si las actualizaciones locales de los pesos están funcionando correctamente sin excesivo coste computacional.

## 3. Motivación (El "Por qué" tiene sentido)
El algoritmo REINFORCE estándar actualiza los pesos de la red de política calculando el gradiente de todo el sistema mediante *Backpropagation*, asignando el crédito a cada peso de forma puramente matemática. Sin embargo, en los cerebros biológicos, la asignación de crédito se resuelve de forma local: las sinapsis cambian basándose en la actividad pre- y post-sináptica a lo largo del tiempo (STDP) modulada por una señal global de recompensa (como la dopamina). 

Esta propuesta plantea sustituir el cálculo del gradiente de la capa oculta por una **regla de aprendizaje de tres factores**. STDP calcula la "traza de elegibilidad" local (qué neuronas se activaron juntas y en qué orden temporal), y el retorno $G_t$ de REINFORCE actúa como la señal de dopamina global que modula el cambio definitivo en los pesos.

## 4. Arquitectura y Flujo de Datos (Punto de inserción: Optimización / Pesos)
El punto de inserción de nuestro componente bioinspirado es el **mecanismo de actualización de pesos** de la red neuronal.

1. **Interacción:** El agente interactúa con el entorno y recoge la trayectoria temporal $\{s_t, a_t, r_{t+1}, s_{t+1}\}$.
2. **Dinámica Local (STDP):** Durante el paso "forward" en cada instante de tiempo $t$, cada sinapsis mantiene y actualiza una traza de elegibilidad $e_{i,j}(t)$ basada en la co-activación de la neurona pre-sináptica y la acción tomada (post-sináptica).
3. **Señal Global (REINFORCE):** Al final de la trayectoria (o en cada paso de actualización), el agente RL calcula el retorno descontado $G_t$.
4. **Actualización (Regla de tres factores):** El peso sináptico se actualiza multiplicando la traza local acumulada por la recompensa global proporcionada por RL: 
   $$ \Delta W_{i,j} = \alpha \cdot G_t \cdot e_{i,j} $$

## 5. Prueba de Validez (Checklist)
* **¿Ablación clara?** *Sí.* Si eliminamos el cálculo de la traza de elegibilidad local (STDP) y restauramos la actualización estándar con *Backpropagation* de PyTorch, el sistema vuelve a ser la línea base del REINFORCE clásico (se supera la prueba 1).
* **¿Dinámica adaptativa?** *Sí.* Las trazas de elegibilidad evolucionan de forma dinámica en cada instante de tiempo basándose en la actividad neuronal, no son un preprocesado fijo (se supera la prueba 2).
* **¿Flujo de datos claro?** *Sí.* El componente de RL (REINFORCE) envía el error global ($G_t$) al módulo bioinspirado (STDP), que usa esa señal para escalar sus trazas locales y modificar los parámetros de la política (se supera la prueba 3).

## 6. Hiperparámetros a Barrer
Para analizar la robustez del sistema, se experimentará con los siguientes hiperparámetros:
* **Tasa de decaimiento de la traza STDP ($\tau$):** Cuánto tiempo "recuerda" una sinapsis que fue activada antes de desvanecerse.
* **Tasa de aprendizaje base ($\alpha$):** Paso de actualización para los pesos.
* **Factor de descuento de RL ($\gamma$):** Horizonte temporal para el cálculo del retorno.

## 7. Plan de Evaluación
Se comparará la eficiencia muestral y la estabilidad del entrenamiento (recompensa media acumulada por episodio) de:
1. Modelo Base: REINFORCE clásico (con Backpropagation).
2. Modelo Híbrido: REINFORCE modificado con regla Hebbiana / STDP de 3 factores.
