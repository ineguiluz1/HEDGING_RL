# TensorBoard Integration - Optuna Hyperparameter Optimization

## Overview

Los scripts de optimización `optuna_td3.py` y `optuna_sac.py` ahora incluyen:
- **Mismo flujo de entrenamiento que `run_training.py`**: Incluyendo early stopping, curriculum learning, etc.
- **Environments compartidos**: Se crean una sola vez y se reusan para todos los trials (eficiencia y consistencia)
- **Logging completo con TensorBoard**: Para visualizar y analizar los entrenamientos

## Características Principales

### 1. Mismo Entrenamiento que `run_training.py`

Cada trial usa exactamente la misma configuración de entrenamiento:
- ✅ **Early Stopping**: Detiene fases si no hay mejora tras N episodios
- ✅ **Curriculum Learning**: Neutral → Mixed o Low Vol → High Vol (según config)
- ✅ **Reproducibilidad**: Seeds controladas para cada trial
- ✅ **Flujo completo**: Usa `train_multi_env()` internamente

### 2. Environments Compartidos (Eficiencia)

Los environments se crean **una sola vez al inicio**:
- Todos los trials usan los **mismos** trajectories de entrenamiento
- Los **mismos** episodios de test
- Garantiza comparación justa entre hiperparámetros
- Ahorra tiempo (no se regeneran datos cada trial)

## Estructura de Logs

```
optuna_studies/
├── tensorboard/
│   ├── td3/
│   │   ├── trial_0/           # Baseline con parámetros actuales
│   │   ├── trial_1/           # Primera optimización Optuna
│   │   ├── trial_2/
│   │   └── ...
│   └── sac/
│       ├── trial_0/           # Baseline con parámetros actuales
│       ├── trial_1/
│       └── ...
├── td3_optimization.db        # Base de datos Optuna TD3
├── td3_best_params.json       # Mejores hiperparámetros TD3
├── sac_optimization.db        # Base de datos Optuna SAC
└── sac_best_params.json       # Mejores hiperparámetros SAC
```

## Métricas Registradas

### 1. Training Metrics (durante entrenamiento)
- `training/episode_reward`: Reward por episodio
- `training/cumulative_reward`: Reward acumulado

### 2. Network Weights & Gradients (cada 200 updates)
- `td3_weights/actor/*` o `sac_weights/actor/*`: Histogramas de pesos del actor
- `td3_weights/critic/*` o `sac_weights/critic/*`: Histogramas de pesos del critic
- `td3_gradients/actor/*` o `sac_gradients/actor/*`: Histogramas de gradientes del actor
- `td3_gradients/critic/*` o `sac_gradients/critic/*`: Histogramas de gradientes del critic

**Útil para detectar**:
- **Vanishing gradients**: Gradientes muy pequeños (cerca de 0)
- **Exploding gradients**: Gradients muy grandes (>1)
- **Dead neurons**: Pesos que no cambian
- **Weight drift**: Pesos que crecen sin control

### 3. Evaluation Metrics (al final del trial)
- `evaluation/total_pnl`: P&L total en test set
- `evaluation/sharpe_ratio`: Sharpe ratio
- `evaluation/pnl_improvement`: Mejora de P&L vs benchmark
- `evaluation/sharpe_improvement`: Mejora de Sharpe vs benchmark
- `evaluation/max_drawdown`: Máximo drawdown
- `evaluation/pnl_variance`: Varianza del P&L

### 4. Objective
- `objective/value`: Valor de la función objetivo (sharpe_improvement + 0.01 * pnl_improvement)

### 5. Hyperparameters (HParams Plugin)
- Todos los hiperparámetros numéricos vinculados con métricas finales
- Permite análisis de correlación automático en la tab **HPARAMS**
- Métricas vinculadas:
  - `hparam/sharpe_improvement`
  - `hparam/pnl_improvement`
  - `hparam/total_pnl`
  - `hparam/sharpe_ratio`
  - `hparam/max_drawdown`

## Visualización

### Opción 1: Script de conveniencia

```bash
# Ver ambos estudios (TD3 y SAC)
./view_tensorboard.sh

# Ver solo TD3
./view_tensorboard.sh td3

# Ver solo SAC
./view_tensorboard.sh sac
```

### Opción 2: Comando directo

```bash
# Ver ambos estudios
tensorboard --logdir=optuna_studies/tensorboard --port=6006

# Ver solo TD3
tensorboard --logdir=optuna_studies/tensorboard/td3 --port=6006

# Ver solo SAC
tensorboard --logdir=optuna_studies/tensorboard/sac --port=6006
```

Luego abre tu navegador en: http://localhost:6006

## Análisis en TensorBoard

### 1. Comparar trials

En la interfaz de TensorBoard:
- **SCALARS tab**: Compara métricas entre diferentes trials
- Usa regex para filtrar: `trial_[0-9]+` para ver todos
- Smooth curves para ver tendencias
- Select specific runs para comparar trials específicos

### 2. Analizar hiperparámetros (HParams Plugin)

La tab **HPARAMS** proporciona análisis automático:
- **Parallel Coordinates**: Visualiza relación entre hiperparámetros y métricas
- **Scatter Plot Matrix**: Correlaciones entre todas las variables
- **Table View**: Ordena trials por métrica de interés

**Ejemplo de uso**:
1. Ve a la tab **HPARAMS**
2. Ordena por `hparam/sharpe_improvement` (click en columna)
3. Identifica rangos óptimos de hiperparámetros
4. Usa parallel coordinates para ver patrones

### 3. Identificar mejores configuraciones

1. Ve a la tab **SCALARS**
2. Ordena por `evaluation/sharpe_improvement` o `objective/value`
3. Identifica los trials con mejor performance
4. Revisa sus hiperparámetros en **HPARAMS**

### 4. Analizar convergencia

- **Training rewards**: Verifica que los episodios convergen
- **Cumulative reward**: Debe incrementar consistentemente
- **Evaluation metrics**: Compara final performance entre trials

### 5. Detectar overfitting/underfitting

Compara:
- `training/cumulative_reward` (alto) vs `evaluation/sharpe_improvement` (bajo) → Overfitting
- Ambos bajos → Underfitting o mala configuración

### 6. Debug de redes neuronales (DISTRIBUTIONS/HISTOGRAMS tab)

**Detectar vanishing gradients**:
- Ve a **HISTOGRAMS** → `td3_gradients` o `sac_gradients`
- Si los gradientes son extremadamente pequeños (<1e-6), hay vanishing gradients
- Solución: Aumentar learning rate, cambiar inicialización, usar batch normalization

**Detectar exploding gradients**:
- Gradientes muy grandes (>10)
- Solución: Reducir learning rate, usar gradient clipping

**Analizar distribución de pesos**:
- **HISTOGRAMS** → `td3_weights` o `sac_weights`
- Los pesos deben tener distribución razonable (no todos cerca de 0 o muy grandes)
- Cambios en la distribución indican que la red está aprendiendo

**Dead neurons**:
- Pesos que no cambian entre updates
- Indica que algunas neuronas no están contribuyendo

## Workflow Recomendado

### 1. Ejecutar optimización

```bash
cd src

# Optimizar TD3 (100 trials)
# Los environments se crean una vez al inicio
python optuna_td3.py

# En paralelo o después, optimizar SAC
python optuna_sac.py
```

**Nota importante**: 
- Los environments se crean al inicio y se reusan para todos los trials
- Cada trial entrena con el mismo flujo que `run_training.py` (early stopping incluido)
- Los hiperparámetros cambian entre trials, pero los datos son consistentes

### 2. Monitorear en tiempo real

Durante la optimización, en otra terminal:

```bash
./view_tensorboard.sh
```

Esto te permite ver el progreso de los trials mientras se ejecutan.

### 3. Análisis post-optimización

Una vez completados los 100 trials:

1. **TensorBoard**: Analiza visualmente los trials
2. **Optuna database**: Usa optuna-dashboard para análisis avanzado
3. **Best params JSON**: Revisa los mejores hiperparámetros encontrados

```bash
# Ver mejores parámetros TD3
cat optuna_studies/td3_best_params.json

# Ver mejores parámetros SAC
cat optuna_studies/sac_best_params.json
```

## Visualizaciones Útiles

### Comparación de algoritmos (TD3 vs SAC)

Abre ambos estudios simultáneamente:
```bash
tensorboard --logdir=optuna_studies/tensorboard
```

Luego en TensorBoard:
- Filtra por prefijo: `td3/` vs `sac/`
- Compara las distribuciones de `evaluation/sharpe_improvement`
- Identifica qué algoritmo converge mejor

### Análisis de sensibilidad

Para un hiperparámetro específico:
1. Exporta los datos de TensorBoard (Download CSV)
2. Correlaciona el valor del hiperparámetro con el objective value
3. Identifica rangos óptimos

### Timeline de optimización

En **SCALARS** tab:
- X-axis: Relative (muestra tiempo de ejecución)
- Observa si los primeros trials son suficientes o si Optuna mejora con más trials

## Tips

1. **Trial 0 es el baseline**: Siempre usa los parámetros actuales de CONFIG, úsalo como referencia

2. **Compara vs baseline**: El objetivo es mejorar sobre trial_0

3. **No todos los trials son exitosos**: Algunos pueden fallar o tener métricas extremas, esto es normal en optimización

4. **Usa smoothing**: En TensorBoard, ajusta el slider "Smoothing" para ver tendencias más claras

5. **Multi-run comparison**: Selecciona múltiples runs (Ctrl+click) para comparar específicos trials

6. **HParams tab es poderoso**: Usa parallel coordinates para identificar patrones entre hiperparámetros y performance

7. **Histogramas de pesos**: Revisa cada 200 updates. Si ves patrones extraños (todos 0, muy grandes, no cambian), hay un problema

8. **Gradientes saludables**: Deben estar en rango [1e-4, 1e-1]. Fuera de este rango indica problemas

9. **Compara distribuciones**: En HISTOGRAMS, compara trials exitosos vs fallidos para ver diferencias en pesos/gradientes

10. **Download data**: TensorBoard permite exportar datos (Download CSV) para análisis offline

## Troubleshooting

### TensorBoard no muestra datos

```bash
# Verifica que existan logs
ls -la optuna_studies/tensorboard/td3/
ls -la optuna_studies/tensorboard/sac/

# Reinicia TensorBoard
pkill -f tensorboard
./view_tensorboard.sh
```

### Puerto 6006 en uso

```bash
# Usa otro puerto
tensorboard --logdir=optuna_studies/tensorboard --port=6007
```

### Logs muy grandes

```bash
# Limpia trials antiguos (cuidado: esto borra logs)
rm -rf optuna_studies/tensorboard/td3/trial_*
rm -rf optuna_studies/tensorboard/sac/trial_*

# O limpia trials específicos
rm -rf optuna_studies/tensorboard/td3/trial_{1..10}
```

## Integración con Optuna Dashboard

TensorBoard complementa (no reemplaza) Optuna Dashboard:

```bash
# Instalar optuna-dashboard
pip install optuna-dashboard

# Ver estudios interactivamente
optuna-dashboard optuna_studies/td3_optimization.db optuna_studies/sac_optimization.db
```

- **TensorBoard**: Mejor para ver curvas de entrenamiento y métricas temporales
- **Optuna Dashboard**: Mejor para análisis de hiperparámetros, importancia, y parallel coordinates

## Referencias

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [PyTorch SummaryWriter](https://pytorch.org/docs/stable/tensorboard.html)
