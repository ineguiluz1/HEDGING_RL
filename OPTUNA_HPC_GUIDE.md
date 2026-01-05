# Hyperparameter Tuning con Optuna en HPC

Este documento explica cómo ejecutar la optimización de hiperparámetros en paralelo usando Optuna en un cluster HPC con SLURM.

## 📋 Descripción General

Los scripts de Optuna (`optuna_sac.py` y `optuna_td3.py`) están configurados para usar SQLite como storage compartido, lo que permite la ejecución paralela de múltiples workers que comparten el mismo estudio.

## 🚀 Opciones de Ejecución

### Opción 1: Un solo worker (ejecución secuencial)

Ejecuta un único job que procesará todos los trials de forma secuencial:

```bash
# Para SAC
sbatch run_optuna_sac.sh

# Para TD3
sbatch run_optuna_td3.sh
```

**Ventajas:**
- Simple y directo
- No requiere coordinación entre workers

**Desventajas:**
- Más lento (trials secuenciales)
- No aprovecha la capacidad de paralelización del HPC

### Opción 2: Múltiples workers en paralelo (RECOMENDADO)

Lanza varios jobs simultáneamente que compartirán la misma base de datos y ejecutarán trials en paralelo:

```bash
# Para SAC - lanza 4 workers en paralelo
./launch_parallel_optuna_sac.sh 4

# Para TD3 - lanza 4 workers en paralelo
./launch_parallel_optuna_td3.sh 4

# Puedes ajustar el número de workers según disponibilidad
./launch_parallel_optuna_sac.sh 8  # 8 workers
```

**Ventajas:**
- Mucho más rápido (trials en paralelo)
- Aprovecha eficientemente los recursos del HPC
- Cada worker ejecuta trials independientes
- Optuna gestiona automáticamente la coordinación

**Desventajas:**
- Requiere más recursos simultáneos (más GPUs)

## 🔧 Cómo Funciona la Paralelización

1. **Base de datos compartida**: Todos los workers leen y escriben en la misma SQLite database:
   - SAC: `optuna_studies/sac_optimization.db`
   - TD3: `optuna_studies/td3_optimization.db`

2. **Coordinación automática**: Optuna maneja automáticamente:
   - Asignación de trials a cada worker
   - Evita trials duplicados
   - Sincronización de resultados
   - Selección de hiperparámetros usando TPE Sampler

3. **Tolerancia a fallos**: Si un worker falla:
   - Los demás continúan funcionando
   - Los trials completados se preservan
   - Puedes relanzar workers adicionales en cualquier momento

## 📊 Monitoreo

### Ver jobs en ejecución
```bash
squeue -u $USER
```

### Ver logs en tiempo real
```bash
# SAC
tail -f logs/optuna_sac_*.out

# TD3
tail -f logs/optuna_td3_*.out
```

### Cancelar todos los workers
```bash
# Cancelar todos los jobs de SAC
scancel -n optuna_sac_hedging

# Cancelar todos los jobs de TD3
scancel -n optuna_td3_hedging
```

## 🎯 Configuración Recomendada

### Para desarrollo/prueba
```bash
# 2-3 workers, suficiente para probar
./launch_parallel_optuna_sac.sh 2
```

### Para optimización completa
```bash
# 4-8 workers, balance entre velocidad y recursos
./launch_parallel_optuna_sac.sh 6
./launch_parallel_optuna_td3.sh 6
```

### Para búsqueda intensiva
```bash
# 10+ workers si el cluster lo permite
./launch_parallel_optuna_sac.sh 10
```

## 💡 Consejos

1. **Número óptimo de workers**: 
   - Depende de GPUs disponibles en el cluster
   - Recomendado: 4-8 workers por algoritmo
   - Más workers = optimización más rápida

2. **Tiempo de ejecución**:
   - Con 1 worker: ~23 horas para 100 trials
   - Con 4 workers: ~6 horas para 100 trials
   - Con 8 workers: ~3 horas para 100 trials

3. **Relanzar optimización**:
   - Puedes añadir más workers en cualquier momento
   - Los trials completados no se repiten
   - Útil si quieres más trials después de ver resultados iniciales

4. **Resultados**:
   - Todos los workers actualizan el mismo estudio
   - Los mejores parámetros se guardan automáticamente
   - TensorBoard logs separados por trial

## 📁 Estructura de Archivos

```
HEDGING_RL/
├── run_optuna_sac.sh              # Script SBATCH individual para SAC
├── run_optuna_td3.sh              # Script SBATCH individual para TD3
├── launch_parallel_optuna_sac.sh  # Launcher paralelo para SAC
├── launch_parallel_optuna_td3.sh  # Launcher paralelo para TD3
├── logs/                          # Logs de ejecución
│   ├── optuna_sac_*.out
│   └── optuna_td3_*.out
└── optuna_studies/                # Resultados de optimización
    ├── sac_optimization.db        # Base de datos compartida SAC
    ├── td3_optimization.db        # Base de datos compartida TD3
    ├── sac_best_params.json       # Mejores parámetros SAC
    ├── td3_best_params.json       # Mejores parámetros TD3
    └── tensorboard/               # Logs de TensorBoard
        ├── sac/
        └── td3/
```

## 🐛 Troubleshooting

### Workers no empiezan
- Verifica disponibilidad de GPUs: `sinfo -p <partition>`
- Revisa logs de error: `logs/optuna_*_*.err`

### SQLite database locked
- Normal con muchos workers simultáneos
- Optuna reintenta automáticamente
- Si persiste, reduce el número de workers

### Resultados inconsistentes
- Verifica que todos los workers usen la misma versión del código
- Asegúrate de que `optuna_studies/` es accesible por todos los workers

## 📧 Notificaciones

Los scripts están configurados para enviar emails cuando:
- El job comienza (BEGIN)
- El job termina (END)
- El job falla (FAIL)

Email configurado: `aitor.diez@opendeusto.es`

Puedes cambiar esto en los archivos `run_optuna_*.sh`:
```bash
#SBATCH --mail-user=tu_email@dominio.com
```
