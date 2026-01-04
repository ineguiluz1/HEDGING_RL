#!/bin/bash
# Script para visualizar los estudios de Optuna con TensorBoard
# 
# Uso:
#   ./view_tensorboard.sh         # Ver ambos estudios (TD3 y SAC)
#   ./view_tensorboard.sh td3     # Ver solo TD3
#   ./view_tensorboard.sh sac     # Ver solo SAC

set -e

cd "$(dirname "$0")"

if [ ! -d "optuna_studies/tensorboard" ]; then
    echo "❌ No se encontraron logs de TensorBoard en optuna_studies/tensorboard/"
    echo "   Ejecuta primero optuna_td3.py y/o optuna_sac.py"
    exit 1
fi

case "${1:-both}" in
    td3)
        echo "🔍 Visualizando estudios TD3..."
        echo "📊 TensorBoard disponible en: http://localhost:6006"
        tensorboard --logdir=optuna_studies/tensorboard/td3 --port=6006
        ;;
    sac)
        echo "🔍 Visualizando estudios SAC..."
        echo "📊 TensorBoard disponible en: http://localhost:6006"
        tensorboard --logdir=optuna_studies/tensorboard/sac --port=6006
        ;;
    both|all)
        echo "🔍 Visualizando ambos estudios (TD3 y SAC)..."
        echo "📊 TensorBoard disponible en: http://localhost:6006"
        echo ""
        echo "En TensorBoard podrás filtrar por:"
        echo "  - td3/trial_* para ver trials TD3"
        echo "  - sac/trial_* para ver trials SAC"
        tensorboard --logdir=optuna_studies/tensorboard --port=6006
        ;;
    *)
        echo "Uso: $0 [td3|sac|both]"
        exit 1
        ;;
esac
