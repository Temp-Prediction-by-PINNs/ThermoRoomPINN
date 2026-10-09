#!/bin/bash
$CLEAN_FILES=$(
rm comparison_error_maps.png
rm comparison_error_over_time.png
rm comparison_metrics.txt
rm loss_convergence.png
rm loss_history.csv
rm pinn_heat2d.pt
rm pinn_predictions.npz
rm logs/*
)

echo "cleaning" $CLEAN_FILES

JOB_TRAIN=$(sbatch --parsable run_train.sh)
echo "Training : $JOB_TRAIN"

JOB_PRED=$(sbatch --parsable \
    --job-name=ThermoPred \
    --dependency=afterok:$JOB_TRAIN \
    run.sh export_pinn_predictions.py)
echo "Prediction : $JOB_PRED"

JOB_COMP=$(sbatch --parsable \
    --job-name=ThermoComp \
    --dependency=afterok:$JOB_PRED \
    run.sh compare_pinn_reference.py)
echo "comparaison : $JOB_COMP"

JOB_LOSS=$(sbatch --parsable \
    --job-name=ThermoLoss \
    --dependency=afterok:$JOB_COMP \
    run.sh plot_loss_history.py)
echo "Affichage Loss : $JOB_LOSS"
