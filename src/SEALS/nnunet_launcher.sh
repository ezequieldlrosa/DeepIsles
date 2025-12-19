#!/bin/bash

# Accept raw_data_dir as the first argument to this script
RAW_DATA_DIR=$1
BASE_RESULTS_PATH=$2
RESULTS_FOLDER_PATH="${BASE_RESULTS_PATH}/SEALS/nnUNet_trained_models"
chmod -R u+rwX "$RESULTS_FOLDER_PATH" 2>/dev/null || true

# Suppressing output of environment variable exports
export nnUNet_raw="data/nnUNet_raw_data_base/nnUNet_raw_data" >/dev/null 2>&1
export nnUNet_preprocessed="data/nnUNet_preprocessed" >/dev/null 2>&1
export nnUNet_results="$RESULTS_FOLDER_PATH" >/dev/null 2>&1

python dataset_conversion/Task500_Ischemic_Stroke_Test.py --raw_data_dir "$RAW_DATA_DIR" >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
nnUNetv2_predict_from_modelfolder \
               -i $nnUNet_raw/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold0 \
               -m $nnUNet_results/nnUNet/3d_fullres/Task500_Ischemic_Stroke_Test/nnUNetTrainer__nnUNetPlans__3d_fullres \
               -f 0 \
               --save_probabilities \
               --disable_tta \
               >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
nnUNetv2_predict_from_modelfolder \
               -i $nnUNet_raw/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold1 \
               -m $nnUNet_results/nnUNet/3d_fullres/Task500_Ischemic_Stroke_Test/nnUNetTrainer__nnUNetPlans__3d_fullres \
               -f 1 \
               --save_probabilities \
               --disable_tta \
               >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
nnUNetv2_predict_from_modelfolder \
               -i $nnUNet_raw/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold2 \
               -m $nnUNet_results/nnUNet/3d_fullres/Task500_Ischemic_Stroke_Test/nnUNetTrainer__nnUNetPlans__3d_fullres \
               -f 2 \
               --save_probabilities \
               --disable_tta \
               >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
nnUNetv2_predict_from_modelfolder \
               -i $nnUNet_raw/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold3 \
               -m $nnUNet_results/nnUNet/3d_fullres/Task500_Ischemic_Stroke_Test/nnUNetTrainer__nnUNetPlans__3d_fullres \
               -f 3 \
               --save_probabilities \
               --disable_tta \
               >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
nnUNetv2_predict_from_modelfolder \
               -i $nnUNet_raw/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold4 \
               -m $nnUNet_results/nnUNet/3d_fullres/Task500_Ischemic_Stroke_Test/nnUNetTrainer__nnUNetPlans__3d_fullres \
               -f 4 \
               --save_probabilities \
               --disable_tta \
               >/dev/null 2>&1

# Suppressing the python scripts for postprocessing
python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold0 \
                        -m preliminary_phase \
                        -f fold0 \
                        --raw_data_dir $RAW_DATA_DIR \
                        >/dev/null 2>&1

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold1 \
                        -m preliminary_phase \
                        -f fold1 \
                        --raw_data_dir $RAW_DATA_DIR \
                        >/dev/null 2>&1

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold2 \
                        -m preliminary_phase \
                        -f fold2 \
                        --raw_data_dir $RAW_DATA_DIR \
                        >/dev/null 2>&1

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold3 \
                        -m preliminary_phase \
                        -f fold3 \
                        --raw_data_dir $RAW_DATA_DIR \
                        >/dev/null 2>&1

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold4 \
                        -m preliminary_phase \
                        -f fold4 \
                        --raw_data_dir $RAW_DATA_DIR \
                        >/dev/null 2>&1
# Suppressing ensemble softmax
model_0=test_result_recover/preliminary_phase/fold0
model_1=test_result_recover/preliminary_phase/fold1
model_2=test_result_recover/preliminary_phase/fold2
model_3=test_result_recover/preliminary_phase/fold3
model_4=test_result_recover/preliminary_phase/fold4

nnUNetv2_ensemble \
       -f $model_0 \
          $model_1 \
          $model_2 \
          $model_3 \
          $model_4 \
       -o test_ensemble/ \
       >/dev/null 2>&1

# Suppressing thresholding final output and passing the raw_data_dir
python threshold_redirect.py \
                            -i test_ensemble/ \
                            -o ${RAW_DATA_DIR}/output/seals \
                            --raw_data_dir $RAW_DATA_DIR \
                            >/dev/null 2>&1
