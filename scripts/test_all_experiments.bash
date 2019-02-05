#!/bin/bash
# This quickly tests all of the different types of runs--forward, inverse, 
# 2-model, jointly-trained.

NSLOTS=1

#===============================================================================

export OMP_NUM_THREADS=${NSLOTS}
TRAIN_DATA_DIR="elliptic/data/kl_16_32_train"
TEST_DATA_DIR="elliptic/data/kl_16_32_test"
N_TEST=3
NOISE=0.008
SUBSAMPLE=16

cd ..
source .venv/bin/activate

echo "FSS..."
DIR="TEST_RUN/FSS"
mkdir -p ${DIR}
python elliptic_forward_sgplvm_separate.py \
    --tf_num_threads ${NSLOTS} \
    --save_model_file_in "${DIR}/model_in.pkl" \
    --save_model_file_out "${DIR}/model_out.pkl" \
    --n_test ${N_TEST} \
    --save_dir ${DIR} >& ${DIR}/run.out
if [ $? -eq 0 ]; then
    echo "OK!"
else
    echo "FAILED!"
fi

echo "FPS..."
DIR="TEST_RUN/FPS"
mkdir -p ${DIR}
python elliptic_forward_sgplvm_separate.py \
    --tf_num_threads ${NSLOTS} \
    --xi_kern_in PCA \
    --xi_kern_out Linear \
    --save_model_file_in "${DIR}/model_in.pkl" \
    --save_model_file_out "${DIR}/model_out.pkl" \
    --n_test ${N_TEST} \
    --save_dir ${DIR} >& ${DIR}/run.out
if [ $? -eq 0 ]; then
    echo "OK!"
else
    echo "FAILED!"
fi

echo "FSJ..."
DIR="TEST_RUN/FSJ"
mkdir -p ${DIR}
python elliptic_forward_sgplvm_joint.py \
    --joint_model 1 \
    --tf_num_threads ${NSLOTS} \
    --save_model_file "${DIR}/model.pkl" \
    --n_test ${N_TEST} \
    --save_dir ${DIR} >& ${DIR}/run.out
if [ $? -eq 0 ]; then
    echo "OK!"
else
    echo "FAILED!"
fi

echo "ISS..."
DIR="TEST_RUN/ISS"
mkdir -p ${DIR}
python elliptic_inverse_sgplvm_separate.py \
    --tf_num_threads ${NSLOTS} \
    --save_model_file_in "${DIR}/model_in.pkl" \
    --save_model_file_out "${DIR}/model_out.pkl" \
    --n_test ${N_TEST} \
    --save_dir ${DIR} \
    --obs_noise ${NOISE} \
    --obs_subsample ${SUBSAMPLE} \
    >& ${DIR}/run.out
if [ $? -eq 0 ]; then
    echo "OK!"
else
    echo "FAILED!"
fi

echo "ISS..."
DIR="TEST_RUN/ISS"
mkdir -p ${DIR}
python elliptic_inverse_sgplvm_separate.py \
    --tf_num_threads ${NSLOTS} \
    --save_model_file_in "${DIR}/model_in.pkl" \
    --save_model_file_out "${DIR}/model_out.pkl" \
    --n_test ${N_TEST} \
    --save_dir ${DIR} \
    --obs_noise ${NOISE} \
    --obs_subsample ${SUBSAMPLE} \
    >& ${DIR}/run.out
if [ $? -eq 0 ]; then
    echo "OK!"
else
    echo "FAILED!"
fi

echo "IPS..."
DIR="TEST_RUN/IPS"
mkdir -p ${DIR}
python elliptic_inverse_sgplvm_separate.py \
    --tf_num_threads ${NSLOTS} \
    --xi_kern_in PCA \
    --xi_kern_out Linear \
    --save_model_file_in "${DIR}/model_in.pkl" \
    --save_model_file_out "${DIR}/model_out.pkl" \
    --n_test ${N_TEST} \
    --save_dir ${DIR} \
    --obs_noise ${NOISE} \
    --obs_subsample ${SUBSAMPLE} \
    >& ${DIR}/run.out
if [ $? -eq 0 ]; then
    echo "OK!"
else
    echo "FAILED!"
fi

echo "ISJ..."
DIR="TEST_RUN/ISJ"
mkdir -p ${DIR}
python elliptic_inverse_sgplvm_joint.py \
    --joint_model 1 \
    --tf_num_threads ${NSLOTS} \
    --save_model_file "${DIR}/model.pkl" \
    --n_test ${N_TEST} \
    --save_dir ${DIR} \
    --obs_noise ${NOISE} \
    --obs_subsample ${SUBSAMPLE} \
    >& ${DIR}/run.out
if [ $? -eq 0 ]; then
    echo "OK!"
else
    echo "FAILED!"
fi

deactivate
