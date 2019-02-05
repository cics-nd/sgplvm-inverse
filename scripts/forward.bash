#!/bin/bash
# Forward experiments
#
# Usage:
#
# ./forward.bash [1] [2] [3] [4]
#
# [1]: KL terms (32 or 128)
# [2]: separate or joint
# [3]: Kernel type (options=PCA, Linear, RBF, or Sum)
#      NOTE: Can only use PCA with "separate"
# [4]: number of training data

if [ ! $# -eq 4 ]; then
    echo "./forward.bash [32 or 128] [separate or joint] [kernel type] \
[training data]"
    exit 1
fi

if [ -z ${NSLOTS+x} ]; then
    echo "Set NSLOTS=1"
    NSLOTS=1
fi
KL=${1}
MODEL_TYPE=${2}
KERNEL_TYPE=${3}
N=${4}

#===============================================================================

TIME_STAMP_WITH_SPACES=$(date)
TIME_STAMP=${TIME_STAMP_WITH_SPACES// /_}
echo "Begin at ${TIME_STAMP}"
if [ ! -z ${OMP_NUM_THREADS+x} ]; then
    if [ ! ${OMP_NUM_THREADS} -eq ${NSLOTS} ]; then
        echo "export OMP_NUM_THREADS=${NSLOTS}"
        export OMP_NUM_THREADS=${NSLOTS}
    fi
else
    echo "export OMP_NUM_THREADS=${NSLOTS}"
    export OMP_NUM_THREADS=${NSLOTS}
fi

cd ..
source venv/bin/activate

if [ ${MODEL_TYPE} == "separate" ]; then
    MODEL_DIR="results/forward-${KL}/2M-${KERNEL_TYPE}/${N}"
    if [ ! -d ${MODEL_DIR} ]; then
        mkdir -p ${MODEL_DIR}
    fi
    python elliptic_forward_sgplvm_separate.py \
        --tf_num_threads ${NSLOTS} \
        --xi_kern_in ${KERNEL_TYPE} \
        --xi_kern_out RBF \
        --n_train ${N} \
        --save_model_file_in ${MODEL_DIR}/model_in.pkl \
        --save_model_file_out  ${MODEL_DIR}/model_out.pkl \
        --train_dir elliptic/data/kl_16_${KL}_train \
        --test_dir elliptic/data/kl_16_${KL}_test \
        --save_dir ${MODEL_DIR} \
        >& ${MODEL_DIR}/run_${TIME_STAMP}.out
elif [ ${MODEL_TYPE} == "joint" ]; then
    MODEL_DIR="results/forward-${KL}/JM-${KERNEL_TYPE}/${N}"
    if [ ! -d ${MODEL_DIR} ]; then
        mkdir -p ${MODEL_DIR}
    fi
    python elliptic_forward_sgplvm_joint.py \
        --tf_num_threads ${NSLOTS} \
        --joint_model 1 \
        --xi_kern ${KERNEL_TYPE} \
        --n_train ${N} \
        --save_model_file ${MODEL_DIR}/model.pkl \
        --train_dir elliptic/data/kl_16_${KL}_train \
        --test_dir elliptic/data/kl_16_${KL}_test \
        --save_dir ${MODEL_DIR} \
        >& ${MODEL_DIR}/run_${TIME_STAMP}.out
else
    echo "Unrecognized model type ${MODEL_TYPE}"
fi

deactivate

T_END=$(date)
echo "Finished at ${T_END}"
