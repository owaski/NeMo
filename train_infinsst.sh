#!/bin/bash
#SBATCH -A convai_convaird_nemo-speech 
#SBATCH -J "infinisst"            # job name (<< CHANGE ! >>)
#SBATCH -p batch_singlenode
#SBATCH -N 1  # number of nodes
#SBATCH -t 4:00:00              # wall time
#SBATCH --time-min 04:00:00
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0

set -x

if [ -z "$1" ]; then
    echo "First argument (random seed) is missing"
    exit 1
fi
SEED="${1}"

GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
TOTAL_NUM_GPUS=`expr $GPUS_PER_NODE \* $SLURM_JOB_NUM_NODES`

WANDB="66ac1187790dd51beb174e9aa8e4ec58c8a8c25b" # replace with your own WandB API key

ROOT=/lustre/fsw/portfolios/convai/users/souyang

CONTAINER=$ROOT/images/nemo-25.04.01.sqsh
CODE_DIR=$ROOT/code
LHOTSE_DIR=$ROOT/code/lhotse
CKPTS_DIR=$ROOT/ckpts
DATA_DIR=$ROOT/data
HF_CACHE_DIR=$ROOT/.cache/huggingface

MOUNTS='--container-mounts=${CODE_DIR}:/code,${CKPTS_DIR}:/ckpts,${DATA_DIR}:/data,$HF_CACHE_DIR:/hfcache'

# TODO: install whisper_normalizer to docker image

CONFIG_PATH=$CODE_DIR  # Adjust if launching from outside this directory.
CONFIG_NAME="train_infinisst"
EXP_NAME="${CONFIG_NAME}_${SLURM_JOB_NUM_NODES}node"
RESULTS_DIR="${CKPTS_DIR}/infinisst/dev/${EXP_NAME}"
mkdir -p ${RESULTS_DIR}

read -r -d '' cmd <<EOF
export WANDB_API_KEY="${WANDB}" \
&& export AIS_ENDPOINT="http://asr.iad.oci.aistore.nvidia.com:51080" \
&& export PYTHONPATH="${CODE_DIR}:${LHOTSE_DIR}:${PYTHONPATH}" \
&& export HF_HOME="/hfcache" \
&& export OMP_NUM_THREADS=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 \
python /code/NeMo/examples/speechlm2/infinisst_train.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    exp_manager.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    exp_manager.explicit_log_dir=${RESULTS_DIR} \
    data.train_ds.seed=$SEED \
    data.validation_ds.seed=$SEED 
EOF


#trainer.strategy.data_parallel_size=${TOTAL_NUM_GPUS} \

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

CNAME=staszek
srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" --container-name=$CNAME $MOUNTS bash -c "${cmd}"
