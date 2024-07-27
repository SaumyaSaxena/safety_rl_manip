#!/bin/bash

# Specify some hydra config parameters that we want to be constant across all runs for 
# this experiment. These arguments are provided directly to each file.
args=(
    # Add: Any extra command line arguments here.
    train_cfg.warmup=True
    # train_cfg.warmup_cfg.warmup_type=False
    train_cfg.warmup_cfg.add_expert_to_buffer=False
    train_cfg.warmup_cfg.mode='lagrange'
)

seeds=("8")

gpus=("1" "1" "2" "2" "1" "2")

lrs=("0.0001")

# Define the hydra config file we want to use for training.
config_name='train/train_point_mass_cont_ddpg.yaml'

echo "${args[@]}"

i=0
conda_bash="eval $(conda shell.bash hook)"
for lr in ${lrs[@]}; do
for seed in ${seeds[@]}; do
    echo "Running task:" $task
    echo ${cuda_devices[$i]}

    # This is the command that we will be using to run each "task" separately.
    cmd="HYDRA_FULL_ERROR=1 python train_ddpg.py \
         --config-name=${config_name} \
         seed=${seed} bc_kwargs.lr=${lr} gpu=${gpus[$i]}\
         ${args[@]}"
    echo $cmd

    # Create a temporary executable (.sh) file that sets the appropriate environment variables,
    # sets the right conda environments, cd into the right directory and then run the cmd above.
    run_filename="tmp_gen_1_${task}_idx_${i}.sh"
    cat > ${run_filename} <<EOF
#!/bin/zsh

echo "Will run"
eval $(conda shell.bash hook)
cd /home/saumyas/Projects/safe_control/safety_rl_manip
conda activate drabe
${cmd}
EOF

    # Now create a new tmux environment and run the above saved executable.
    chmod +x ${run_filename}
    sess_name="train_warmup_${seed}_${cuda_devices[$i]}_idx_${i}"
    tmux new-session -d -s "$sess_name" /home/saumyas/Projects/safe_control/safety_rl_manip/bash/${run_filename}

    i=`expr $i + 1`
    sleep 20
done
done