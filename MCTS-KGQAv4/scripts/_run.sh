output_dir=/home/xxxxxxxx/mcts/MCTS/outputs
task_name=webqsp
current_date=$(date +"%Y_%m_%d_%H_%M")
filename="${task_name}_${current_date}.log"
mkdir -p $output_dir/logs/$task_name/

CUDA_VISIBLE_DEVICES=3 python /workspace/xxxx/KGQA/MCTS-KGQA/evaluate.py \
    --task_name $task_name \
    --shuffle \
    --shuffle_times 2 \
    --num_plan_branch 8 \
    --num_branch 4 \
    --iteration_limit 60 \
#     --use_vllm \
# > $output_dir/logs/$task_name/$filename