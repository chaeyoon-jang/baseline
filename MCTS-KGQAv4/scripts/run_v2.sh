output_dir=/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs
propose_method=qwen14b
task_name=cwq
current_date=$(date +"%Y_%m_%d_%H_%M_%S")
filename="${task_name}_${current_date}.log"
mkdir -p $output_dir/logs/$task_name/

CUDA_VISIBLE_DEVICES=3 python /workspace/xxxxxx/KGQA/MCTS-KGQAv4/evaluate_v2.py \
    --propose_method $propose_method \
    --task_name $task_name \
    --use_freebase \
    --shuffle \
    --shuffle_times 2 \
    --num_plan_branch 7 \
    --num_branch 4 \
    --iteration_limit 7 \
> $output_dir/logs/$task_name/$filename
#     --use_vllm \
# > $output_dir/logs/$task_name/$filename