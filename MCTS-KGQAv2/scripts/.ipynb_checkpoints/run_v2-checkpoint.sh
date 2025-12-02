output_dir=/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs
propose_method=qwen14b
task_name=cwq
current_date=$(date +"%Y_%m_%d_%H_%M_%S")
filename="${task_name}_${current_date}.log"
mkdir -p $output_dir/logs/$task_name/

CUDA_VISIBLE_DEVICES=7 python /workspace/xxxxx/KGQA/MCTS-KGQAv2/evaluate_v2.py \
    --propose_method $propose_method \
    --task_name $task_name \
    --shuffle \
    --shuffle_times 2 \
    --num_plan_branch 7 \
    --num_branch 3 \
    --iteration_limit 40 \
> $output_dir/logs/$task_name/$filename
#     --use_vllm \
# > $output_dir/logs/$task_name/$filename