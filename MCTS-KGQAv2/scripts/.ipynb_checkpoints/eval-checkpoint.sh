output_dir=/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs
task_name=cwq
current_date=$(date +"%Y_%m_%d_%H_%M_%S")
filename="${task_name}_${current_date}.log"
mkdir -p $output_dir/logs/$task_name/evaluation

CUDA_VISIBLE_DEVICES=0 python /workspace/xxxxx/KGQA/MCTS-KGQAv2/answer_generation.py \
    --use_local_method False \
    --propose_method deepseekv3 \
# > $output_dir/logs/$task_name/evaluation/$filename