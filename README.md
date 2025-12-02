# Enhancing Large Language Models with Reward-guided Tree Search for Knowledge Graph Question and Answering
This is the official implementation code of the RTSoG algorithm, which leverages large language models for reasoning-assisted guidance to perform question answering on existing knowledge graphs. The related paper, "[Enhancing Large Language Models with Reward-guided Tree Search for Knowledge Graph Question and Answering]," proposes a reward-guided tree search framework for knowledge graph question answering. The core idea of this framework is to utilize large language models as agent intelligences to iteratively explore and reason over knowledge within the knowledge graph. During this process, an innovative Self-Critic Monte Carlo Tree Search (SC-MCTS) algorithm is employed for knowledge exploration, enabling "reflection-backtracking-correction" in reasoning.

<img src="framework1.png" width = "800" />

<img src="framework2.png" width = "600" />

## Requirements
1、Hardware Requirements: An NVIDIA GPU environment is required, with CUDA version 12.4 or higher and driver version 550 or above.
2、Install Dependencies: Set up the required environment by running the command:
```
pip install -r requirements.txt
```
3、Knowledge Graph Setup: First, configure the knowledge graph environment. For details, refer to the freebase folder.


## Version illustration
We provide two solutions for different data formats. The first solution, version v2, is designed for the subgraph data format used in RoG. The second solution, version v4, is tailored for direct SPARQL retrieval from large-scale knowledge graphs like Freebase, as implemented in frameworks such as ToG.



## How to run
First, after configuring the knowledge graph environment according to the instructions in the freebase folder, you need to download your required large language model (e.g., Qwen2.5-7B) locally and configure the relevant paths in the script files, and navigate to the directory of your chosen version.

### Step 1: Generate the Reasoning Tree for the Question
1、Run sh scripts/run_v2.sh to start the code and build the reasoning tree.

```
output_dir=/workspace/xxxxxx/KGQA/RTSoG/outputs
propose_method=qwen14b
task_name=cwq
current_date=$(date +"%Y_%m_%d_%H_%M_%S")
filename="${task_name}_${current_date}.log"
mkdir -p $output_dir/logs/$task_name/

CUDA_VISIBLE_DEVICES=3 python /workspace/xxxxxx/KGQA/RTSoG/evaluate_v2.py \
    --propose_method $propose_method \
    --task_name $task_name \
    --use_freebase \
    --shuffle \
    --shuffle_times 2 \
    --num_plan_branch 7 \
    --num_branch 4 \
    --iteration_limit 7 \
> $output_dir/logs/$task_name/$filename
```

The generated reasoning tree will be saved in the output directory.

### Step 2: Parse the Generated JSON File to Obtain the shortcut.json File

```
python split_json.py
```

### Step 3: Answer Generation
Run sh scripts/eval.sh to perform answer reasoning and generate the final results.
```
output_dir=your_output_dir
task_name=cwq
current_date=$(date +"%Y_%m_%d_%H_%M_%S")
filename="${task_name}_${current_date}.log"
mkdir -p $output_dir/logs/$task_name/evaluation

CUDA_VISIBLE_DEVICES=4 python /workspace/answer_generation.py \
    --use_local_method False \
    --propose_method deepseekv3 \
> $output_dir/logs/$task_name/evaluation/$filename
```

## Acknowledgement
We refer to the data processing code of ToG[https://github.com/DataArcTech/ToG] and RoG[https://github.com/RManLuo/reasoning-on-graphs]. Thanks for their contributions.