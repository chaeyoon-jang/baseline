import json
import os
import sys
sys.path.append('..')
from collections import defaultdict
from models.inference_models import *
import argparse
from tqdm import tqdm



os.environ['CUDA_VISIBLE_DEVICES'] = '4'

path_list2 = ['/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1-320_alltree.json',
                '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_320-640_alltree.json',
                '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_640-960_alltree.json',
                '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_960-1280_alltree.json',
                '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1280-1600_alltree.json',]



base_args = argparse.ArgumentParser()
base_args.add_argument('--propose_method', type=str, choices=['qwen14b', 'qwenqwq', 'qwen32b', 'deepseekv3'], default='qwen14b')
base_args.add_argument('--truncation', type=bool, default=True)
base_args.add_argument('--temperature', type=float, default=0.7)
base_args.add_argument('--max_len', type=int, default=8000)
base_args.add_argument('--max_new_tokens', type=int, default=256)
base_args.add_argument('--do_sample', type=bool, default=False)
base_args.add_argument('--use_vllm', default=False, action="store_true")

# base_args.add_argument('--model_id', type=int, default=4)
arguments = base_args.parse_args()



model_path = '/workspace/LLaMA-Factory/models/Qwen2.5-14B-Instruct'
outputpath = '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/qwen14b_prediction.json'
tokenizer, model = get_inference_model_qwen(model_path)
io_system = IO_System(args=arguments, tokenizer=tokenizer, model=model)


prompt="""
Please answer the question. You only need to output the answer, without any additional irrelevant content, such as explanations. The question is {question}
"""

def dump_json(source, datas):
    with open(source, 'w', encoding='utf-8') as f:
        for item in datas:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


save_data_list =[]
for path in path_list2:
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            save_data = defaultdict(list)
            data = json.loads(line)
            is_legal = data['is_legal']
            qid = data['qid']
            question = data['question']
            gt_answer = data['answer']
            if is_legal:
                save_data['qid'] = qid
                save_data['question'] = question
                save_data['answer'] = gt_answer
                question = question + '?'
                input = prompt.format(question=question)
                response = io_system.get_local_response(query=input,
                                                        max_length=arguments.max_len,
                                                        max_new_tokens=arguments.max_new_tokens,
                                                        temperature=arguments.temperature,
                                                        do_sample=arguments.do_sample,
                                                        truncation=arguments.truncation)
                save_data['prediction'] = response
                save_data_list.append(save_data)
dump_json(outputpath, save_data_list)