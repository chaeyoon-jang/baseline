import json
import os
import sys
sys.path.append('..')
from collections import defaultdict
from models.inference_models import *
import argparse
from tqdm import tqdm



os.environ['CUDA_VISIBLE_DEVICES'] = '4'

path_list2 = ['/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/webquestions/webquestion_legal_topic.json']



base_args = argparse.ArgumentParser()
base_args.add_argument('--propose_method', type=str, choices=['qwen14b', 'qwenqwq', 'qwen32b', 'deepseekv3'], default='deepseekv3')
base_args.add_argument('--truncation', type=bool, default=True)
base_args.add_argument('--temperature', type=float, default=0.7)
base_args.add_argument('--max_len', type=int, default=8000)
base_args.add_argument('--max_new_tokens', type=int, default=256)
base_args.add_argument('--do_sample', type=bool, default=False)
base_args.add_argument('--use_vllm', default=False, action="store_true")

# base_args.add_argument('--model_id', type=int, default=4)
arguments = base_args.parse_args()


tokenizer, model = None, None
# model_path = '/workspace/LLaMA-Factory/models/Qwen2.5-14B-Instruct'
outputpath = '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/webquestions/webquestion_legal_topic_prediction.json'
# tokenizer, model = get_inference_model_qwen(model_path)
io_system = IO_System(args=arguments, tokenizer=tokenizer, model=model)


prompt="""
##Instruction:
Suppose you are an **expert** in problem analysis, and you will receive a complex problem. Your task is to carefully analyze the problem and identify the core thematic entity within it.
Noted: This entity must exist in the problem; please do not fabricate it yourself, and ensure that the entity remains consistent with the one in the problem.
##Example 1
**Question**:
what does jamaican people speak?
##Ouput:
```
<entity> jamaican </entity>
```
##Example 2
**Question**:
who was vice president after kennedy died?
##Ouput:
```
<entity> John F. Kennedy </entity>
```
##Example 3
**Question**:
what did george clemenceau do?
##Ouput:
```
<entity> Georges Clemenceau </entity>
```
##Input
**Question**:
{question}
"""

def dump_json(source, datas):
    with open(source, 'w', encoding='utf-8') as f:
        for item in datas:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def parse_xml(response:str, key='choice') -> str:
    """
    response: <choice> location.country.languages_spoken </choice>
    return: location.country.languages_spoken
    """
    start_symbol, end_symbol = f'<{key}>', f'</{key}>'
    start_idx = response.find(start_symbol) + len(start_symbol)
    end_idx = response.rfind(end_symbol)
    if start_idx == -1 or end_idx == -1:
        return None
    value = response[start_idx:end_idx].strip()
    return value

save_data_list =[]
for path in path_list2:
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            # save_data = defaultdict(list)
            data = json.loads(line)
            if data["legal"]:
                question = data['question']
                topic_entity = data['topic_entity']

                input = prompt.format(question=question)
                # response = io_system.get_local_response(query=input,
                #                                         max_length=arguments.max_len,
                #                                         max_new_tokens=arguments.max_new_tokens,
                #                                         temperature=arguments.temperature,
                #                                         do_sample=arguments.do_sample,
                #                                         truncation=arguments.truncation)
                response = io_system.get_api_response(query=input)
                pre_list =[]
                for rawdata in response:
                    pre = parse_xml(rawdata, key='entity')
                    if pre:
                        pre_list.append(pre)
                data['topic_pre'] = pre_list
                data['topic_flag'] = False
                for pre in pre_list:
                    for top in topic_entity:
                        if pre.lower() in top.lower or top.lower() in pre.lower():
                            data['topic_flag'] = True
                save_data_list.append(data)
dump_json(outputpath, save_data_list)