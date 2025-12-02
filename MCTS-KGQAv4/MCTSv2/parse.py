import json
import pdb
import sys
sys.path.append('..')
from collections import defaultdict
import os
from tqdm import tqdm
from utils.verify_MATH import *


def dump_json(source, datas):
    with open(source, 'w', encoding='utf-8') as f:
        for item in datas:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            


def parse_shortcut(datapath):
    data_dir = datapath.split('.json')[0]
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    save_data_list = []
    with open(datapath, 'r') as f:
        lines = f.readlines()
        for id, line in enumerate(lines):
            save_data = defaultdict(list)
            data = json.loads(line)
            question = data['question']
            gt_answer = data['gt_answer']
            shortcut_path_list = data['shortcut_path']
            save_data['question'] = question
            save_data['ground_truth_answer'] = gt_answer
            save_data['ground_truth_solution'] = None
            save_data['back'] = None
            save_data['process'] = None
            save_data['process_label'] = None
            save_data['process_label_type'] = None
            # save_data['shortcut_solution'] = 
            shortcut_path_new_list = []
            for path in shortcut_path_list:
                if '<|im_start|>' in path:
                    continue
                else:
                    shortcut_path_new_list.append(path)
            if len(shortcut_path_new_list) > 0:
                shortcut_path = shortcut_path_new_list[0]
            else:
                shortcut_path = shortcut_path_list[0]
            save_data["MCTS_solution"] = shortcut_path
            save_data["MCTS_solution_sft"] = shortcut_path
            save_data["MCTS_solution_type"] = "MCTS"
            save_data["source_from"] = "REST-MCTS"
            save_data["author"] = "longxiao"

            if len(shortcut_path) != 0:
                save_data['shortcut_path'] = shortcut_path
                save_data_list.append(save_data)

    dump_json(os.path.join(data_dir, 'sft_shortcut.json'), save_data_list)
        # outputpath = os.path.join(outputpath, '1.json')
        # for key in step.keys():
        #     step_list = step[key]
        #     step_string = str(key) + ', \n'.join(step_list)
        # dump_json(outputpath, step)



def verify(datapath):
    data_dir, dataname = '/'.join(path.split('/')[:-1]), path.split('/')[-1]
    data_list = []
    with open(datapath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            gt_answer = data['ground_truth_answer']
            solution = data["MCTS_solution"]
            if math_answer_compare(gt_answer, solution):
                data_list.append(data)
    dump_json(os.path.join(data_dir, 'v2_' + dataname), data_list)



def count_leaf_node(datapath, gt):
    leaf_node = []
    with open(datapath) as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            for key in data.keys():
                step_list = data[key]
                for step in step_list:
                    if "boxed" in step:
                        answer = math_postprocess_v2(step)
                        if answer != '' and answer.isdigit(): # 针对只有answer的 没有输出完全的情况
                            leaf_node.append(answer)



if __name__ == '__main__':
    path = '/home/longxiao.lx/mcts/ReST-MCTSv4/outputs/math/amc/1081_all/shortcut/polished_normal_sft_shortcut.json'
    # parse_shortcut(path)
    verify(path)