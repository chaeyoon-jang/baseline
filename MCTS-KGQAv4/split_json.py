import json
import pdb
import re
import os
from unicodedata import name
from collections import defaultdict


def split_json(datapath):
    data_dir = datapath.split('.json')[0]
    # pdb.set_trace()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    with open(datapath, 'r') as f:
        lines = f.readlines()
        for id, line in enumerate(lines):
            data = json.loads(line)
            with open(os.path.join(data_dir, str(id)+'.json'), 'w') as f2:
                json.dump(data, f2, indent=4)


def split_step(datapath):
    data_dir = datapath.split('.json')[0]
    # pdb.set_trace()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    with open(datapath, 'r') as f:
        lines = f.readlines()
        # pdb.set_trace()
        for index, line in enumerate(lines):
            data = json.loads(line)
            step_dict = defaultdict(list)
            tree_nodes = data['steps']
            for node in tree_nodes:
                step_info = node['node_details']['step_text']
                if len(step_info) != 0:
                    id = step_info[5]
                    key = 'step' + str(id)
                    step_dict[key].append(step_info)
                else:
                    pass
            # pdb.set_trace()
            with open(os.path.join(data_dir, str(index)+'_step.json'), 'w') as f2:
                json.dump(step_dict, f2, indent=4)


def find_max_number_after_step(s):
    # 定义要查找的子字符串
    step = "Step"
    
    # 查找所有包含 'step' 的部分
    matches = re.findall(rf'{step}\s+(\d+)', s)
    
    # 将找到的数字转换为整数并返回最大值
    if matches:
        return max(int(num) for num in matches)
    else:
        return None  # 如果没有找到数字，返回 None
    


def dump_json(source, datas):
    with open(source, 'w', encoding='utf-8') as f:
        for item in datas:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

# illegal_id_list = []
# illegal_dict_list = []
legal_id_list = []
legal_dict_list = []
incorrect_id_list = []
incorrect_dict_list = []

total_illegal_data_num = 0
total_correct_data_num = 0

def parse_shortcut(datapath, k):

    data_dir = datapath.split('.json')[0]
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    save_data_list = []
    with open(datapath, 'r') as f:
        lines = f.readlines()
        total_num = 0
        correct_num = 0
        topk_correct_num = 0
        illegal = 0
        id_list = []
        for id, line in enumerate(lines):
            save_data = defaultdict(list)
            data = json.loads(line)
            total_num += 1
            id2topic_entity_list = data['id2topic_entity_list']
            qid = data['qid']
            question = data['question']
            gt_answer = data['answer']
            answer_mid_list = data['answer_mid_list']
            
            # subquestions = data['subquestions']
            save_data['qid'] = qid
            save_data['question'] = question
            save_data['id2topic_entity_list'] = id2topic_entity_list
            save_data['gt_answer'] = gt_answer
            save_data['answer_mid_list'] = answer_mid_list
            save_data['subquestions'] = []


            all_path_with_Q = []
            answer_flag_path = []
            all_gt_path = []
            final_ans_flag = False
            topk_final_ans_flag = False
            
            for id2topic_entity in id2topic_entity_list:
                topic_entity = list(id2topic_entity.values())[0]
                if topic_entity in data.keys():
                    tree_list = data[topic_entity]['steps']
                    subquestions = data[topic_entity]['subquestions']
                    save_data['subquestions'] += subquestions
                    for node in tree_list: #遍历每一个节点
                        answer_flag = node['node_details']['meta_info']['final_ans_flag']
                        history_path = node['node_details']['meta_info']['history_step_info']
                        node_text = node['node_details']['node_text']
                        step_value = node['node_details']['step_value']
                        
                        for answer in gt_answer:
                            if answer == node_text or answer in node_text:
                                final_ans_flag = True
                                all_gt_path.append((node_text, history_path, step_value))
                        if not final_ans_flag:
                            for answer_mid in answer_mid_list:
                                if answer_mid == node_text:
                                    final_ans_flag = True
                                    all_gt_path.append((node_text, history_path, step_value))

                        if answer_flag:
                            answer_flag_path.append((node_text, history_path, step_value))
                        all_path_with_Q.append((node_text, history_path, step_value))

            sorted_all_path_with_Q = sorted(all_path_with_Q, key=lambda x: x[-1], reverse=True)
            topk_value = sorted(set(item[-1] for item in sorted_all_path_with_Q), reverse=True)[:k]
            topk_path_with_Q = [item for item in sorted_all_path_with_Q if item[-1] in topk_value]


            save_data['topk_path'] = topk_path_with_Q
            save_data['answer_flag_path'] = answer_flag_path
            save_data['all_gt_path'] = all_gt_path
            if final_ans_flag:
                correct_num += 1
                save_data['final_ans'] = True
            else:
                incorrect_dict = defaultdict(list)
                incorrect_dict[qid] = [{'question':question}, {'id2topic_entity_list':id2topic_entity_list}, {'gt_answer':gt_answer}]
                incorrect_dict_list.append(incorrect_dict)
                incorrect_id_list.append(qid)
                save_data['final_ans'] = False
                
            ########### gt_answer 和 answer_mid_list的长度有可能不一样############
            for answer in gt_answer:
                for path in topk_path_with_Q:
                    pre_entity, p, _ = path
                    if answer == pre_entity or answer in p:
                    # if answer == pre_entity:
                        topk_final_ans_flag = True
            if not topk_final_ans_flag:
                for answer_mid in answer_mid_list:
                    for path in topk_path_with_Q:
                        pre_entity, p, _ = path
                        if answer_mid == pre_entity or answer_mid in p:
                            topk_final_ans_flag = True

            if topk_final_ans_flag:
                # print(qid)
                topk_correct_num += 1
            

            save_data_list.append(save_data)
            # else:
            #     illegal_dict = defaultdict(list)
            #     illegal_dict[qid] = [{'question':question}, {'topic_entity_list':topic_entity_list}, {'gt_answer':gt_answer}]
            #     illegal_dict_list.append(illegal_dict)
            #     illegal_id_list.append(qid)
            #     illegal += 1
            #     save_data_list.append(save_data)
                
    global total_correct_data_num, total_illegal_data_num
    total_correct_data_num += correct_num
    total_illegal_data_num += total_num
    
    print(f"总数据有{total_num},topk正确数据有{topk_correct_num},比例为{topk_correct_num}/{total_num} = {topk_correct_num/total_num}")
    print(f"总数据有{total_num},正确数据有{correct_num},比例为{correct_num}/{total_num} = {correct_num/total_num}")
    dump_json(os.path.join(data_dir, 'shortcut.json'), save_data_list)


if __name__ == '__main__':
    path_list1 = ['/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/graliqa/mcts/qwen14b/qwen14b-2-7-4_1_18_14_39_43_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/graliqa/mcts/qwen14b/qwen14b-2-7-4_1_18_14_40_40_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/graliqa/mcts/qwen14b/qwen14b-2-7-4_1_18_14_41_54_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/graliqa/mcts/qwen14b/qwen14b-2-7-4_1_18_14_42_28_alltree.json']
    
    path_list2 = ['/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/cwq/mcts/qwen14b/qwen14b-2-7-4_1_18_20_7_48_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/cwq/mcts/qwen14b/qwen14b-2-7-4_1_18_20_8_8_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/cwq/mcts/qwen14b/qwen14b-2-7-4_1_18_20_8_24_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/cwq/mcts/qwen14b/qwen14b-2-7-4_1_18_20_8_36_alltree.json']
    
    path_list3 = ['/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/graliqa/mcts/qwen14b/qwen14b-2-7-4_1_13_16_19_51_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/graliqa/mcts/qwen14b/qwen14b-2-7-4_1_17_21_57_19_alltree.json',]
    
    path_list4 = ['/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-4_1_27_10_18_49_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-4_1_27_10_20_13_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-4_1_27_10_20_30_alltree.json',
                  '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-4_1_27_10_20_49_alltree.json']
    
    datapath = '/workspace/xxxxxx/KGQA/MCTS-KGQAv4/outputs/graliqa/mcts/qwen14b/qwen14b-2-7-4_1_13_16_19_51_alltree.json'
    # parse_shortcut(datapath, 10)
    # split_json(datapath)
    # split_step(datapath)
    for path in path_list2:
        task = path.split('/')[-4]
        parse_shortcut(path, 10)
        
    print('\n', '*'*20, f'错误问题集合数量:{len(incorrect_id_list)} ', '*'*20, incorrect_id_list,)
    print(f"总数据有{total_illegal_data_num},正确数据有{total_correct_data_num},比例为{total_correct_data_num}/{total_illegal_data_num} = {total_correct_data_num/total_illegal_data_num}")
    # output_dir = f'/workspace/xxxxxx/KGQA/MCTS-KGQAv2/outputs/{task}'
    # with open(output_dir + '_incorrect.json', 'w', encoding='utf-8') as f:
    #     json.dump(incorrect_dict_list, f, ensure_ascii=False, indent=4)
    # with open(output_dir + '_illegal.json', 'w', encoding='utf-8') as f:
    #     json.dump(illegal_dict_list, f, ensure_ascii=False, indent=4)
    
            
