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

illegal_id_list = []
illegal_dict_list = []
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
            is_legal = data['is_legal']
            if is_legal:
                total_num += 1
            topic_entity_list = data['topic_entity_list']
            qid = data['qid']
            question = data['question']
            gt_answer = data['answer']
            # subquestions = data['subquestions']
            save_data['qid'] = qid
            save_data['question'] = question
            save_data['is_legal'] = is_legal
            save_data['topic_entity_list'] = topic_entity_list
            save_data['gt_answer'] = gt_answer
            save_data['subquestions'] = []

            if len(data.keys()) > 5:
                all_path_with_Q = []
                answer_flag_path = []
                all_gt_path = []
                final_ans_flag = False
                topk_final_ans_flag = False
                for topic_entity in topic_entity_list:
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
                                if answer == node_text:
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
                    incorrect_dict[qid] = [{'question':question}, {'topic_entity_list':topic_entity_list}, {'gt_answer':gt_answer}]
                    incorrect_dict_list.append(incorrect_dict)
                    incorrect_id_list.append(qid)
                    save_data['final_ans'] = False
                
                for answer in gt_answer:
                    for path in topk_path_with_Q:
                        pre_entity, p, _ = path
                        # if answer == pre_entity or answer in p:
                        if answer == pre_entity:
                            topk_final_ans_flag = True
                if topk_final_ans_flag:
                    # print(qid)
                    topk_correct_num += 1
                

                save_data_list.append(save_data)
            else:
                illegal_dict = defaultdict(list)
                illegal_dict[qid] = [{'question':question}, {'topic_entity_list':topic_entity_list}, {'gt_answer':gt_answer}]
                illegal_dict_list.append(illegal_dict)
                illegal_id_list.append(qid)
                illegal += 1
                save_data_list.append(save_data)
                
    global total_correct_data_num, total_illegal_data_num
    total_illegal_data_num += total_num
    total_correct_data_num += correct_num
    print(f"能找到答案的数据数量{total_num},不合法的数据数量{illegal}")
    print(f"总数据有{total_num},topk正确数据有{topk_correct_num},比例为{topk_correct_num}/{total_num} = {topk_correct_num/total_num}")
    print(f"总数据有{total_num},正确数据有{correct_num},比例为{correct_num}/{total_num} = {correct_num/total_num}")
    dump_json(os.path.join(data_dir, 'shortcut.json'), save_data_list)


if __name__ == '__main__':
    path_list1 = ['/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_12_27_20_31_2_alltree.json',
                 '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_12_27_20_31_32_alltree.json',
                 '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_12_27_20_32_5_alltree.json',
                 '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_12_27_20_32_26_alltree.json',
                 '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_12_27_20_33_18_alltree.json',
                 '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_12_27_20_33_45_alltree.json',
                 '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_12_27_20_34_12_alltree.json',
                 '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_12_29_20_55_37_alltree.json']
    
    path_list2 = ['/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1-320_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_320-640_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_640-960_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_960-1280_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1280-1600_alltree.json',]
    
    path_list3 = ['/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1-450_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_2_10_38_55_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_2_10_40_51_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_2_10_42_8_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_2_19_33_38_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_2_19_34_29_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_4_18_2_56_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_9_13_42_12_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_12_13_18_2_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_12_13_19_54_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_12_13_20_35_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_12_13_21_54_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_12_13_22_39_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_12_13_23_16_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1_12_13_23_45_alltree.json']
    
    path_list4 = ['/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1-450_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_450-900_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_900-1350_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1350-1800_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1800-2250_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_2250-2700_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_2700-3150_alltree.json',
                  '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_3150-3600_alltree.json']
    
    # path_list4 = [] 
    datapath = '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-4-7-4_1_6_14_22_43_alltree.json' # bad case refine
    # datapath = '/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-1-200_alltree.json'
    # parse_shortcut(datapath, 10)
    # split_json(datapath)
    # split_step(datapath)
    for path in path_list4:
        task = path.split('/')[-4]
        parse_shortcut(path, 10)
        
    print('\n', '*'*20, f'不合法的实体数量{len(illegal_id_list)}: ', '*'*20, illegal_id_list, )
    print('\n', '*'*20, f'错误问题集合数量:{len(incorrect_id_list)} ', '*'*20, incorrect_id_list,)
    print(f"总数据有{total_illegal_data_num},正确数据有{total_correct_data_num},比例为{total_correct_data_num}/{total_illegal_data_num} = {total_correct_data_num/total_illegal_data_num}")
    # output_dir = f'/workspace/xxxxx/KGQA/MCTS-KGQAv2/outputs/{task}'
    # with open(output_dir + '_incorrect.json', 'w', encoding='utf-8') as f:
    #     json.dump(incorrect_dict_list, f, ensure_ascii=False, indent=4)
    # with open(output_dir + '_illegal.json', 'w', encoding='utf-8') as f:
    #     json.dump(illegal_dict_list, f, ensure_ascii=False, indent=4)
    
    # p1 = incorrect_id_list[:130]
    # p2 = incorrect_id_list[130: 260]
    # p3 = incorrect_id_list[260:390]
    # p4 = incorrect_id_list[390:]
    # print(p1, '\n\n', p2, '\n\n', p3, '\n\n', p4)

