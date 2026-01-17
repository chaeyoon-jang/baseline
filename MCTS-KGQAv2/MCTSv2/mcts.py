import time
import math
import random
import numpy
import sys
from collections import defaultdict
sys.path.append('..')
sys.path.append('.')
from functools import partial
import copy
from MCTSv2.base import treeNode
from utils.tools import *
import networkx as nx
from utils.graph_utils import *
import pdb
from utils.verify_MATH import *
import re
import torch
from sentence_transformers import SentenceTransformer




def selectNode(node, mcts_task):
    # 判断节点是否完全展开 并选择最好的子节点
    while node.isFullyExpanded:# 完全展开了 就根据ucb选择 
        node = getBestChild(node, mcts_task)
    if isTerminal(node, mcts_task): # 如果是终点 返回node true false判断是否为终点叶子节点final-anser
        node.final_ans_flag = 1
        return True, node
    else: # 如果不是终点
        return False, node 

def isTerminal(node: treeNode, mcts_task) -> bool:
    if node.V >= mcts_task.end_gate or node.depth > mcts_task.limited_depth:
        return True
    else:
        return False

def expand(node: treeNode, 
           mcts_task, 
           graph:nx.Graph, 
           question:str, 
           topic_entity:str,
           iter: int,
           subquestions_list: list=None,
           shuffle: bool=False,
           shuffle_times: int=1):
    """
    扩展节点进行边的选择 节点是当前选择的node

    """ 
    if iter == 0: # 这个是最开始从topic entity扩展 边的数量可以多一些

        relations2score_dict = get_next_relations_expand(node=node, 
                                                         mcts_task=mcts_task, 
                                                         graph=graph, 
                                                         question=question,
                                                         topic_entity=topic_entity,
                                                         num_branch=mcts_task.num_plan_branch,
                                                         subquestions_list=subquestions_list,
                                                         shuffle=shuffle,
                                                         shuffle_times=shuffle_times) # relations_with_score: {relation1: score1, relation2: score2, ...} 可以选择检索还是生成
        if len(relations2score_dict) == 0: # 说明这个topic entity不在子图当中 直接返回
            node.reflection = '<end>'
            return node
        # pdb.set_trace()

    else: # 后序的展开都是边的数量有所不同 边的数量3
        relations2score_dict = get_next_relations_expand(node=node, 
                                                         mcts_task=mcts_task, 
                                                         graph=graph, 
                                                         question=question,
                                                         topic_entity=topic_entity,
                                                         num_branch=mcts_task.num_branch,
                                                         subquestions_list=subquestions_list,
                                                         shuffle=shuffle,
                                                         shuffle_times=shuffle_times) # relations_with_score: {relation1: score1, relation2: score2, ...} 可以选择检索还是生成
    if len(relations2score_dict) == 0:# 说明该节点（不在graph中） 目前不会出现这种情况 因为 我们对问题进行了过滤
        node.reflection = '<end>'
        return node
    # 保证relations2score_dict不为空
    # pdb.set_trace()
    actions = get_actions_do_reweight(node=node, 
                                      mcts_task=mcts_task,
                                      graph=graph,
                                      question=question,
                                      topic_entity=topic_entity,
                                      subquestions_list=subquestions_list,
                                      relations2score_dict=relations2score_dict) # actions: {(ri, ti): scorei, ...}
    # pdb.set_trace()
    if len(actions) == 0: # 如果展开为空
        return node
    for (relation, node_text), value in actions.items():
        if node_text not in node.children.keys():
            # pdb.set_trace()
            node.append_children(node_text, relation) # treenode 增加step作为child node.children:{}
            child = node.children[node_text] # node.children {node_text1 :treenode, nodetext2: treenode, nodetext3: treenode}
            if relation in  relations2score_dict.keys():
                pre_value = relations2score_dict[relation]
            # 这里 score 和 value 怎么平衡
            try:
                value = float(value)
            except Exception as e:
                continue
            value = 2/3 * value + 1/3 * pre_value
            child.update_value(value)

    node.isFullyExpanded = True # 只要actions不为空 认为node已经被完全展开了
    return node


def back_propagate(node):
    while node is not None:
        node.numVisits += 1
        if node.isFullyExpanded:
            child_Vs = [child.V * child.numVisits for child in node.children.values()]
            total_num_visits = sum([child.numVisits for child in node.children.values()])
            if total_num_visits > 0:
                node.V = sum(child_Vs) / total_num_visits
        node = node.parent

def getBestChild(node, mcts_task):
    bestValue = mcts_task.low
    bestNodes = []
    for child in node.children.values():
        nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)

def math_cheak(gt, answer):
    eval_res_1 = math_answer_compare(gt, answer)
    return eval_res_1

def get_next_steps_roll(y: str, step_n: int, mcts_task):
    # pdb.set_trace()
    next_steps = []
    for i in range(mcts_task.roll_branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_step(y, step_n)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps

def get_next_relations_expand(node: treeNode, 
                              mcts_task, 
                              graph:nx.Graph, 
                              question: str, 
                              topic_entity: str,
                              num_branch: int,
                              subquestions_list: list=None,
                              shuffle_times: int=3,
                              shuffle: bool=False) -> dict:
    """
    功能 给定头实体 筛选并打分 邻接关系 的数量计算 每10个关系 选择出最好的2个候选 最后的关系数量 由num_branch控制 第一步扩展 8条关系 之后的扩展 3条关系
    input 实体的文本 在这中间我们进行筛选过滤
    这里可以选择 进行shuffle 或者不进行shuffle 按照关系的数量进行分组 10个为一组
    output 筛选的邻接的关系以及分数 relations2score_dict {relation1:score1, relation2:score2, ...} 如果没有则返回空字典
    """
    # pdb.set_trace()
    relations2score_dict = defaultdict(list) # {relation1: [score1, score2, ...], relation2: [score2,] ...} 
    reflection = node.reflection
    history_path = node.history_path # 提取当前节点的历史路径
    node_text = node.node_text # 根据当前的选择的node的实体信息 进行展开
    subquestions_list_str = convert_subquestion_list2str(subquestions_list)
    questions_list = [question] + subquestions_list # [原问题, 子问题1, 子问题2, 子问题3] 这个用来筛选关系的 当数量大于40
    ######################### 先不实现 #############################
    if mcts_task.use_generator:
        for i in range(mcts_task.branch):
            proposal = ''
            cnt = 3
            while not proposal and cnt:
                if mcts_task.use_reflection:
                    proposal = mcts_task.get_next_step_use_reflection(node.y, node.depth + 1, reflection)
                else:
                    proposal = mcts_task.get_next_step(node.y, node.depth + 1)
                cnt -= 1
            if not proposal:
                continue
            relations2score_dict.append(proposal)
    ##############################################################
  
    else: # 不使用生成器 生成单步的推理路径
        # 在这里我们 对互逆的关系进行区分
        if node.pre_relation != '':
            node_all_edges = get_one_entity_all_edges(node_text, graph, pre_relation=node.pre_relation) # all_edges_list [r1, r2, ....] 筛掉逆关系
        else:
            node_all_edges = get_one_entity_all_edges(node_text, graph)
        edges_num = len(node_all_edges) 
        ######### 如果边大于40 则进行embed 粗筛选 ##########
        # if edges_num > 40:
        #     emb_model = SentenceTransformer('/workspace/LLaMA-Factory/models/gte_Qwen2-7B-instruct', device="cpu")
        #     node_all_edges = retrieve_top_docs(query_list=questions_list, 
        #                                        docs_list=node_all_edges, 
        #                                        model=emb_model, 
        #                                        topk=40)
        ######### 如果边大于40 则进行embed 粗筛选 ##########  
        print("*"*50, f'{node_text}的邻接关系有{edges_num}条', '*'*50)       
        if edges_num == 0: # 如果为空 说明此节点不在子图当中
            return relations2score_dict 
        
        elif edges_num > 10: # 进行分割解析 每十个选择两个 少于十个的 选择三个 
            
            if shuffle: # 在这里先进行shuffle打乱列表
                for _ in range(shuffle_times):
                    shuffle_node_all_edges = random.sample(node_all_edges, len(node_all_edges)) # 打乱list
                    chunks = split_list_into_chunks(lst=shuffle_node_all_edges, chunk_size=10)
                    for chunk in chunks:
                        relations2score_dict_part = mcts_task.filter_and_score_edges(question=question, 
                                                                                     node_all_edges=chunk, 
                                                                                     topic_entity=topic_entity,
                                                                                     history_path=history_path,
                                                                                     subquestions_list_str=subquestions_list_str,
                                                                                     budget=2) # # {relation1: score1, relation2: score2, ...} sum of score is 1 一个问题的10个关系 回答n次 每次选两个 可能有多样性这里 所以每次返回的可能不止两个
                        for relation, score in relations2score_dict_part.items():
                            relations2score_dict[relation].append(score)
                    # pdb.set_trace()
            else:
                chunks = split_list_into_chunks(lst=node_all_edges, chunk_size=10)
                for chunk in chunks:
                    relations2score_dict_part = mcts_task.filter_and_score_edges(question=question, 
                                                                                 node_all_edges=chunk, 
                                                                                 topic_entity=topic_entity,
                                                                                 history_path=history_path,
                                                                                 subquestions_list_str=subquestions_list_str,
                                                                                 budget=2) # # {relation1: score1, relation2: score2, ...} sum of score is 1
                    for relation, score in relations2score_dict_part.items():
                        relations2score_dict[relation].append(score)
                    # pdb.set_trace()
            # 这里最后的relations2score_dict是最后对每个关系打分之后的结果 dict(list)
            # pdb.set_trace()
            relations2score_dict = get_top_k_items(dictionary=relations2score_dict, k=num_branch)
        else: # 小于10个关系 也可以进行多次采样 取top
            if shuffle:
                for _ in range(shuffle_times):
                    shuffle_node_all_edges = random.sample(node_all_edges, len(node_all_edges)) # 打乱list
                    relations2score_dict_part = mcts_task.filter_and_score_edges(question=question, 
                                                                                 node_all_edges=shuffle_node_all_edges, 
                                                                                 topic_entity=topic_entity,
                                                                                 history_path=history_path,
                                                                                 subquestions_list_str=subquestions_list_str,
                                                                                 budget=3) # # {relation1: score1, relation2: score2, ...}
                    for relation, score in relations2score_dict_part.items():
                        relations2score_dict[relation].append(score)
                    # pdb.set_trace()
                relations2score_dict = get_top_k_items(dictionary=relations2score_dict, k=num_branch)
            else: # 
                relations2score_dict = mcts_task.filter_and_score_edges(question=question, 
                                                                        node_all_edges=node_all_edges, 
                                                                        topic_entity=topic_entity,
                                                                        history_path=history_path,
                                                                        subquestions_list_str=subquestions_list_str,
                                                                        budget=num_branch) # # {relation1: score1, relation2: score2, ...} 
                # pdb.set_trace()
    return relations2score_dict



def get_actions_do_reweight(node: treeNode, 
                             mcts_task,
                             graph: nx.Graph,
                             question: str,
                             topic_entity: str,
                             subquestions_list: list,
                             relations2score_dict: dict) -> defaultdict:
    """
    relations2score_dict: {relation1: score1, relation2: score2, ...} 保证能够找到邻接节点
    node: 当前的扩展的节点node
    return: {(h1, ri):{ti: scorei}, ...} h1 是当前node的text
    """

    relation_entity2score_dict = defaultdict(dict) # {(hi, ri):{ti: scorei}, (h1, r1):{t1: score}, ...}
    node_text = node.node_text
    history_path = node.history_path
    subquestions_list_str = convert_subquestion_list2str(subquestion_list=subquestions_list)
    relation_entity2score_dict = get_one_entity_all_adj_entity(node=node_text, 
                                                               graph=graph, 
                                                               relations2score_dict=relations2score_dict) # relation_entity2score_dict: {('James', 'government.'): [{'m.04': 0.944}, {'m.08_': 0.944}], ...}
    
    relation_entity2score_dict = mcts_task.get_entity_filter(node=node_text,
                                                             question=question,
                                                             subquestions_list_str=subquestions_list_str,
                                                             topic_entity=topic_entity,
                                                             graph=graph,
                                                             history_path=history_path,
                                                             subquestions_list=subquestions_list,
                                                             relation_entity2score_dict=relation_entity2score_dict) # 
    # pdb.set_trace()
    relation_entity2score_dict = mcts_task.get_reweight_value(node=node_text,
                                                              question=question,
                                                              subquestions_list_str=subquestions_list_str,
                                                              topic_entity=topic_entity,
                                                              history_path=history_path,
                                                              relation_entity2score_dict=relation_entity2score_dict)

    return relation_entity2score_dict # {(h1, ri):[{ti: scorei}, ...], (h1, r1):[{t1: score}, ...] ...}
    


def randomPolicy(node: treeNode, mcts_task):
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1
    if mcts_task.use_reflection == 'common':
        reflection = mcts_task.get_reflection(strs, cur_step)
    else:
        reflection = mcts_task.get_simple_reflection(strs, cur_step)
    node.update_reflection(reflection)
    if reflection == '<end>':
        print('This step has been resolved and does not require simulation.\n')
        return node.V
    for i in range(mcts_task.roll_forward_steps):
        next_steps = get_next_steps_roll(strs, cur_step, mcts_task)
        if not next_steps:
            break
        action = random.choice(next_steps)  # str
        strs = strs + action
        cur_step += 1
        value = mcts_task.get_step_value(strs)
        if value > max_V:
            max_V = value
        if mcts_task.use_reflection == 'common':
            cur_ref = mcts_task.get_reflection(strs, cur_step)
        else:
            cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
        if cur_ref == '<end>':
            break
    return max_V


def greedyPolicy(node: treeNode, mcts_task):
    # pdb.set_trace()
    max_V = mcts_task.low
    rollout_ans_flag = False
    rollout_wrong_ans_flag = False
    strs = node.y
    rollpath = []
    cur_step = node.depth + 1
    for i in range(mcts_task.roll_forward_steps):
        # rollout steps 我这里rollout只是为了找到最优的节点并得到v值
        one_step_roll_path = []
        # pdb.set_trace()
        actions = get_next_steps_roll(strs, cur_step, mcts_task)  # str_list 这里可以选用多次采样
        if not actions:
            print('-' * 40, "本次向下一次rollout没有结果", '-' * 40)
            break
    
        new_ys = [strs + action for action in actions] # 把之前的步骤和此步骤合并 history_info
        cur_step += 1
        values = [mcts_task.get_step_value(new_y) for new_y in new_ys] # critic
        idx = numpy.argmax(values)
        strs = new_ys[idx]
        value = values[idx]
        for roll_step, roll_value in zip(actions, values): #
            one_step_roll_path.append(roll_step)
            one_step_roll_path.append(roll_value)
        for action in actions:#这里加入对rollout的判断#
            if re.search('final answer is|answer is|answer has', action.lower()):
                flag = math_cheak(mcts_task.answer, action)
                # flag = mcts_task.eval_is_answer(action)
                answer = math_postprocess_v2(action)
                if answer != '' and answer.isdigit(): # 针对只有answer的 没有输出完全的情况
                    if flag:
                        rollout_ans_flag = True
                        value = 1.0
                        break
                    else:
                        rollout_wrong_ans_flag = True
                        value = value * 0.4
                        break
        # 每次都会判断本次rollout是否终止
        values[idx] = value
        if value > max_V:
            max_V = value
        if rollout_ans_flag == True: # 如果这次找到错误答案或者正确答案 终止
            rollpath.append(one_step_roll_path)
            break
        if rollout_wrong_ans_flag == True:
            rollpath.append(one_step_roll_path)
            break
        rollpath.append(one_step_roll_path)
    return max_V, rollpath, rollout_ans_flag



def executeRound(root: treeNode=None,
                 mcts_task=None, 
                 topic_entity: str='', 
                 graph: nx.Graph=None,
                 question: str=None, 
                 iter_num: int=0,
                 subquestions_list: list=None,
                 shuffle: bool=False,
                 shuffle_times: int=1):
    """
    execute a selection-expansion-simulation-backpropagation round
    """
    print('-' * 40)
    print('选择节点阶段\n')
    path_with_reward = []
    is_in_graph = True
    flag, node = selectNode(root, mcts_task) # 选择最好的node或者 就是当前node
    # pdb.set_trace()
    if flag: # 到terminal节点为真 则返回
        if mcts_task.sample_value == 'simple': # 如果无需全部采样完全 遇到terminal直接返回
            return True, is_in_graph, node, root, path_with_reward
        else: # 全部展开 获得所有的节点路径
            node.reflection = '<end>' # 如果
            path_with_reward.append((node.node_text, node.V))


    print('-' * 40)
    print('扩充阶段\n')
    if node.reflection == '<end>':
        print('跳过此阶段。\n')
    else:
        node = expand(node=node, 
                      mcts_task=mcts_task, 
                      graph=graph, 
                      question=question,
                      topic_entity=topic_entity,
                      iter=iter_num,
                      subquestions_list=subquestions_list,
                      shuffle=shuffle,
                      shuffle_times=shuffle_times) # node.children [action1:node1, action2:node2, action3:node3
    
    if iter_num == 0 and node.reflection == '<end>':
        ####### 此时是没有expand的 该节点是叶子节点 ########
        print('-' * 40)
        print('该节点不在子图当中 直接返回 生成下一条数据 \n')
        is_in_graph = False
        return False, is_in_graph, node, root, path_with_reward, 

    # else:
    #     print('-' * 40)
    #     print('模拟搜索阶段\n')
    #     roll_node = getBestChild(node, mcts_task) # 选了node下最优的字节点进行rollout 这里v值为0.5 roll_node.y 存储了step值
    #     if roll_node.reflection == '<end>':
    #         roll_node.numVisits += 1
        # pdb.set_trace()
        
        # greedy policy 测一下
        # else:
        #     best_V, rollpath, rollout_ans_flag = greedyPolicy(roll_node, mcts_task) if mcts_task.roll_policy == 'greedy' \
        #                     else randomPolicy(roll_node, mcts_task)
        #     for item in rollpath:
        #         roll_node.rollpath.append(item)
        #     if rollout_ans_flag == True:
        #         roll_node.vm_eval_ans_flag = 'correct'
        #     roll_node.rollout_ans_flag = rollout_ans_flag
        #     # path_with_reward[index].append(rollpath)
        #     roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
        #     roll_node.numVisits += 1

    print('-' * 40)
    print('反向传播阶段\n')
    back_propagate(node)
    root.trace_path()
    root.count_node()
    # pdb.set_trace()
    print('backup is over')
    print('-'*40, f'已有节点数量:{root.node_num}', f'最大深度:{root.maxdepth}')
    # 返回顺序需与调用处解구结构와 일치: node, is_in_graph, flag, root, path_with_reward_list, subquestions_list
    return node, is_in_graph, node, root, path_with_reward, subquestions_list



def MCTS_search(mcts_task):
    """
    
    """
    data = mcts_task.data
    question = data['question']
    if not question.endswith('?'):
        question += '?'
    qid = data['id']
    raw_graph = data['graph']
    topic_entity_list = data['q_entity']
    topic_entity = mcts_task.topic_entity
    # pdb.set_trace()
    topic_entity_str = '; '.join(topic_entity_list)
    answer_entity_list = data['a_entity']
    shuffle = mcts_task.shuffle
    shuffle_times = mcts_task.shuffle_times
    graph = build_graph(raw_graph)
    path_with_reward_list = [] #收集路径信息
    # pdb.set_trace()

    root = treeNode(node_text=topic_entity, history_path=topic_entity) #根节点
    subquestions_list = mcts_task.get_intension_decompose(question=question,
                                                          topic_entity=topic_entity) # 针对每个问题 进行意图分解
    # pdb.set_trace()
    # pdb.set_trace()
    if mcts_task.limit_type == 'time':
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        while time.time() < timeLimit:
            print(f'<开始新搜索轮次，目前总时间:{time.time() - time_start}>\n')
            flag, is_in_graph, node, root, path_with_reward = executeRound(root=root, 
                                                                topic_entity=topic_entity, 
                                                                mcts_task=mcts_task,
                                                                graph=graph,
                                                                question=question,
                                                                subquestions_list=subquestions_list,
                                                                shuffle=shuffle,
                                                                shuffle_times=shuffle_times)
            path_with_reward_list.append(path_with_reward)
            if flag:
                print('已找到解决方案！\n')
                return node, is_in_graph, flag, root, path_with_reward_list, subquestions_list
    else:
        time_start = time.time()
        for iter in range(mcts_task.iteration_limit):
            print(f'<开始新搜索轮次，目前总时间:{time.time() - time_start}>\n')
            print(f'<开始新搜索轮次，目前已完成轮次数:{iter}>\n')
            # pdb.set_trace()
            # pdb.set_trace()
            flag, is_in_graph, node, root, path_with_reward, subquestions_list = executeRound(root=root,
                                                                topic_entity=topic_entity,
                                                                mcts_task=mcts_task,
                                                                graph=graph,
                                                                question=question,
                                                                subquestions_list=subquestions_list,
                                                                iter_num=iter,
                                                                shuffle=shuffle,
                                                                shuffle_times=shuffle_times)
            if is_in_graph == False:
                return node, is_in_graph, flag, root, path_with_reward_list, subquestions_list
            path_with_reward_list.append(path_with_reward)
            all_time = time.time() - time_start
            
            if flag:
                print('已找到解决方案！\n')
                return node, is_in_graph, flag, root, path_with_reward_list, subquestions_list
            elif all_time > 3600:
                print('超过限制时间！\n')
                return node, is_in_graph, flag, root, path_with_reward_list, subquestions_list

    return node, is_in_graph, flag, root, path_with_reward_list, subquestions_list