import json
import pdb
import networkx as nx
import pyarrow.parquet as pq
from datasets import load_dataset
from collections import defaultdict
import os
import sys
sys.path.append('..')
from tasks.prompts import *
import os.path
from utils.graph_utils import *
from sentence_transformers import util
import numpy as np
import torch

def read_json(source):
    json_list = []
    if not os.path.exists(source):
        return json_list
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


def prepare_dataset(input_dir):
    if 'cwq' in input_dir:
        with open(input_dir, encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        qid_string = 'ID'
    elif 'webqsp' in input_dir:
        with open(input_dir, encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
        qid_string = 'QuestionId'
    elif 'grailqa' in input_dir:
        with open(input_dir, encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        qid_string = 'qid'
    elif 'simpleqa' in input_dir:
        with open(input_dir, encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
        qid_string = None
    elif 'webquestions' in input_dir:
        with open(input_dir, encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        qid_string = None  
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string, qid_string


def parse_freebase_data(data: dict,
                        task_name: str,
                        question_string:str,
                        q_string: str,
                        index: int):
    """
    解析数据 返回freebase版本下的数据
    Args:
        data (dict): _description_
        task_name (str): _description_
        question_string (str): _description_
        q_string (str): _description_
        index (int): _description_

    Returns:
        topic_entity_list: [{mid:str_name}, {mid:str_name}]
    """
    answer_entity_list = []
    topic_entity_list = []
    question = data[question_string]
    if q_string is not None:
        qid = data[q_string]
    else:
        qid = None
    if task_name != 'webquestions':
        topic_entity_dict = data['topic_entity']
        for mid, entity_str in topic_entity_dict.items():
            topic_entity_list.append({mid: entity_str})
    else:
        topic_entity_list = None
        
    if task_name == 'webqsp':
        parses = data["Parses"]
        for parse in parses:
            for answer in parse['Answers']:
                if answer['EntityName'] == None:
                    answer_entity_list.append(answer['AnswerArgument'])
                else:
                    answer_entity_list.append(answer['EntityName'])
    
    elif task_name == 'cwq':
        if 'answers' in data:
            answers = data["answers"]
        else:
            answers = data["answer"]
        for answer in answers:
            answer_entity_list.append(answer)

    elif task_name == 'graliqa':
        answers = data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_entity_list.append(answer['entity_name']) # entity_str
            else:
                answer_entity_list.append(answer['answer_argument']) # mid

    elif task_name == 'webquestions':
        answer_entity_list = data['a_entity']   
    
    return question, qid, topic_entity_list, answer_entity_list



def dump_json(source, datas):
    with open(source, 'w', encoding='utf-8') as f:
        for item in datas:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def read_data(input_dir, mode='test'):
    input_files = os.listdir(input_dir)
    # pdb.set_trace()
    files = {'train':[], 'test':[], 'validation': []}
    for file in input_files:
        if file.split('.')[-1] == 'parquet':
            if file.split('-')[0] == 'train':
                files['train'].append(os.path.join(input_dir, file))
            if file.split('-')[0] == 'test':
                files['test'].append(os.path.join(input_dir, file))
            if file.split('-')[0] == 'validation':
                files['validation'].append(os.path.join(input_dir, file))
        else:
            pass
    files = {
    key: sorted(value, key=lambda x: int(x.split('/')[-1].split('-')[1]))  # 提取文件名中的数字部分并排序
    for key, value in files.items()
    }
    dataset = load_dataset('parquet', data_files=files[mode])
    dataset = dataset['train']
    return dataset

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def convert2candiate_path(history_path: str,
                          relation_entity2score_dict: defaultdict) -> str:
    """
    relation_entity2score_dict: {(hi, ri):{ti: scorei}, (h1, r1):{t1: score}, ...}
    """
    current_path = ''
    current_path_list = []
    for (head_entity, relation) in relation_entity2score_dict.keys():
        tail_entity2score = relation_entity2score_dict[(head_entity, relation)]
        for tail_entity, _ in tail_entity2score.items():
            current_path = history_path + ' -> ' + relation + ' -> ' + tail_entity + ' \n'
            current_path_list.append(current_path)
    return current_path_list
    
def split_list_into_chunks(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_top_k_items(dictionary, k):
    # 使用 sorted 函数对字典按值排序，返回一个包含 (key, value) 元组的列表
    sorted_items = sorted(dictionary.items(), key=lambda item: sum(item[1])*len(item[1]), reverse=True)
    if len(dictionary) >= k:
        # 选择前 k 个键值对
        top_k_items = sorted_items[:k]
        # 将结果转换为字典
        result_dict = {key: sum(value)/len(value) for key, value in top_k_items}
    else:
        top_k_items = sorted_items
        result_dict = {key: sum(value)/len(value) for key, value in top_k_items}
    return result_dict

def convert_list2str(relation_list):
    relation_str = ''
    for relation in relation_list[:-1]:
        relation_str += relation + '; '
    relation_str += relation_list[-1]
    return relation_str


def parse_edges_and_score(xml_response_list: list, input_relations_text: str)->dict:
    """
    xml_response_list: ['<count> 5 </count>', '<choice> location.country.languages_spoken </choice>', ...]
    return: {relation:score, ...} 合法的relation and score
    """
    choice_list, score_list = [], []
    edges_score_dict = {}
    for raw_data in xml_response_list:
        choice = parse_xml(raw_data, key='choice')
        score = parse_xml(raw_data, key='score')
        if choice:
            choice_list.append(choice)
            print(choice)
        if score:
            score_list.append(score)
    # pdb.set_trace()
    if len(choice_list) != len(score_list):
        length = min(len(choice_list), len(score_list))
        choice_list, score_list = choice_list[:length], score_list[:length]
    for relation, value in zip(choice_list, score_list):
        if relation not in input_relations_text: # 检查关系的合法性
            continue
        else:
            try:
                value = float(value)
            except Exception as e:
                print(f'不合法的分数！{e}')
                continue
        edges_score_dict[relation] = value
    return edges_score_dict


def parse_edges_and_score_with_list(xml_response_list: list, input_relations_text: str)->dict:
    """
    xml_response_list: ['<count> 5 </count>', '<choice> location.country.languages_spoken </choice>', ...]
    return: {relation:score, ...} 合法的relation and score
    """
    choice_list, score_list = [], []
    edges_score_dict_list = defaultdict(list)
    edges_score_dict = {}
    for raw_data in xml_response_list:
        choice = parse_xml(raw_data, key='choice')
        score = parse_xml(raw_data, key='score')
        if choice:
            choice_list.append(choice)
            print(choice)
        if score:
            score_list.append(score)
    # pdb.set_trace()
    if len(choice_list) != len(score_list):
        length = min(len(choice_list), len(score_list))
        choice_list, score_list = choice_list[:length], score_list[:length]
    for relation, value in zip(choice_list, score_list):
        if relation not in input_relations_text: # 检查关系的合法性
            continue
        else:
            try:
                value = float(value)
            except Exception as e:
                print(f'不合法的分数！{e}')
                continue
        edges_score_dict_list[relation].append(value)
    for relation in edges_score_dict_list.keys():
        edges_score_dict[relation] = sum(edges_score_dict_list[relation]) / len(edges_score_dict_list[relation])
    return edges_score_dict


def parse_rank_edges_and_score(xml_response_list: list, input_relations_text: str)->dict:
    """
    xml_response_list: ['<rank> 1 </rank>', '<choice> location.country.languages_spoken </choice>', ...]
    return: {relation:score, ...} 合法的relation and score
    """
    choice_list, rank_list = [], []
    edges_score_dict = {}
    for raw_data in xml_response_list:
        rank = parse_xml(raw_data, key='rank')
        choice = parse_xml(raw_data, key='choice')
        if choice:
            choice_list.append(choice)
            print(choice)
        if rank:
            score_list.append(rank)
    if len(choice_list) != len(score_list):
        length = min(len(choice_list), len(score_list))
        choice_list, score_list = choice_list[:length], score_list[:length]
    for relation, value in zip(choice_list, score_list):
        if relation not in input_relations_text: # 检查关系的合法性
            continue
        else:
            try:
                value = float(value)
            except Exception as e:
                print(f'不合法的分数！{e}')
                continue
        edges_score_dict[relation] = value
    return edges_score_dict


def parse_entity_score(xml_response_list: list):
    """
    xml_response_list: []
    返回 小数的分数
    Args:
        xml_response_list (list): _description_
    """
    entity_list, score_list = [], []
    for raw_data in xml_response_list:
        entity = parse_xml(raw_data, key='entity')
        score = parse_xml(raw_data, key='score')
        if entity:
            entity_list.append(entity)
        if score:
            score_list.append(score)
    if len(entity_list) != len(score_list):
        length = min(len(entity_list), len(score_list))
        choice_list, score_list = choice_list[:length], score_list[:length]
    try:
        score = float(score_list[0])
    except Exception as e:
        print(f'不合法的分数！{e}')
    return score

def is_legal_data(topic_entity_list: list, answer_entity_list: list, graph: list):
    """ 判断该数据是否合法

    Args:
        topic_entity_list (list): _description_
        answer_entity_list (list): _description_
        graph (list): _description_

    Returns:
        _type_: _description_
    """
    topic_entity_flag = False
    answer_entity_flag = False
    for triple in graph:
        for topic_entity in topic_entity_list:
            if topic_entity in triple:
                topic_entity_flag = True
        for answer_entity in answer_entity_list:
            if answer_entity in triple:
                answer_entity_flag = True
    return topic_entity_flag and answer_entity_flag


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


def parse_relation_entity2score(xml_response_list: list,
                                node_text: str,
                                relation_entity2score_dict: defaultdict,
                                candidate_path: str):
    """
    xml_response_list: ['<count> 5 </count>', '<choice> location.country.languages_spoken </choice>', ...]
    relation_entity2score_dict: {(hi, ri):{ti: scorei}, (h1, r1):{t1: score}, ...}
    retrun: {(relation, entity) : score, ...}
    """
    reweight_relation_entity2score_dict = defaultdict(dict)
    legal_rel_tail = []
    for (head, relation) in relation_entity2score_dict.keys():
        tail2score = relation_entity2score_dict[(head, relation)]
        tail_entity = list(tail2score.keys())[0]
        legal_rel_tail.append((relation, tail_entity))
    
    path_list, score_list = [], []
    for raw_data in xml_response_list:
        path = parse_xml(raw_data, key='path')
        score = parse_xml(raw_data, key='score')
        reason = parse_xml(raw_data, key='reason')
        if path:
            path_list.append(path)
        if score:
            score_list.append(score)
        if reason:
            print(path, '\n')
            print(reason)
    # pdb.set_trace()
    if len(path_list) != len(score_list):
        length = min(len(path_list), len(score_list))
        path_list, score_list = path_list[:length], score_list[:length]
    try:
        score = float(score_list[0])
    except Exception as e:
        print('reweight 出错 设置score: 0.1')
        score = 0.1

    if candidate_path.find(node_text) != -1: # 判断解析路径的合法性
        strat_idx = candidate_path.find(node_text) + len(node_text) + len(' ->')
        candidate_path = candidate_path[strat_idx:]
        split_list = candidate_path.strip().split('->')
        
        if len(split_list) == 2:
            relation, tail_entity = split_list[0].strip(), split_list[1].strip()
            # pdb.set_trace()
            if (relation, tail_entity) in legal_rel_tail: # 检验合法性
                reweight_relation_entity2score_dict[(relation, tail_entity)] = score

    return reweight_relation_entity2score_dict


def parse_relation_entity2score_list(xml_response_list: list,
                                     node_text: str,
                                     relation_entity2score_dict: defaultdict,
                                     candidate_path: str):
    """
    xml_response_list: ['<count> 5 </count>', '<choice> location.country.languages_spoken </choice>', ...]
    relation_entity2score_dict: {(hi, ri):{ti: scorei}, (h1, r1):{t1: score}, ...} 这是上一步存的所有的邻接尾实体
    candidate_path: 
    retrun: {(relation, entity) : score, ...}
    """
    reweight_relation_entity2score_dict = defaultdict(dict)
    edges_score_dict_list = defaultdict(list)
    edges_score_dict = {}
    legal_rel_tail = []
    for (head, relation) in relation_entity2score_dict.keys():
        tail2score = relation_entity2score_dict[(head, relation)]
        tail_entity = list(tail2score.keys())[0]
        legal_rel_tail.append((relation, tail_entity))
    
    path_list, score_list = [], []
    for raw_data in xml_response_list:
        path = parse_xml(raw_data, key='path')
        score = parse_xml(raw_data, key='score')
        reason = parse_xml(raw_data, key='reason')
        if path:
            path_list.append(path)
        if score:
            try:
                score = float(score)
                score_list.append(score)
            except:
                continue
            
        if reason:
            print(path, '\n')
            print(reason)

    if len(path_list) != len(score_list):
        length = min(len(path_list), len(score_list))
        path_list, score_list = path_list[:length], score_list[:length]
    # pdb.set_trace()
    score = sum(score_list) / len(score_list)

    if candidate_path.find(node_text) != -1: # 判断解析路径的合法性
        strat_idx = candidate_path.find(node_text) + len(node_text) + len(' ->')
        cur_path = candidate_path[strat_idx:]
        split_list = cur_path.strip().split('->')
        # pdb.set_trace()
        if len(split_list) == 2:
            relation, tail_entity = split_list[0].strip(), split_list[1].strip()
            # pdb.set_trace()
            if (relation, tail_entity) in legal_rel_tail: # 检验合法性
                # pdb.set_trace()
                reweight_relation_entity2score_dict[(relation, tail_entity)] = score

    return reweight_relation_entity2score_dict



def convert_subquestion_list2str(subquestion_list: list)-> str:
    subquestion_str = '[' + ', '.join(f"'{item}'" for item in subquestion_list) + ']'
    return subquestion_str


def parse_subquestions_list(xml_response_list:list):
    subquestions_list = []
    for raw_data in xml_response_list:
        subquestion = parse_xml(raw_data, key='subquestion')
        if subquestion:
            subquestions_list.append(subquestion)
    if len(subquestions_list) > 3:
        subquestions_list = subquestions_list[:3]
    return subquestions_list


def retrieve_top_docs(query_list: list, 
                      docs_list: list, 
                      model, 
                      topk=40):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query_list (str): The input query list .
    - docs_list (list of str): The list of documents to search from.
    - model The emb model to use.
    - topk (int): The number of top documents to return.

    Returns:
    - list of str: A list of the topn documents.
    """
    weight_vetor = [0.5] + [0.5 / (len(query_list)-1)] * (len(query_list)-1) # 
    weight_vetor = torch.tensor(weight_vetor).reshape(len(query_list), 1) # n_question, 1
    
    query_embeddings = model.encode(query_list) # n_question * dim
    doc_embeddings = model.encode(docs_list) # n_relation or n_entity * dim

    similarity_scores = util.dot_score(query_embeddings, doc_embeddings) # tensor: n_question * (n_relation or n_entity)
    weighted_similarity_scores = torch.matmul(similarity_scores.T, weight_vetor) # n_entity, n_question * n_question, 1 = n_entity or n_relation * 1
    sorted_indices = torch.argsort(weighted_similarity_scores, dim=0, descending=True) # 
    
    sorted_docs_list = [docs_list[i.item()] for i in sorted_indices]
    topk_docs_list = sorted_docs_list[:topk]
    
    return topk_docs_list


def retrieve_top_docs_score(query_list: list, 
                            docs_list: list, 
                            model, 
                            topk=5):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query_list (str): The input query list .
    - docs_list (list of str): The list of documents to search from.
    - model The emb model to use.
    - topk (int): The number of top documents to return.

    Returns:
    - list of str: A list of the topn documents.
    """
    weight_vetor = [0.5] + [0.5 / (len(query_list)-1)] * (len(query_list)-1) # 
    weight_vetor = torch.tensor(weight_vetor).reshape(len(query_list), 1) # n_question, 1
    # pdb.set_trace()
    query_embeddings = model.encode(query_list) # n_question * dim
    doc_embeddings = model.encode(docs_list) # n_relation or n_entity * dim

    similarity_scores = util.dot_score(query_embeddings, doc_embeddings) # tensor: n_question * (n_relation or n_entity)
    weighted_similarity_scores = torch.matmul(similarity_scores.T, weight_vetor) # n_entity, n_question * n_question, 1 = n_entity or n_relation * 1
    # sorted_indices = torch.argsort(weighted_similarity_scores, dim=0, descending=True) # 
    weighted_similarity_scores = weighted_similarity_scores.squeeze()
    
    if weighted_similarity_scores.shape[0] > topk:
        topk_scores, _ = torch.topk(weighted_similarity_scores, k=topk)
        mean_topk_scores = topk_scores.mean()
    else:
        mean_topk_scores = weighted_similarity_scores.mean()
    
    return mean_topk_scores.cpu().tolist()



def parse_decision(xml_response):
    """
    返回yes or no信号
    Args:
        xml_response: []
    """
    output_list, reason_list = [], []
    for raw_data in xml_response:
        output = parse_xml(raw_data, key='response')
        if output:
            output_list.append(output)
        reason = parse_xml(raw_data, key='reason')
        if reason:
            reason_list.append(reason)
    if len(output_list) == 0:
        print('\n', '没判断出来！')
        return True
    output = output_list[0]
    if output.lower() in ['yes', 'no']:
        if len(reason_list) != 0:
            print(f'\n{output}, 判断原因是: ', reason_list[0])
        if output.lower() == 'yes':
            return True
        elif output.lower() == 'no':
            return False
    else:
        print('\n', '判断不出来 继续扩展')
        return True



def parse_path_score(xml_response_list: list,):
    """
    xml_response_list: ['<count> 5 </count>', '<choice> location.country.languages_spoken </choice>', ...]
    """    
    path_list, score_list, reason_list = [], [], []
    for raw_data in xml_response_list:
        path = parse_xml(raw_data, key='path')
        score = parse_xml(raw_data, key='score')
        reason = parse_xml(raw_data, key='reason')
        if path:
            path_list.append(path)
        if score:
            score_list.append(score)
        if reason:
            reason_list.append(reason)
            print(path, '\n')
            print(reason)
    # pdb.set_trace()
    if len(score_list) > 0:
        try:
            score = float(score_list[0])
            if len(reason_list) > 0:
                print(f'打分是{str(score)}, 原因是: ', reason_list[0])
        except Exception as e:
            print('reweight 出错 设置score: 0.1')
            score = 0.1
    else:
        score =0.1
    return score


def parse_n_of_k(xml_response:list, candidate_entity_list: list):
    path_list, reason_list, output_list = [], [], []
    for raw_data in xml_response:
        path = parse_xml(raw_data, key='path')
        if path:
            path_list.append(path)
            print(path, '\n')
        reason = parse_xml(raw_data, key='reason')
        if reason:
            reason_list.append(reason)
            print(reason, '\n')
    for path in path_list:
        candidate_entity = path.split(' -> ')[-1]
        if candidate_entity in candidate_entity_list:
            output_list.append(candidate_entity)
    return output_list
        
    

############################################################### evaluation #################################################################
def eval_hit(prediction_list, answer_list):
    """
    计算hit值
    """
    for a in answer_list:
        for p in prediction_list:
            if a == p:
                return 1
    return 0

def eval_acc(prediction_list, answer_list):
    """
    计算acc值
    """
    matched = 0.
    for a in answer_list:
        for p in prediction_list:
            if a == p:
                matched += 1
    return matched / len(answer_list)

def eval_f1(prediction_list, answer_list):
    """
    计算f1值
    """
    if len(prediction_list) == 0:
        return 0, 0, 0
    matched = 0
    for a in answer_list:
        for p in prediction_list:
            if p == a:
                matched += 1
    precission = matched / len(prediction_list)
    recall = matched / len(answer_list)
    if precission + recall == 0:
        return 0, precission, recall
    else:
        return 2 * precission * recall / (precission + recall), precission, recall
############################################################### evaluation #################################################################