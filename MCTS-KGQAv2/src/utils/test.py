import pyarrow.parquet as pq
from datasets import load_dataset
import sys
import pdb
import json
import datasets
# sys.path.append('.')
import concurrent.futures
from graph_utils import *
import networkx as nx # type: ignore
from tqdm import tqdm


def parse_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=5, question=None, id=None):
    '''
    Get all simple paths connecting question and answer entities within given hop
    '''
    # Select paths
    path_save = {}
    path_len_save = {}
    path_dict = defaultdict(lambda: defaultdict(list))
    path_save['id'] = id
    path_save['question'] = question
    path_len_save['id'] = id
    path_len_save['question'] = question
    # pdb.set_trace()
    path_len_num_dict = defaultdict(lambda: defaultdict(int))
    # path_dict[q_entity[0] + ' -> ' +  a_entity[0]]['-1'] = ['None'] 
    # path_len_num_dict[q_entity[0] + ' -> ' +  a_entity[0]]['-1'] = ['None']
    paths = []
    for h in q_entity:
        if h not in graph:
            # path_dict[h + ' -> ' +  a_entity[0]] = ['None']
            # path_len_num_dict[h + ' -> ' +  a_entity[0]]['-1'] = ['None']
            continue
        for t in a_entity:
            if t not in graph:
                # path_dict[h + ' -> ' +  t] = ['None']
                print(f"{t} is not in graph")
                # path_len_num_dict[h + ' -> ' +  t]['-1'] = ['None']
                continue
            try:
                for plan in range(1, 6):
                    paths = []
                    for p in nx.all_simple_edge_paths(graph, h, t, cutoff=plan):
                        paths.append(p)
                    for p in paths:
                        # pdb.set_trace()
                        path = [graph[e[0]][e[1]]['relation'] for e in p]
                        if len(path) != plan:
                            continue
                        else:
                            path_dict[h + ' -> ' +  t][str(plan)].append(path)
                    path_dict[h + ' -> ' +  t][str(plan)] = [list(item) for item in set(tuple(sublist) for sublist in path_dict[h + ' -> ' +  t][str(plan)])]
                    path_len_num_dict[h + ' -> ' +  t][str(plan)] = str(len(path_dict[h + ' -> ' +  t][str(plan)]))
                    # pdb.set_trace()
                    if plan >=4 :
                        # pdb.set_trace()
                        if len(path_dict[h + ' -> ' +  t][str(plan)]) >= 50:
                            path_dict[h + ' -> ' +  t][str(plan)] = random.sample(path_dict[h + ' -> ' +  t][str(plan)], 50)
                    # path_dict[h + '_' +  t][str(plan)] = list(set(path_dict[h + '_' +  t][str(plan)]))
            except:
                path_dict[h + ' -> ' +  t]['None Path'] = ['None']
                path_len_num_dict[h + ' -> ' +  t]['None Path'] = '0'
                print("fail to pasre")
    # # Add relation to paths
    # result_paths = []
    # for p in paths:
    #     plan = str(len(p))
    #     result_paths.append([(e[0], graph[e[0]][e[1]]['relation'], e[1]) for e in p])
    #     path_dict[]
    # pdb.set_trace()
    path_save['path'] = path_dict
    path_len_save['path'] = path_len_num_dict
    return path_save, path_len_save


def dump_json(path, datas):
    with open(path, 'w', encoding='utf-8') as f:
        for data in datas:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.write('\n')


def parse_data(datapath):
    dataset = load_dataset('parquet', data_files=datapath)
    # pdb.set_trace()
    # dataset = dataset['train'][:2]
    # dataset = datasets.Dataset.from_dict(dataset)
    dataset = dataset['train']
    data_list = []
    data_list2 = []
    # dataset = dataset[900:]
    # dataset = datasets.Dataset.from_dict(dataset)
    for data in tqdm(dataset):
        graph_data = data['graph']
        q_entity_list = data['q_entity']
        a_entity_list = data['a_entity']
        id = data['id']
        question = data['question']
        graph = build_graph(graph_data)
        # pdb.set_trace()
        path_dict, path_len_num_dict = parse_simple_paths(q_entity_list, a_entity_list, graph, question=question, id=id)
        data_list.append(path_dict)
        data_list2.append(path_len_num_dict)
    dump_json('/workspace/longxiao/KGQA/MCTS-KGQA/data/cwq_test_statisitc31.json', data_list)
    dump_json('/workspace/longxiao/KGQA/MCTS-KGQA/data/cwq_test_statisitc32.json', data_list2)


def process_single_data(sample):
    # pdb.set_trace()
    # dataset = dataset['train'][:2]
    # dataset = datasets.Dataset.from_dict(dataset)
    # dataset = dataset['train']
    # data_list = []
    # data_list2 = []
    # for data in tqdm(datas):
    # pdb.set_trace()
    graph_data = sample['graph']
    q_entity_list = sample['q_entity']
    a_entity_list = sample['a_entity']
    # sample['path_len_num'] = {}
    graph = build_graph(graph_data)
    path_dict, path_len_num_dict = get_simple_paths(q_entity_list, a_entity_list, graph)
    # data_list.append(path_dict)
    # data_list2.append(path_len_num_dict)
    # pdb.set_trace()
    sample['graph'] = ['none']
    sample['choices'] = ['none']
    # sample['path_len_num'] = []
    # sample['path_len_num'] = defaultdict(lambda: defaultdict(int))
    # pdb.set_trace()
    list1 = []
    for key, value in path_len_num_dict.items():
        if key and value:
            a = ': '.join([key, '//'.join([item[0] + ':' + item[1] for item in list(value.items())])])
            list1.append(a)
    # list1 = [item for sublist in list1 for item in sublist]
    sample['path_len_num'] = list1
    # pdb.set_trace()
    # sample['path_len_num'].append(list1)
    # sample['path_len_num'] = path_len_num_dict
    # pdb.set_trace()
    # sample['path_info'] = path_dict
    # data['path_num'] = data_list2
    # pdb.set_trace()
    return sample


def filter_function(example):
    example_dict = example['path_len_num']
    for key, value in example_dict.items():
        if not value:
            del example_dict[key]
        else:
            continue
    return example


def processed_data(dataset, max_workers=10):
    # 使用多线程处理数据集
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # 使用 tqdm 显示进度条
    #     results = list(tqdm(executor.map(process_single_data, dataset), total=len(dataset)))
    # return results
    processed_dataset = dataset.map(
        process_single_data,
        num_proc=max_workers,
    )
    # filtered_dataset = processed_dataset.filter(filter_function)
    return processed_dataset



if __name__ == "__main__":

    webqsp_path = ['/workspace/longxiao/KGQA/MCTS-KGQA/data/webqsp/test-00000-of-00002-9ee8d68f7d951e1f.parquet',
                    '/workspace/longxiao/KGQA/MCTS-KGQA/data/webqsp/test-00001-of-00002-773a7b8213e159f5.parquet']



    cwq_path = ['/workspace/longxiao/KGQA/MCTS-KGQA/data/cwq/test-00000-of-00003-e62a559c5d2b56ca.parquet', 
                '/workspace/longxiao/KGQA/MCTS-KGQA/data/cwq/test-00001-of-00003-2fa9a898639e7d1e.parquet',
                '/workspace/longxiao/KGQA/MCTS-KGQA/data/cwq/test-00002-of-00003-c659cd388440c4ae.parquet']
    
    parse_data(cwq_path)
    
    # dataset = load_dataset('parquet', data_files=webqsp_path)
    # dataset = datasets.Dataset.from_dict(dataset['train'][:10])
    # # dataset = dataset['train']
    # # # pdb.set_trace()
    # processed_dataset = processed_data(dataset, max_workers=1)
    # # pdb.set_trace()
    # processed_dataset = processed_dataset.remove_columns(['graph', 'choices', ])
    # dump_json('/workspace/longxiao/KGQA/MCTS-KGQA/data/webqsp_test_statisitc_3.json', processed_dataset)
    
    # print('final')
    # question = data['question'][0]
    # raw_data = data['graph'][0]
    # # print(question, raw_data)
    # graph = build_graph(raw_data)
    # a_entity, q_entity = data[0]['a_entity'], data[0]['q_entity']
    # path_dict = get_simple_paths(q_entity, a_entity, graph)