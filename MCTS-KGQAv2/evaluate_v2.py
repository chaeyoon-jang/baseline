import os
import pathlib
from MCTSv2.task import MCTS_Task
import argparse
from tqdm import tqdm
import networks as nx
import pyarrow.parquet as pq
from datasets import load_dataset
from utils.tools import *
from MCTSv2.base import*
from datetime import datetime
from models.inference_models import *
import traceback
from sentence_transformers import SentenceTransformer
from similarities import *

MODEL_PATH = {'qwen7b':'Qwen/Qwen2.5-7B-Instruct',
              'llama3':'meta-llama/Meta-Llama-3-8B-Instruct',
              'llama3.1':'meta-llama/Llama3-1-8B-Instruct',
              'qwen14b':'Qwen/Qwen2.5-14B-Instruct',
              'qwenqwq':'Qwen/QwQ-32B-Preview',
              'qwen32b':'Qwen/Qwen2.5-32B-Instruct'}

gte_model_path = '/workspace/LLaMA-Factory/models/gte_Qwen2-7B-instruct'
emb_model_path = '/workspace/LLaMA-Factory/models/text2vec-base-multilingual'
tog_model_path = '/workspace/LLaMA-Factory/models/msmarco-distilbert-base-tas-b'

def run(arguments:argparse.ArgumentParser):
    print('-'*30, 'Begin inference', '-'*30, '\n')
    dataset_source = ''
    try:
        if arguments.dataset_path:
            dataset = load_jsonl_kgqa_dataset(arguments.dataset_path)
            dataset_source = arguments.dataset_path
        else:
            data_dir = arguments.data_dir or f'/workspace/xxxxx/KGQA/MCTS-KGQA/data/KGData/{arguments.task_name}/'
            dataset = read_data(data_dir, mode='test')
            dataset_source = data_dir
        data_len = len(dataset)
    except Exception as e:
        print(f'데이터셋 로드 실패. Error type:{e}\n')
        return

    assert data_len > 0, "데이터 목록이 비어있습니다!\n"
    start_idx = max(arguments.start_idx, 0)
    end_idx = arguments.end_idx if arguments.end_idx != -1 else data_len
    end_idx = min(end_idx, data_len)
    if start_idx >= end_idx:
        print(f'처리할 샘플 없음 (start_idx={start_idx}, end_idx={end_idx}).')
        return

    print(f'{dataset_source}에서 {end_idx - start_idx} / {data_len} 샘플 로드됨.')
    # assert 'content' in data_list[0].keys() and 'answer' in data_list[0].keys(), "Key error, Make sure json object contain correct keys!\n"
    output_list = []
    correct_count = 0
    path_list = []
    tree_list = []
    now = datetime.now() # 현재 시간 및 분 가져오기
    current_day, current_month, current_hour, current_minute, current_second = now.day, now.month, now.hour, now.minute, now.second
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # gte_model = SentenceTransformer(gte_model_path)
    # text2vec_model = BertSimilarity(model_name_or_path=emb_model_path)
    # sentence_bert = SentenceTransformer(tog_model_path)
    
    # if arguments.emb_model == 'gte':
    #     emb_model = gte_model.to(torch.device("cpu"))
    # elif arguments.emb_model == 'text2vec':
    #     emb_model = text2vec_model.to(torch.device("cpu"))
    # elif arguments.emb_model == 'sentence_bert':
    #     emb_model = sentence_bert.to(torch.device("cpu"))
    
    tokenizer, model = None, None
    if arguments.use_local_method:
        if arguments.use_vllm: # 여러 샘플링을 위해 가장 가능성 높은 관계를 투표로 결정
            from models.vllm_models import load_vLLM_model
            if arguments.propose_method == 'qwen':
                model_path = MODEL_PATH['qwen']
                tokenizer, model = load_vLLM_model(model_path)
                print("*"*50, "qwen 모델 로드!", "*"*50)
            if arguments.propose_method == 'llama':
                model_path = MODEL_PATH['llama']
                tokenizer, model = load_vLLM_model(model_path)
                print("*"*50, "llama 모델 로드!", "*"*50)
        else: # vllm 미사용 
            if 'qwen' in arguments.propose_method:
                model_path = MODEL_PATH[arguments.propose_method]
                from models.inference_models import get_inference_model_qwen
                tokenizer, model = get_inference_model_qwen(model_path)
                print("*"*50, "qwen 모델 로드!", "*"*50)              
            elif 'llama' in arguments.propose_method:
                model_path = MODEL_PATH[arguments.propose_method]
                from models.inference_models import get_inference_model_llama
                tokenizer, model = get_inference_model_llama(model_path)
                print("*"*50, "llama 모델 로드!", "*"*50)
    else: # API 사용, 모델 로드 불필요
        pass
    # poor_case = ["WebQTest-12", "WebQTest-20", "WebQTest-39", "WebQTest-44", "WebQTest-58", "WebQTest-62", 'WebQTest-77', 'WebQTest-115', 'WebQTest-180', 'WebQTest-182', 'WebQTest-221', 'WebQTest-231', 'WebQTest-257', 'WebQTest-261', 'WebQTest-277', 'WebQTest-367', 'WebQTest-375', 'WebQTest-421', 'WebQTest-432', 'WebQTest-450', 'WebQTest-472', 'WebQTest-523', 'WebQTest-535', 'WebQTest-549', 'WebQTest-564', 'WebQTest-590', 'WebQTest-608', 'WebQTest-609', 'WebQTest-656', 'WebQTest-671', 'WebQTest-676', 'WebQTest-689', 'WebQTest-696', 'WebQTest-708', 'WebQTest-728', 'WebQTest-734', 'WebQTest-759', 'WebQTest-778', 'WebQTest-785', 'WebQTest-836', 'WebQTest-868', 'WebQTest-873', 'WebQTest-884', 'WebQTest-918', 'WebQTest-936', 'WebQTest-941', 'WebQTest-943', 'WebQTest-944', 'WebQTest-1052', 'WebQTest-1118', 'WebQTest-1143', 'WebQTest-1145', 'WebQTest-1154', 'WebQTest-1179', 'WebQTest-1200', 'WebQTest-1203', 'WebQTest-1240', 'WebQTest-1252', 'WebQTest-1271', 'WebQTest-1279', 'WebQTest-1296', 'WebQTest-1350', 'WebQTest-1388', 'WebQTest-1436', 'WebQTest-1443', 'WebQTest-1468', 'WebQTest-1477', 'WebQTest-1478', 'WebQTest-1480', 'WebQTest-1523', 'WebQTest-1539', 'WebQTest-1544', 'WebQTest-1547', 'WebQTest-1554', 'WebQTest-1558', 'WebQTest-1563', 'WebQTest-1568', 'WebQTest-1597', 'WebQTest-1622', 'WebQTest-1707', 'WebQTest-1759', 'WebQTest-1774', 'WebQTest-1808', 'WebQTest-1811', 'WebQTest-1853', 'WebQTest-1884', 'WebQTest-1938', 'WebQTest-1955', 'WebQTest-1960', 'WebQTest-1969', 'WebQTest-1975', 'WebQTest-1996', 'WebQTest-2013', 'WebQTest-2019']
    poor_case = ['WebQTest-1200', 'WebQTest-1203', 'WebQTest-1240', 'WebQTest-1252', 'WebQTest-1271', 'WebQTest-1279', 'WebQTest-1296', 'WebQTest-1350', 'WebQTest-1388', 'WebQTest-1436', 'WebQTest-1443', 'WebQTest-1468', 'WebQTest-1477', 'WebQTest-1478', 'WebQTest-1480', 'WebQTest-1523', 'WebQTest-1539', 'WebQTest-1544', 'WebQTest-1547', 'WebQTest-1554', 'WebQTest-1558', 'WebQTest-1563', 'WebQTest-1568', 'WebQTest-1597', 'WebQTest-1622', 'WebQTest-1707', 'WebQTest-1759', 'WebQTest-1774', 'WebQTest-1808', 'WebQTest-1811', 'WebQTest-1853', 'WebQTest-1884', 'WebQTest-1938', 'WebQTest-1955', 'WebQTest-1960', 'WebQTest-1969', 'WebQTest-1975', 'WebQTest-1996', 'WebQTest-2013', 'WebQTest-2019']
    io_system = IO_System(args=arguments, tokenizer=tokenizer, model=model)
    # pdb.set_trace()
    for i in tqdm(range(start_idx, end_idx), total=end_idx - start_idx):
        try:
            # 문제 해결
            print(f'문제 {i+1} 해결 시작...\n')
            data = dataset[i]
            question = data.get('question', '')
            topic_entity_list = data.get('q_entity', [])
            answer_entity_list = data.get('a_entity', [])
            qid = data.get('id', f'sample_{i}')
            if not topic_entity_list:
                print(f'Skip {qid}: 빈 주제 엔티티 목록.')
                continue
            ################################################# 나쁜 케이스 #####################################################
            # if qid not in poor_case:
            #     continue
            # else:
                ################################################# 일반 케이스 ###################################################
            output_tree_json = defaultdict(lambda: defaultdict(list))
            output_tree_json = {'qid': qid, 'topic_entity_list':topic_entity_list, 'question': question, 'answer': data['a_entity'], 'is_legal':True}
            if 'graph' not in data or not data['graph']:
                print(f"*" * 40 + f' 질문: {question} 그래프 정보 누락, qid: {qid}')
                output_tree_json['is_legal'] = False
                tree_list.append(output_tree_json)
                continue

            if not is_legal_data(topic_entity_list=topic_entity_list, answer_entity_list=answer_entity_list, graph=data['graph']): # 그래프에서 찾을 수 없는 잘못된 데이터 제외
                print('*' * 40, f'질문: {question}은(는) 비정상, qid: {qid}','*' * 40)
                output_tree_json['is_legal'] = False
                tree_list.append(output_tree_json)
                continue
            for topic_entity in topic_entity_list:
                print(f'Begin to solve the problem: {question}, \n topic entity: {topic_entity}...\n')
                output_tree_json[topic_entity] = {}
                
                if arguments.mode == 'mcts':
                    Task = MCTS_Task(data=data,
                                    topic_entity=topic_entity,
                                    # emb_model=emb_model 임베딩 모델
                                    io_system=io_system, 
                                    propose_method=arguments.propose_method, 
                                    value_method=arguments.value_method,
                                    use_generator=arguments.use_generator, 
                                    end_gate=arguments.end_gate,
                                    roll_policy=arguments.roll_policy, 
                                    roll_branch=arguments.roll_branch, 
                                    num_plan_branch=arguments.num_plan_branch,
                                    num_branch=arguments.num_branch,
                                    sample_value=arguments.sample_value, 
                                    roll_forward_steps=arguments.roll_forward_steps, 
                                    time_limit=arguments.time_limit,
                                    iteration_limit=arguments.iteration_limit, 
                                    exploration_constant=arguments.exploration_constant, 
                                    alpha=arguments.alpha, 
                                    inf=arguments.inf,
                                    temperature=arguments.temperature, 
                                    use_reflection=arguments.use_reflection, 
                                    max_tokens=arguments.max_tokens, 
                                    max_new_tokens=arguments.max_new_tokens, 
                                    max_length=arguments.max_len, 
                                    try_num=arguments.try_num, 
                                    min_iteration_limit=arguments.min_iteration_limit, 
                                    max_child_num=arguments.max_child_num,
                                    low=arguments.low, 
                                    high=arguments.high, 
                                    limited_depth=arguments.limited_depth,
                                    use_vllm=arguments.use_vllm,
                                    shuffle=arguments.shuffle,
                                    shuffle_times=arguments.shuffle_times,
                                    use_rank_prompt=arguments.use_rank_prompt)
                    # pdb.set_trace()
                    is_in_graph, finish, root, path_with_reward, subquestions_list = Task.run()
                    if is_in_graph == False:
                        question = data['question']
                        print('*' * 40, f'이 데이터의 주제 엔티티 {topic_entity}은(는) 부분 그래프에 없음, 문제: {question}')
                        continue
                root.trace_path()
                root.count_node()
                treeNode.reset_class_variable()
                
                output_tree_json[topic_entity]['steps'] = root.tree_list
                output_tree_json[topic_entity]['node_num'] = root.node_num
                output_tree_json[topic_entity]['maxdepth'] = root.maxdepth
                output_tree_json[topic_entity]['subquestions'] = subquestions_list
            tree_list.append(output_tree_json)
                # if arguments.visualize:
                #     visualize(root, Task, arguments.task_name, arguments.file, i + 1)

            print(f'문제 {i+1}의 트리 완성.\n')
            base_dir = os.getcwd()
            output_dir = pathlib.Path(f'{base_dir}/outputs/{arguments.task_name}/{Task.mode}/{Task.propose_method}')
            output_file = f'{base_dir}/outputs/{arguments.task_name}/{Task.mode}/{Task.propose_method}/{Task.propose_method}-{arguments.shuffle_times}-{arguments.num_plan_branch}-{arguments.num_branch}_{current_month}_{current_day}_{current_hour}_{current_minute}_{current_second}_alltree.json'
            pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
            dump_json(output_file, tree_list)
            ################################################# 일반 케이스 ###################################################
        except Exception as e:
            print(f"이 데이터 생성 실패, 다음 데이터 진행!\nError type:{e}\n")
            print(traceback.format_exc())
            continue

    print('_' * 60)
    # 정확도

    if arguments.evaluate:
        print(f'테스트 정확도: {correct_count / data_len}\n')
        print(f'정답 문제 수: {correct_count}\n전체 질문 수: {data_len}\n')
    print('_' * 60)



def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='cwq')
    base_args.add_argument('--emb_model', type=str, choices=['gte', 'sentence_bert', 'text2vec'], default='gte')
    base_args.add_argument('--use_local_method', action=argparse.BooleanOptionalAction, default=True)
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'llama3', 'llama3.1', 'qwen7b', 'qwenapi', '4o-mini', 'qwen14b', 'qwenqwq', 'qwen32b'], default='qwen')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'llama', 'qwen', 'qwen', '4o-mini'], default='qwen')
    base_args.add_argument('--do_sample', action=argparse.BooleanOptionalAction, default=False)
    base_args.add_argument('--num_plan_branch', type=int, default=6)
    base_args.add_argument('--num_branch', type=int, default=3)
    base_args.add_argument('--truncation', action=argparse.BooleanOptionalAction, default=True)
    base_args.add_argument('--use_generator', action=argparse.BooleanOptionalAction, default=False)
    base_args.add_argument('--limited_depth', type=int, default=5)
    base_args.add_argument('--sample_value', type=str, choices=['simple', 'full'], default='full')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='mcts')
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--time_limit', type=int, default=None)
    base_args.add_argument('--iteration_limit', type=int, default=2)
    base_args.add_argument('--roll_policy', type=str, choices=['random', 'greedy'], default='greedy')
    base_args.add_argument('--exploration_constant', type=float, default=0.4)
    base_args.add_argument('--roll_forward_steps', type=int, default=2)
    base_args.add_argument('--end_gate', type=float, default=0.95)  # End threshold
    base_args.add_argument('--roll_branch', type=int, default=1)
    base_args.add_argument('--max_len', type=int, default=30000)
    base_args.add_argument('--max_tokens', type=int, default=16000)
    base_args.add_argument('--try_num', type=int, default=5)
    base_args.add_argument('--max_new_tokens', type=int, default=256)
    base_args.add_argument('--inf', type=float, default=0.8)
    base_args.add_argument('--evaluate', action=argparse.BooleanOptionalAction, default=False)  # Whether to evaluate (empty means no evaluation)
    base_args.add_argument('--alpha', type=float, default=0.5)
    base_args.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=False)  # visualization
    base_args.add_argument('--use_reflection', type=str, choices=['simple', 'common'], default='simple')  # Use reflective mode
    base_args.add_argument('--low', type=float, default=0)
    base_args.add_argument('--high', type=float, default=1)
    base_args.add_argument('--select_branch', type=int, default=2)
    base_args.add_argument('--max_depth', type=int, default=8)
    base_args.add_argument('--select_method', type=str, choices=['greedy', 'sample'], default='greedy')
    base_args.add_argument('--min_iteration_limit', type=int, default=86)
    base_args.add_argument('--max_child_num', type=int, default=9)
    base_args.add_argument('--use_vllm', default=False, action="store_true")
    base_args.add_argument('--shuffle', default=False, action='store_true')
    base_args.add_argument('--shuffle_times', type=int, default=1)
    base_args.add_argument('--use_rank_prompt', default=False, action='store_true')
    base_args.add_argument('--dataset_path', type=str, default=None, help='Path to a JSONL dataset (e.g., FINAL/test.json).')
    base_args.add_argument('--data_dir', type=str, default=None, help='Directory containing parquet files when not using dataset_path.')
    base_args.add_argument('--start_idx', type=int, default=0, help='Start index (inclusive) for samples to process.')
    base_args.add_argument('--end_idx', type=int, default=-1, help='End index (exclusive) for samples to process. Use -1 to run to the end.')
    # base_args.add_argument('--model_id', type=int, default=4)
    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    run(args)
    # set_model(args.model_id)
