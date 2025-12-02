import os
import argparse
from tqdm import tqdm
from utils.tools import *
from MCTSv2.base import*
from datetime import datetime
from models.inference_models import *
from tasks.prompts_v2 import *



MODEL_PATH = {'qwen7b':'/workspace/LLaMA-Factory/models/Qwen2___5-7B-Instruct',
              'llama3':'/workspace/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct',
              'llama3.1':'/workspace/LLaMA-Factory/models/Llama3-1-8B-Instruct',
              'qwen14b':'/workspace/LLaMA-Factory/models/Qwen2.5-14B-Instruct',
              'qwenqwq':'/workspace/LLaMA-Factory/models/QwQ-32B-Preview',
              'qwen32b':'/workspace/LLaMA-Factory/models/Qwen2.5-32B-Instruct'}

gte_model_path = '/workspace/LLaMA-Factory/models/gte_Qwen2-7B-instruct'
emb_model_path = '/workspace/LLaMA-Factory/models/text2vec-base-multilingual'
tog_model_path = '/workspace/LLaMA-Factory/models/msmarco-distilbert-base-tas-b'



def generate_answer(arguments,
                    question: str,
                    topic_entity_list: list,
                    subquestions_list: list,
                    topk_path_list: list,
                    io_system) -> list:
    subquestions_list_str = convert_subquestion_list2str(list(set(subquestions_list)))
    topic_entity_str = ', '.join(topic_entity_list)
    answer_list = []
    filtered_topk_path_list = []
    for item in topk_path_list:
        prediction, path, score = item[0], item[1], item[2]
        if prediction.startswith('m.') or prediction.startswith('g.') or path.split(' -> ')[0] == path.split(' -> ')[-1]:
            continue
        else:
            filtered_topk_path_list.append([prediction, path, score])
    history_info = ''
    prediction2score = {}
    
    
    path_str = ''
    candidate_ent = []
    for item in filtered_topk_path_list:
        prediction, path, score = item[0], item[1], item[2]
        path_str += path + ' \n'
        candidate_ent.append(prediction)
    
    #########################################################n选k模式###################################################################
    # prompt = answer_generate_promptv4.format(question=question,
    #                                          subquestions=subquestions_list_str,
    #                                          topic_entity=topic_entity_str,
    #                                          candidate_path=path_str,)

    # # response = io_system.get_local_response(query=prompt,
    # #                                         max_length=arguments.max_len,
    # #                                         max_new_tokens=arguments.max_new_tokens,
    # #                                         temperature=arguments.temperature,
    # #                                         do_sample=arguments.do_sample,
    # #                                         truncation=arguments.truncation)
    # response = io_system.get_api_response(query=prompt)
    # # pdb.set_trace()
    # k_entity_list = parse_n_of_k(response, candidate_ent)
    # if len(k_entity_list) != 0:
    #     answer_list = k_entity_list[:]
    # else:
    #     print(f'该问题{question}没有找到最好的答案 返回所有实体')
    #     answer_list = candidate_ent[:]
    #########################################################n选k模式###################################################################
    
    
    ################################################## 下面都是一条一条选 #######################################################
    for index, item in enumerate(filtered_topk_path_list):
        if len(item) != 3:
            print('路径长度解析错误')
            continue
        prediction, path, score = item[0], item[1], str(round(item[2], 3))
        print('\n', '==============================', '判断推理路径是否可靠', '==============================', '\n')
        print('propose_prompt: \n', '本问题是:  ', question,  '\n若干子问题:  ', subquestions_list_str, '\n主题实体是:', topic_entity_str, '\n待判断的推理路径:', path, '\n已有的推理步骤:\n' + history_info + f'候选推理路径的答案: {prediction}是否可能是最后答案:\n')
        # prompt = answer_generate_prompt.format(question=question,
        #                                         subquestions=subquestions_list_str,
        #                                         topic_entity=topic_entity_str,
        #                                         candidate_path=path,
        #                                         history_info=history_info)


        #########################################################逐一yes or no模式###################################################################
        prompt = answer_generate_promptv2.format(question=question,
                                                topic_entity=topic_entity_str,
                                                candidate_path=path,)

        # response = io_system.get_local_response(query=prompt,
        #                                         max_length=arguments.max_len,
        #                                         max_new_tokens=arguments.max_new_tokens,
        #                                         temperature=arguments.temperature,
        #                                         do_sample=arguments.do_sample,
        #                                         truncation=arguments.truncation)
        response = io_system.get_api_response(query=prompt)
        decision = parse_decision(response)
        if decision == True:
            history_info += path + '\n'
            answer_list.append(prediction)   
        #########################################################逐一yes or no模式###################################################################
                
            
        #########################################################打分模式###################################################################
        # prompt = answer_generate_promptv3.format(question=question,
        #                                          subquestions=subquestions_list_str,
        #                                          topic_entity=topic_entity_str,
        #                                          candidate_path=path,)

        # # response = io_system.get_local_response(query=prompt,
        # #                                         max_length=arguments.max_len,
        # #                                         max_new_tokens=arguments.max_new_tokens,
        # #                                         temperature=arguments.temperature,
        # #                                         do_sample=arguments.do_sample,
        # #                                         truncation=arguments.truncation)
        
        # response = io_system.get_api_response(query=prompt)

        # score = parse_path_score(response)
        # prediction2score[prediction] = score

        # filtered_keys = [key for key, value in prediction2score.items() if value > 0.9]
        # answer_list = filtered_keys
        # if not filtered_keys:
        #     sorted_keys = sorted(prediction2score.keys(), key=lambda x: prediction2score[x], reverse=True)[:4]
        #     answer_list = sorted_keys   
        #########################################################打分模式###################################################################
                 
    return list(set(answer_list))
    

def run(arguments:argparse.ArgumentParser):
    print('-'*30, 'Begin inference', '-'*30, '\n')
    
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    tokenizer, model = None, None
    if arguments.use_local_method:
        if arguments.use_vllm: # 这里是为了多次采样 投票决定最有可能的关系
            from models.vllm_models import load_vLLM_model
            if arguments.propose_method == 'qwen':
                model_path = MODEL_PATH['qwen']
                tokenizer, model = load_vLLM_model(model_path)
                print("*"*50, "加载qwen模型！", "*"*50)
            if arguments.propose_method == 'llama':
                model_path = MODEL_PATH['llama']
                tokenizer, model = load_vLLM_model(model_path)
                print("*"*50, "加载llama模型！", "*"*50)
        else: #不使用vllm 
            if 'qwen' in arguments.propose_method:
                model_path = MODEL_PATH[arguments.propose_method]
                from models.inference_models import get_inference_model_qwen
                tokenizer, model = get_inference_model_qwen(model_path)
                print("*"*50, "加载qwen模型！", "*"*50)              
            elif 'llama' in arguments.propose_method:
                model_path = MODEL_PATH[arguments.propose_method]
                from models.inference_models import get_inference_model_llama
                tokenizer, model = get_inference_model_llama(model_path)
                print("*"*50, "加载llama模型！", "*"*50)
            
            # pdb.set_trace()
    else: # 使用api 不需要加载模型
        pass
    io_system = IO_System(args=arguments, tokenizer=tokenizer, model=model)
    
    for datapath in arguments.datapath_list:
        try:
            dataset = read_json(datapath)
            data_len = len(dataset)
            # pdb.set_trace()
        except Exception as e:
            print(f'File must be standardized json!\nError type:{e}\n')
            return
        assert data_len > 0, "Data list is empty!\n"
        
        acc_list = []
        hit_list = []
        f1_list = []
        precission_list = []
        recall_list = []
        output_dir = '/'.join(datapath.split('/')[:-1]) + '/'
        file_path = output_dir+f'{arguments.propose_method}_eval_result.jsonl'

        # 读取现有数据
        existing_ids = []
        if os.path.exists(file_path):
            filtered_data = []
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line.strip())  # 解析每行的 JSON 对象
                    if data["hit"] != 0 or data['final_ans'] == False:  # 过滤掉 hit=0 的条目
                        filtered_data.append(data)        
            with open(file_path, "w", encoding="utf-8") as file:
                for data in filtered_data:
                    file.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    try:
                        existing_data = json.loads(line)
                        existing_ids.append(existing_data['qid'])
                    except:
                        raise ValueError("Error in line: ", line)
        
        
        with open(file_path, 'a') as f:
            for i in tqdm(range(data_len)):
                print(f'Begin to solve the problem {i+1}...\n')
                data = dataset[i]
                qid = data['qid']
                # pdb.set_trace()
                if len(existing_ids) != 0 and qid in existing_ids:
                    continue
                # question = data['question'] + '?'
                question = data['question']
                topic_entity_list = data['topic_entity_list']
                answer_entity_list = data['gt_answer']
                is_legal = data['is_legal']
                # final_ans = data['final_ans']
                if is_legal:
                    subquestions_list = data['subquestions']
                    # pdb.set_trace()
                    final_ans = data['final_ans']
                    topk_path_list = data['topk_path'][:10] + data['all_gt_path']
                    topk_path_list = [list(item) for item in set(tuple(row) for row in topk_path_list)]
                    
                    prediction_list = generate_answer(arguments,
                                                    question=question,
                                                    topic_entity_list=topic_entity_list,
                                                    subquestions_list=subquestions_list,
                                                    topk_path_list=topk_path_list,
                                                    io_system=io_system)
                    
                    f1_score, precission_score, recall_score = eval_f1(prediction_list=prediction_list, answer_list=answer_entity_list)
                    f1_list.append(f1_score), precission_list.append(precission_score), recall_list.append(recall_score)
                    acc = eval_acc(prediction_list=prediction_list, answer_list=answer_entity_list)
                    hit = eval_hit(prediction_list=prediction_list, answer_list=answer_entity_list)
                    acc_list.append(acc)
                    hit_list.append(hit)
                    f.write(json.dumps({'qid': qid, 'final_ans': final_ans, 'question': question, 'gt_answer': answer_entity_list, 'prediction': prediction_list, 'acc': acc, 'hit': hit, 'f1':f1_score, 'precission':precission_score, 'recall': recall_score}) + '\n')
                    f.flush()
                    
                else:
                    continue
        result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list)) + " F1: " + str(sum(f1_list) * 100 / len(f1_list)) + " Precision: " + str(sum(precission_list) * 100 / len(precission_list)) + " Recall: " + str(sum(recall_list) * 100 / len(recall_list))
        print(result_str)
        with open(output_dir + f'{arguments.propose_method}_eval_result.txt', 'a') as f:
            f.write(result_str)

def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--datapath_list', type=list, default=['/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1-320_alltree/shortcut.json'])
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'llama3', 'llama3.1', 'qwen7b', 'qwenapi', '4o-mini', 'qwen14b', 'qwenqwq', 'qwen32b', 'deepseekv3'], default='deepseekv3')
    base_args.add_argument('--use_local_method', type=bool, default=True)
    base_args.add_argument('--truncation', type=bool, default=True)
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--max_len', type=int, default=4000)
    base_args.add_argument('--max_new_tokens', type=int, default=256)
    base_args.add_argument('--do_sample', type=bool, default=False)
    base_args.add_argument('--use_vllm', default=False, action="store_true")

    # base_args.add_argument('--model_id', type=int, default=4)
    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    # datapath_list = ['/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1-320_alltree/shortcut.json']
    # # datapath_list = ['/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1-320_alltree/shortcut.json',
    # #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_320-640_alltree/shortcut.json',
    # #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_640-960_alltree/shortcut.json',
    # #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_960-1280_alltree/shortcut.json',
    # #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1280-1600_alltree/shortcut.json']
    datapath_list = ['/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_450-900_alltree/shortcut.json']
    # datapath_list = ['/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1-450_alltree/shortcut.json',
    #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_450-900_alltree/shortcut.json',
    #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_900-1350_alltree/shortcut.json',
    #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1350-1800_alltree/shortcut.json',
    #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1800-2250_alltree/shortcut.json',
    #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_2250-2700_alltree/shortcut.json',
    #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_2700-3150_alltree/shortcut.json',
    #                  '/workspace/xxxx/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_3150-3600_alltree/shortcut.json']
    args.datapath_list = datapath_list
    run(args)
    # set_model(args.model_id)