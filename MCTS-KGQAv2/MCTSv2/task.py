import random
import pdb
import sys
import numpy as np
sys.path.append('..')
sys.path.append('.')
from tasks.search_v2 import SearchTask
from MCTSv2.base import treeNode
from collections import defaultdict
from MCTSv2.mcts import MCTS_search
from utils.tools import *
from models.inference_models import *
from similarities import *
from sentence_transformers import SentenceTransformer


class MCTS_Task(SearchTask):
    def __init__(self, 
                 data,
                 topic_entity: str, 
                 io_system: IO_System,
                 propose_method: str ='qwen', 
                 value_method: str ='qwen', 
                 num_branch: int=3, 
                 end_gate: float=0.9, 
                 roll_policy: str='greedy',
                 roll_branch: int=1, 
                 roll_forward_steps: int=5, 
                 time_limit=None, 
                 iteration_limit=None, 
                 exploration_constant=0.7,
                 alpha=0.5, 
                 inf=1.0, 
                 temperature=0.7, 
                 max_tokens=2048, 
                 seed=170, 
                 max_length=2048, 
                 truncation=True, 
                 do_sample=True, 
                 max_new_tokens=256, 
                 use_reflection='simple', 
                 low=0, 
                 high=1, 
                 num_plan_branch=6, 
                 try_num=4, 
                 min_iteration_limit=86, 
                 max_child_num=7,
                 sample_value='full', 
                 answer=None, 
                 use_generator: bool=False,
                 limited_depth: int=None,
                 use_vllm: bool=False,
                 shuffle: bool=False,
                 shuffle_times: int=1,
                 use_rank_prompt: bool=False,
                 use_emb_filter_adj_entity: bool=True,
                 use_llm_filter_adj_entity: bool=False,
                 emb_model=None):
        
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        if emb_model is not None:
            self.emb_model = emb_model
        self.propose_method = propose_method
        self.io_system = io_system
        self.topic_entity = topic_entity
        self.mode = 'mcts'
        self.use_generator = use_generator
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.num_branch = num_branch
        self.low = low
        self.high = high
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1
        self.sample_value = sample_value
        self.try_num = try_num
        self.min_iteration_limit = min_iteration_limit
        self.num_plan_branch = num_plan_branch
        self.answer_pool = []
        self.max_child_num = max_child_num
        self.limited_depth = limited_depth
        self.end_gate = end_gate
        self.use_reflection = use_reflection
        self.answer = answer
        self.use_vllm = use_vllm
        self.shuffle = shuffle
        self.shuffle_times = shuffle_times
        self.use_rank_prompt = use_rank_prompt
        self.use_emb_filter_adj_entity = use_emb_filter_adj_entity
        self.use_llm_filter_adj_entity = use_llm_filter_adj_entity
        assert (self.use_emb_filter_adj_entity ^ self.use_llm_filter_adj_entity), "must choose one method to filter" #
        
    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def set_limit_type(self):
        if self.time_limit is not None:
            if self.iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            self.limit_type = 'time'
        else:
            if self.iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = 'iterations'
    
    def filter_and_score_edges(self, 
                               question: str, 
                               node_all_edges:list, 
                               history_path:str='', 
                               topic_entity: str=None,
                               subquestions_list_str: str=None,
                               budget: int=8) -> dict:
        """
        给定一定长度的边列表 返回筛选的关系以及分数 dict 筛选的数量 budget
        input: node_all_edges: [r1, r2, ...]
        output: {r1: score1, r2: score2, ..} 这里的数量不是固定的 有可能有多个
        subquestions_list_str: ['Identify the countries where the main spoken language is Brahui', 'Find the president of each country', 'Determine the president from 1980']
        budget: 是相对固定的筛选数量 对于超过10个的关系集合 budget为2, 关系小于10个的集合 budget为3 
        """
        
        input_relations_text = convert_list2str(node_all_edges)

        if self.use_rank_prompt:
            input = self.rank_and_score_prompt_wrap(question=question,
                                                    input_relations_text=input_relations_text,
                                                    history_path=history_path,
                                                    topic_entity=topic_entity)
            
            if self.use_vllm == False:
                response = self.io_system.get_local_response(query=input,
                                                            max_length=self.max_length,
                                                            max_new_tokens=self.max_new_tokens,
                                                            temperature=self.temperature,
                                                            do_sample=self.do_sample,
                                                            truncation=self.truncation)
                
                edges_with_score_dict = parse_rank_edges_and_score(xml_response_list=response, 
                                                              input_relations_text=input_relations_text) # response: ['<count> 5 </count>', '<choice> location.country.languages_spoken </choice>', '<reason> This relation is directly relevant as it provides the languages spoken in Jamaica, which is the key information needed to answer the question. </reason>', '<score> 0.9 </score>',]
                # pdb.set_trace()
            else:
                # pdb.set_trace()
                response = self.io_system.generate_with_vLLM_model(query=input,
                                                                   n=5)
                edges_with_score_dict = parse_rank_edges_and_score(xml_response_list=response, 
                                                                   input_relations_text=input_relations_text)

        else: # 不用 rank 的prompt
            input = self.filter_and_score_prompt_wrap(question=question,
                                                      subquestions=subquestions_list_str, 
                                                      input_relations_text=input_relations_text, 
                                                      history_path=history_path, 
                                                      topic_entity=topic_entity,
                                                      budget=budget)
            
            if self.use_vllm == False:
                response = self.io_system.get_local_response(query=input,
                                                            max_length=self.max_length,
                                                            max_new_tokens=self.max_new_tokens,
                                                            temperature=self.temperature,
                                                            do_sample=self.do_sample,
                                                            truncation=self.truncation)
                
                edges_with_score_dict = parse_edges_and_score(xml_response_list=response, 
                                                              input_relations_text=input_relations_text) # response: ['<count> 5 </count>', '<choice> location.country.languages_spoken </choice>', '<reason> This relation is directly relevant as it provides the languages spoken in Jamaica, which is the key information needed to answer the question. </reason>', '<score> 0.9 </score>',]
                # pdb.set_trace()
            else:
                # pdb.set_trace()
                response = self.io_system.generate_with_vLLM_model(query=input,
                                                                   n=5)
                edges_with_score_dict = parse_edges_and_score_with_list(xml_response_list=response, 
                                                                        input_relations_text=input_relations_text) # 这里是进行 平均分 返回的还是
                # pdb.set_trace()
            # pdb.set_trace()
        return edges_with_score_dict


    def get_entity_filter(self,
                          node: str,
                          question: str,
                          topic_entity: str,
                          history_path: str,
                          graph: nx.Graph,
                          relation_entity2score_dict: defaultdict,
                          subquestions_list: list,
                          subquestions_list_str: str=None, ) -> defaultdict:
        """
        input: relation_entity2score_dict: {('James', 'government.'): [{'m.04': 0.944}, {'m.08_': 0.944}], ...}
        node: 前一跳的实体 好像暂时没啥用 先放着
        history_path: 到当前实体前一跳的路径
        筛选出最有可能的一个尾实体
        这个过程 需要用到尾实体周围的信息
        返回 relation_entity2score_filter_dict {('James', 'government.'): {'m.04': 0.944}, ...} 最好的一个candidate entity
        """
        relation_entity2score_filter_dict = defaultdict(dict)
        questions_list = [question] + subquestions_list # [原问题, 子问题1, 子问题2, 子问题3] 
        for (head, relation) in relation_entity2score_dict.keys(): # 
            tail2score_list = relation_entity2score_dict[(head, relation)] # [{'m.04': 0.944}, {'m.08_': 0.944}, ...] relation_score 是相同的
            tail_num = len(tail2score_list)
            print('*'*40, '关系: ', relation, '*'*4, '连接的待筛选候选实体数量为: ', tail_num, '*'*40)
            
            if tail_num == 1: # 如果尾实体只有一个 则不需要进行进一步的过滤
                relation_entity2score_filter_dict[(head, relation)] = tail2score_list[0] # 选的就是最好的那个实体
            elif tail_num <= 10 and tail_num > 0:
                best_score, target_relation_score = -1, 0
                for tail2score in tail2score_list: # 这里需要对每一个尾实体进行打分 如果尾实体很多的话一个个打分 如果超过一定数量 考虑到效率可以换一种方式打分 完全基于emb
                    tail, relation_score = list(tail2score.items())[0]
                    cur_history_path = history_path + ' -> ' + relation + ' -> ' + tail #
                    neighbor_relation_entity_list, neighbor_triple_text_list = get_entity_adj_info(graph=graph, 
                                                                                                   entity=tail,) # 返回的是一个实体的邻接信息 

                    if self.use_emb_filter_adj_entity: # 这一步是返回尾实体周围最有可能帮助回答问题的K个邻居
                        weight_vetor = [0.5] + [0.5 / (len(questions_list)-1)] * (len(questions_list)-1)
                        weight_vetor = torch.tensor(weight_vetor).reshape(len(questions_list), 1) # q_l, 1
                        
                        emb_model = BertSimilarity(model_name_or_path="/workspace/LLaMA-Factory/models/text2vec-base-multilingual")
                        
                        similarity_scores = emb_model.similarity(questions_list, neighbor_triple_text_list) # shape: q_len * nentity
                        weighted_similarity_scores = torch.matmul(similarity_scores.T, weight_vetor) # nentity, q_len * q_len, 1
                        sorted_indices = torch.argsort(weighted_similarity_scores, dim=0, descending=True)
                        sorted_neighbor_relation_entity_list = [neighbor_relation_entity_list[i.item()] for i in sorted_indices]
                        sorted_neighbor_triple_text_list = [neighbor_triple_text_list[i.item()] for i in sorted_indices]
                        if len(neighbor_relation_entity_list) > 10:
                            sorted_neighbor_relation_entity_list = sorted_neighbor_relation_entity_list[:10] # 这是通过筛选出来的尾实体周围最相关的K个邻居 [(h, r, t), (h1, r1, t1)]
                            sorted_neighbor_triple_text_list = sorted_neighbor_triple_text_list[:10] # 这是通过筛选出来的尾实体周围最相关的K个邻居 ['h r is t', 'h1 r1 is t2', ...]
                        
                        # similarity_scores = similarity_scores.numpy()

                    elif self.use_llm_filter_adj_entity:
                        pass
                    
                    sorted_neighbor_relation_entity_list_str = '\n'.join(str(triplet) for triplet in sorted_neighbor_relation_entity_list)
                    sorted_neighbor_triple_text_list_str = '\n'.join(sorted_neighbor_triple_text_list)

                    input = self.score_candidate_entity_prompt_wrap(question=question,
                                                                    subquestions=subquestions_list_str,
                                                                    topic_entity=topic_entity,
                                                                    pre_relation=relation,
                                                                    candidate_entity=tail,
                                                                    history_path=cur_history_path,
                                                                    neighbor_info=sorted_neighbor_triple_text_list_str)
                    
                    # pdb.set_trace()
                    response = self.io_system.get_local_response(query=input,
                                                                 max_length=self.max_length,
                                                                 max_new_tokens=self.max_new_tokens,
                                                                 temperature=self.temperature,
                                                                 do_sample=self.do_sample,
                                                                 truncation=self.truncation)
                    
                    tail_entity_score = parse_entity_score(xml_response_list=response)
                    if tail_entity_score > best_score:
                        best_tail = tail
                        best_score = tail_entity_score
                        target_relation_score = relation_score
                
                relation_entity2score_filter_dict[(head, relation)] = {best_tail: target_relation_score}
            
            # elif tail_num > 10: # 如果尾实体超过200个的话 换一种方法进行过滤 但是最后反正是 返回 relation_entity2score_filter_dict 选择最好的一个candidate entity
            #     total_tail_score_dict = {}
            #     # emb_model = SentenceTransformer('/workspace/LLaMA-Factory/models/gte_Qwen2-7B-instruct', device="cpu")
            #     emb_model = SentenceTransformer('/workspace/LLaMA-Factory/models/msmarco-distilbert-base-tas-b')
            #     pdb.set_trace()
            #     for tail2score in tail2score_list:
            #         tail, relation_score = list(tail2score.items())[0]
            #         tail_neighbor_info = []
            #         tail_neighbor_info.append(tail) # [tail, ]
            #         _, neighbor_triple_text_list = get_entity_adj_info(graph=graph, 
            #                                                            entity=tail,) # 返回的是一个实体的邻接信息 
            #         tail_neighbor_info += neighbor_triple_text_list # [tail, tail邻接信息1, ...]
            #         tail_score = retrieve_top_docs_score(query_list=questions_list,
            #                                              docs_list=tail_neighbor_info,
            #                                              model=emb_model,
            #                                              topk=5)
            #         total_tail_score_dict[tail] = tail_score
            #     filter_total_tail_score_dict = dict(sorted(total_tail_score_dict.items(), key=lambda item: item[1], reverse=True)[:10]) # 这里做一个初步筛选 筛选前十个候选实体
            #     filter_tail2score_list = [{tail:relation_score} for tail in filter_total_tail_score_dict.keys()]
            #     pdb.set_trace()  

            #     best_score, target_relation_score = -1, 0
            #     for tail2score in filter_tail2score_list: # 这里需要对每一个尾实体进行打分 如果尾实体很多的话一个个打分 如果超过一定数量 考虑到效率可以换一种方式打分 完全基于emb
            #         tail, relation_score = list(tail2score.items())[0]
            #         cur_history_path = history_path + ' -> ' + relation + ' -> ' + tail #
            #         neighbor_relation_entity_list, neighbor_triple_text_list = get_entity_adj_info(graph=graph, 
            #                                                                                        entity=tail,) # 返回的是一个实体的邻接信息 

            #         if self.use_emb_filter_adj_entity: # 这一步是返回尾实体周围最有可能帮助回答问题的K个邻居
            #             weight_vetor = [0.5] + [0.5 / (len(questions_list)-1)] * (len(questions_list)-1)
            #             weight_vetor = torch.tensor(weight_vetor).reshape(len(questions_list), 1) # q_l, 1
                        
            #             emb_model = BertSimilarity(model_name_or_path="/workspace/LLaMA-Factory/models/text2vec-base-multilingual")
                        
            #             similarity_scores = emb_model.similarity(questions_list, neighbor_triple_text_list) # shape: q_len * nentity
            #             weighted_similarity_scores = torch.matmul(similarity_scores.T, weight_vetor) # nentity, q_len * q_len, 1
            #             sorted_indices = torch.argsort(weighted_similarity_scores, dim=0)
            #             sorted_neighbor_relation_entity_list = [neighbor_relation_entity_list[i.item()] for i in sorted_indices]
            #             sorted_neighbor_triple_text_list = [neighbor_triple_text_list[i.item()] for i in sorted_indices]
            #             if len(neighbor_relation_entity_list) > 10:
            #                 sorted_neighbor_relation_entity_list = sorted_neighbor_relation_entity_list[:10] # 这是通过筛选出来的尾实体周围最相关的K个邻居 [(h, r, t), (h1, r1, t1)]
            #                 sorted_neighbor_triple_text_list = sorted_neighbor_triple_text_list[:10] # 这是通过筛选出来的尾实体周围最相关的K个邻居 ['h r is t', 'h1 r1 is t2', ...]
                        
            #             # similarity_scores = similarity_scores.numpy()

            #         elif self.use_llm_filter_adj_entity:
            #             pass
                    
            #         sorted_neighbor_relation_entity_list_str = '\n'.join(str(triplet) for triplet in sorted_neighbor_relation_entity_list)
            #         sorted_neighbor_triple_text_list_str = '\n'.join(sorted_neighbor_triple_text_list)

            #         input = self.score_candidate_entity_prompt_wrap(question=question,
            #                                                         subquestions=subquestions_list_str,
            #                                                         topic_entity=topic_entity,
            #                                                         pre_relation=relation,
            #                                                         candidate_entity=tail,
            #                                                         history_path=cur_history_path,
            #                                                         neighbor_info=sorted_neighbor_triple_text_list_str)
                    
            #         # pdb.set_trace()
            #         response = self.io_system.get_local_response(query=input,
            #                                                      max_length=self.max_length,
            #                                                      max_new_tokens=self.max_new_tokens,
            #                                                      temperature=self.temperature,
            #                                                      do_sample=self.do_sample,
            #                                                      truncation=self.truncation)
                    
            #         tail_entity_score = parse_entity_score(xml_response_list=response)
            #         if tail_entity_score > best_score:
            #             best_tail = tail
            #             best_score = tail_entity_score
            #             target_relation_score = relation_score
                
            #     relation_entity2score_filter_dict[(head, relation)] = {best_tail: target_relation_score}
                
            elif tail_num > 10: # 如果尾实体超过200个的话 换一种方法进行过滤 但是最后反正是 返回 relation_entity2score_filter_dict 选择最好的一个candidate entity
                total_tail_score_dict = {}
                # emb_model = SentenceTransformer('/workspace/LLaMA-Factory/models/gte_Qwen2-7B-instruct', device="cpu")
                emb_model = SentenceTransformer('/workspace/LLaMA-Factory/models/msmarco-distilbert-base-tas-b')
                # pdb.set_trace()
                
                for tail2score in tail2score_list:
                    tail, relation_score = list(tail2score.items())[0]
                    tail_neighbor_info = []
                    tail_neighbor_info.append(tail) # [tail, ]
                    _, neighbor_triple_text_list = get_entity_adj_info(graph=graph, 
                                                                       entity=tail,) # 返回的是一个实体的邻接信息 
                    tail_neighbor_info += neighbor_triple_text_list # [tail, tail邻接信息1, ...]
                    tail_score = retrieve_top_docs_score(query_list=questions_list,
                                                         docs_list=tail_neighbor_info,
                                                         model=emb_model,
                                                         topk=5)
                    total_tail_score_dict[tail] = tail_score
                filter_total_tail_score_dict = dict(sorted(total_tail_score_dict.items(), key=lambda item: item[1], reverse=True)[:10]) # 这里做一个初步筛选 筛选前十个候选实体
                
                
                filter_tail2score_list = [{tail:relation_score} for tail in filter_total_tail_score_dict.keys()]
                # pdb.set_trace()  

                best_score, target_relation_score = -1, 0
                for tail2score in filter_tail2score_list: # 这里需要对每一个尾实体进行打分 如果尾实体很多的话一个个打分 如果超过一定数量 考虑到效率可以换一种方式打分 完全基于emb
                    tail, relation_score = list(tail2score.items())[0]
                    cur_history_path = history_path + ' -> ' + relation + ' -> ' + tail #
                    neighbor_relation_entity_list, neighbor_triple_text_list = get_entity_adj_info(graph=graph, 
                                                                                                   entity=tail,) # 返回的是一个实体的邻接信息 

                    if self.use_emb_filter_adj_entity: # 这一步是返回尾实体周围最有可能帮助回答问题的K个邻居
                        weight_vetor = [0.5] + [0.5 / (len(questions_list)-1)] * (len(questions_list)-1)
                        weight_vetor = torch.tensor(weight_vetor).reshape(len(questions_list), 1) # q_l, 1
                        
                        emb_model = BertSimilarity(model_name_or_path="/workspace/LLaMA-Factory/models/text2vec-base-multilingual")
                        
                        similarity_scores = emb_model.similarity(questions_list, neighbor_triple_text_list) # shape: q_len * nentity
                        weighted_similarity_scores = torch.matmul(similarity_scores.T, weight_vetor) # nentity, q_len * q_len, 1
                        sorted_indices = torch.argsort(weighted_similarity_scores, dim=0)
                        sorted_neighbor_relation_entity_list = [neighbor_relation_entity_list[i.item()] for i in sorted_indices]
                        sorted_neighbor_triple_text_list = [neighbor_triple_text_list[i.item()] for i in sorted_indices]
                        if len(neighbor_relation_entity_list) > 10:
                            sorted_neighbor_relation_entity_list = sorted_neighbor_relation_entity_list[:10] # 这是通过筛选出来的尾实体周围最相关的K个邻居 [(h, r, t), (h1, r1, t1)]
                            sorted_neighbor_triple_text_list = sorted_neighbor_triple_text_list[:10] # 这是通过筛选出来的尾实体周围最相关的K个邻居 ['h r is t', 'h1 r1 is t2', ...]
                        
                        # similarity_scores = similarity_scores.numpy()

                    elif self.use_llm_filter_adj_entity:
                        pass
                    
                    sorted_neighbor_relation_entity_list_str = '\n'.join(str(triplet) for triplet in sorted_neighbor_relation_entity_list)
                    sorted_neighbor_triple_text_list_str = '\n'.join(sorted_neighbor_triple_text_list)

                    input = self.score_candidate_entity_prompt_wrap(question=question,
                                                                    subquestions=subquestions_list_str,
                                                                    topic_entity=topic_entity,
                                                                    pre_relation=relation,
                                                                    candidate_entity=tail,
                                                                    history_path=cur_history_path,
                                                                    neighbor_info=sorted_neighbor_triple_text_list_str)
                    
                    # pdb.set_trace()
                    response = self.io_system.get_local_response(query=input,
                                                                 max_length=self.max_length,
                                                                 max_new_tokens=self.max_new_tokens,
                                                                 temperature=self.temperature,
                                                                 do_sample=self.do_sample,
                                                                 truncation=self.truncation)
                    
                    tail_entity_score = parse_entity_score(xml_response_list=response)
                    if tail_entity_score > best_score:
                        best_tail = tail
                        best_score = tail_entity_score
                        target_relation_score = relation_score
                
                relation_entity2score_filter_dict[(head, relation)] = {best_tail: target_relation_score}
            
        # pdb.set_trace()      
        return relation_entity2score_filter_dict
                    
                        


    def get_reweight_value(self,
                           node: str,
                           question: str,
                           topic_entity: str,
                           history_path: str,
                           relation_entity2score_dict: defaultdict,
                           subquestions_list_str: str=None,) -> defaultdict:
        """
        给定筛选出来的 关系-打分 字典 对尾实体进行检索 并重新进行打分 数量和输入不变
        relation_entity2score_dict: {(hi, ri):[{ti: scorei}, {tj: scorej}, ...], (h1, r1):[{t1: score}], ...}
        output: reweight_relation_entity2score_dict {(hi, ri):{ti: scorei}, (hi, r1):{t1: score}, ...}
        """
        candiate_path_text_list = convert2candiate_path(history_path=history_path, # canditate_path_dict: 
                                                         relation_entity2score_dict=relation_entity2score_dict)
        
        reweight_relation_entity2score_dict = defaultdict(dict)
        for candidate_path in candiate_path_text_list:
            input = self.reweight_value_prompt_wrap(question=question,
                                                    history_path=candidate_path,
                                                    subquestions=subquestions_list_str,
                                                    topic_entity=topic_entity,)
            
            if self.use_vllm:
                response = self.io_system.generate_with_vLLM_model(query=input,
                                                                   n=3)
                # pdb.set_trace()
                one_reweight_relation_entity2score = parse_relation_entity2score_list(xml_response_list=response,
                                                                                      node_text=node,
                                                                                      relation_entity2score_dict=relation_entity2score_dict,
                                                                                      candidate_path=candidate_path) # response: ['<count> 5 </count>', '<choice> location.country.languages_spoken </choice>', '<reason> This relation is directly relevant as it provides the languages spoken in Jamaica, which is the key information needed to answer the question. </reason>', '<score> 0.9 </score>',]

            else:
                response = self.io_system.get_local_response(query=input,
                                                            max_length=self.max_length,
                                                            max_new_tokens=self.max_new_tokens,
                                                            temperature=self.temperature,
                                                            do_sample=self.do_sample,
                                                            truncation=self.truncation)
                # pdb.set_trace()
                one_reweight_relation_entity2score = parse_relation_entity2score(xml_response_list=response,
                                                                                  node_text=node,
                                                                                  relation_entity2score_dict=relation_entity2score_dict,
                                                                                  candidate_path=candidate_path) # response: ['<count> 5 </count>', '<choice> location.country.languages_spoken </choice>', '<reason> This relation is directly relevant as it provides the languages spoken in Jamaica, which is the key information needed to answer the question. </reason>', '<score> 0.9 </score>',]
            reweight_relation_entity2score_dict.update(one_reweight_relation_entity2score)


        return reweight_relation_entity2score_dict

    def get_intension_decompose(self,
                                question: str,
                                topic_entity: str):
        """
        进行问题的意图分解
        输入 原始问题 主题实体
        输出 分解过后的子问题 [subquestion1, subquestion2, ...]
        """
        input = self.get_intension_decompose_prompt_wrap(question=question,
                                                         topic_entity=topic_entity)
        if self.use_vllm:
            response = self.io_system.generate_with_vLLM_model(query=input,
                                                               n=1)
        else:
            response = self.io_system.get_local_response(query=input,
                                                         max_length=self.max_length,
                                                         max_new_tokens=self.max_new_tokens,
                                                         temperature=self.temperature,
                                                         do_sample=self.do_sample,
                                                         truncation=self.truncation)
        subquestions_list = parse_subquestions_list(xml_response_list=response)
        # pdb.set_trace()
        return subquestions_list

    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, is_in_graph, finish, root, path_with_reward, subquestions_list = MCTS_search(self)
        return is_in_graph, finish, root, path_with_reward, subquestions_list

