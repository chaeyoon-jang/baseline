import re
import os
from tasks.prompts_v2 import *
import pdb
# pdb.set_trace()

# data: question: str
# mode: 'cot', 'tot', 'mcts'
# method: 'llama', 'gpt', 'local'
class SearchTask(object):
    def __init__(self, data, topic_entity_list, propose_method='glm', value_method='glm'):
        super().__init__()
        self.data = data
        self.propose_method = propose_method
        self.value_method = value_method
        self.value_cache = {}

    def clear_cache(self):
        self.value_cache = {}

    @staticmethod
    def filter_and_score_prompt_wrap(question: str, 
                                     subquestions: str,
                                     history_path: str, 
                                     input_relations_text: str, 
                                     topic_entity: str, 
                                     budget: int) -> str:
        print('\n', '==============================', '过滤关系 并打分!', '==============================', '\n')
        print('propose_prompt: \n', '本问题是:  ', question, '\n若干子问题:  ', subquestions, '\n主题实体是:', topic_entity, '\n已有的推理步骤:\n' + history_path + ' 基于以上步骤，可能的当前步可能的关系是:\n')
        prompt = filter_and_score_edges_prompt.format(question=question, 
                                                      subquestions=subquestions,
                                                      topic_entity=topic_entity,
                                                      path=history_path,
                                                      relation=input_relations_text,
                                                      budget=budget)
        return prompt
    
    @staticmethod
    def rank_and_score_prompt_wrap(question: str,
                                   history_path: str,
                                   input_relations_text: str,
                                   topic_entity: str,) -> str:
        print('\n', '==============================', '排序关系 并打分!', '==============================', '\n')
        print('propose_prompt: \n', '本问题是:  ', question, '\n 主题实体是:', topic_entity, '\n已有的推理步骤:\n' + history_path + ' 基于以上步骤，可能的当前步可能的关系是:\n')
        prompt = rank_edges_prompt.format(question=question, 
                                          topic_entity=topic_entity,
                                          path=history_path,
                                          relation=input_relations_text,)
        return prompt
        

    @staticmethod
    def reweight_value_prompt_wrap(question: str,
                                   subquestions: str,
                                   history_path: str,
                                   topic_entity: str,
                                   ):
        """
        history_path: topic_entity -> r -> t1 \n topic_entity -> r -> t2 \n ....
        topic_entity: entity
        """
        print('\n', '==============================', '过滤关系 并打分!', '==============================', '\n')
        print('propose_prompt: \n', '本问题是:  ', question,  '\n 若干子问题:  ', subquestions, '\n主题实体是:', topic_entity, '\n已有的推理步骤:\n' + history_path + '基于以上步骤，可能的当前步可能的关系是:\n')
        prompt = reweight_value_prompt.format(question=question,
                                              subquestions=subquestions,
                                              topic_entity=topic_entity,
                                              candidate_path=history_path,
                                              )
        
        return prompt

    @staticmethod
    def get_intension_decompose_prompt_wrap(question: str,
                                            topic_entity: str,):
        print('\n', '==============================', '分解问题意图!', '==============================', '\n')
        print('propose_prompt: \n', '本问题是:  ', question, '\n主题实体是:', topic_entity,)
        prompt = intension_compose.format(question=question,
                                              topic_entity=topic_entity,
                                              )
        
        return prompt
    
    @staticmethod
    def score_candidate_entity_prompt_wrap(question: str,
                                           subquestions: str,
                                           topic_entity: str,
                                           pre_relation: str,
                                           candidate_entity: str,
                                           history_path: str,
                                           neighbor_info: str):
        print('\n', '==============================', '给候选的尾实体打分!', '==============================', '\n')
        print('propose_prompt: \n', '本问题是:  ', question, '\n子问题是:  ', subquestions, '\n前一跳关系是:', pre_relation, '\n候选尾实体是:', candidate_entity, '',  '\n历史路径是:', history_path, '\n邻接信息是:', neighbor_info, )
        prompt = score_candidate_entity.format(question=question,
                                               subquestions=subquestions,
                                               candidate_entity=candidate_entity,
                                               history_path=history_path,
                                               neighbor_info=neighbor_info,
                                               )
        
        return prompt


    @staticmethod
    def zero_single_propose_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n') # x是问题 y是已有的解题步骤
        if lang == 'zh':
            if not y:
                y = '无\n'
            prompt = zero_single_proposal_prompt + x + '\n已有步骤:\n' + y + '\n输出:'
        else:
            if not y:
                y = 'None\n'
            # prompt = zero_single_proposal_prompt_en.format(x, y) zero_go_a_step_proposal_prompt_en
            prompt = zero_single_proposal_prompt_en.format(x, y)
        return prompt

    @staticmethod
    def zero_single_plan_wrap(x: str,  
                              step: int = 0) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n' + '对于该问题，可能的解题规划是:\n') # x是问题 y是已有的解题步骤
        y = ''
        prompt = zero_single_plan_prompt_en.format(x)
        # prompt = zero_single_plan_prompt_en_v2.format(x, y)
        return prompt


    @staticmethod
    def zero_single_propose_wrap_gpt(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            prompt = zero_single_proposal_prompt_gpt + x + '\n已有步骤:\n' + y + '\n输出:'
        else:
            if not y:
                y = 'None\n'
            prompt = zero_single_proposal_prompt_gpt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap_use_reflection(x: str, y: str = '', step: int = 0, ref: str = '', lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            if not ref:
                ref = '无\n'
            prompt = zero_single_proposal_prompt_use_reflection + x + '\n已有步骤:\n' + y + '\n意见:' + ref + '\n输出:'
        else:
            if not y:
                y = 'None\n'
            if not ref:
                ref = 'None\n'
            prompt = zero_single_proposal_prompt_use_reflection_en + x + '\nExisting Steps:\n' + y + '\nAnalysis: ' + ref + '\nOutput:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap_use_reflection_gpt(x: str, y: str = '', step: int = 0, ref: str = '', lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            if not ref:
                ref = '无\n'
            prompt = zero_single_proposal_prompt_use_reflection_gpt + x + '\n已有步骤:\n' + y + '\n意见:' + ref + '\n'
        else:
            if not y:
                y = 'None\n'
            if not ref:
                ref = 'None\n'
            prompt = zero_single_proposal_prompt_use_reflection_gpt_en + x + '\nExisting Steps:\n' + y + '\nAnalysis: ' + ref + '\nOutput:'
        return prompt

    @staticmethod
    def single_reflection_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            prompt = single_reflection_prompt + x + '\n已有步骤:\n' + y + '\n输出:'  # glm style
        else:
            if not y:
                y = 'None\n'
            prompt = single_reflection_prompt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def single_reflection_wrap_gpt(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        if not y:
            y = '无\n'
        prompt = single_reflection_prompt_gpt + x + '\n已有步骤:\n' + y  # gpt style
        return prompt

    @staticmethod
    def single_evaluation_warp_gpt(x: str, y:str='', gt:str='', step: int=0 ) -> str:
        print('\n', '==============================', 'self-evaluation', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        if not y:
            y = '无\n'
        
        prompt = self_eval_prompt.format(gt, y)
        return prompt

    @staticmethod
    def single_reflection_wrap_llama(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        if not y:
            y = '无\n'
        prompt = single_reflection_prompt_llama + x + '\n已有步骤:\n' + y + '\n空\n请你给出意见，不要解答问题，你给出的意见应该完全基于给定的步骤。'  # llama style
        return prompt

    @staticmethod
    def single_reflection_wrap_simple(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            prompt = single_reflection_prompt_simple + x + '\n已有步骤:\n' + y + '\n输出:'  # simple style
        else:
            if not y:
                y = 'None\n'
            prompt = single_reflection_prompt_simple_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def single_reflection_wrap_simple_mistral(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        if not y:
            y = 'None\n'
        prompt = single_reflection_prompt_simple_mistral + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'critic', '==============================', '\n')
        value_prompt = critic_simplified + x + '\n已有步骤:\n' + y.strip() + '\n输出:'
        return value_prompt

    @staticmethod
    def value_prompt_wrap_en(x: str, y: str) -> str:
        print('\n', '==============================', 'critic', '==============================', '\n')
        value_prompt = critic_simplified_en + x + '\nExisting Steps:\n' + y.strip() + '\nOutput:'
        return value_prompt


    @staticmethod
    def self_critic_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'self-critic', '==============================', '\n')
        if not y:
            y = 'None\n'
        pdb.set_trace()
        critic_prompt = self_critic_prompt + x + '\nSolution:\n' + y + '\nScore:'
        return critic_prompt

    @staticmethod
    def cot_prompt_wrap(x: str, lang: str = 'zh', use_math: bool = False) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\n')
        if not use_math:
            if lang == 'zh':
                prompt = cot_prompt + x + "\n解答过程:"
            else:
                prompt = cot_prompt_en + x + "\nSolution:"
        else:
            prompt = MATH_cot_prompt.format(query=x)
        print('propose_prompt: \n', prompt, '\n')
        return prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, low=0.0, high=1.0) -> float:
        out_value = low
        all_out = ''
        for _ in value_outputs:
            all_out = all_out + _
        if '分数' not in all_out:
            print('分数输出不合法!\n')
            return out_value
        stp = all_out.split('分数')[-1].strip()
        try:
            match = re.findall(r'-?[0-9]+\.?[0-9]*', stp)[-1]
            out_value = float(match)
            out_value = min(max(low, out_value), high)
        except Exception as e:
            print(f'分数输出有误！错误类型:{e}\n')
            return low
        return out_value


    @staticmethod
    def value_outputs_unwrap_en(value_outputs: list, low=0.0, high=1.0) -> float:
        out_value = low
        all_out = ''
        for _ in value_outputs:
            all_out = all_out + _
        if 'score' not in all_out.lower():
            print('分数输出不合法!\n')
            return out_value
        stp = all_out.lower().split('score')[-1].strip()
        try:
            match = re.findall(r'-?[0-9]+\.?[0-9]*', stp)[-1]
            out_value = float(match)
            out_value = min(max(low, out_value), high)
        except Exception as e:
            print(f'分数输出有误！错误类型:{e}\n')
            return low
        return out_value
