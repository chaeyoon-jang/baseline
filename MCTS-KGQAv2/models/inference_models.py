import pdb
# pdb.set_trace()
import os
import sys
sys.path.append('.')
from accelerate import Accelerator
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import pdb
from openai import OpenAI
from models.vllm_models import *


_BASE_URL = "https://api.deepseek.com/v1"
os.environ['OPENAI_API_KEY'] = 'sk-c5bfc65aa44c452781271052d6885dea'

# get model and tokenizer
def get_inference_model(model_dir):
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    inference_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    inference_model.eval()
    return inference_tokenizer, inference_model


# get llama model and tokenizer
def get_inference_model_llama(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return inference_tokenizer, inference_model


# get mistral model and tokenizer
def get_inference_model_mistral(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # inference_tokenizer.pad_token = inference_tokenizer.eos_token
    return inference_tokenizer, inference_model


def get_inference_model_qwen(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return inference_tokenizer, inference_model


def get_inference_model_qwen_qwq(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                                           device_map="auto",
                                                           trust_remote_code=True, 
                                                           torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    return inference_tokenizer, inference_model


def get_inference_model_deepseek(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return inference_tokenizer, inference_model


class IO_System:
    """Input/Output system"""

    def __init__(self, args, tokenizer=None, model=None) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = args.temperature
        self.model = model
        self.max_length = args.max_len
        self.max_new_tokens = args.max_new_tokens
        self.do_sample = args.do_sample
        self.truncation = args.truncation
        self.token_conuter = 0
        self.call_counter = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.inference_type = args.propose_method
        self.use_vllm = args.use_vllm


    def get_usage(self):
        model2cost = {
            'qwen': 0.07,
            '4o': 0.09
        }
        if self.completion_tokens != 0:
            cost = self.completion_tokens / 1000 * 0.07 + self.prompt_tokens / 1000 * 0.0018

        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}


    def get_local_response(self, 
                           query, 
                           max_length=2048, 
                           truncation=True, 
                           do_sample=False, 
                           max_new_tokens=1024,
                           temperature=0.7):

        inference_type = self.inference_type
        if 'qwen' in inference_type:
             inference_type = 'qwen'
        elif 'llama' in inference_type:
            inference_type = 'llama'
        assert inference_type in ['qwen', 'llama', 'deepseek']
        
        max_length = self.max_length
        max_new_tokens = self.max_new_tokens
        temperature = self.temperature
        do_sample = self.do_sample
        truncation = self.truncation
        if self.use_vllm:
            # vLLM 경로: 내부 vLLM 생성 함수를 사용해 응답을 반환
            return self.generate_with_vLLM_model(
                query,
                temperature=temperature,
                n=1,
                max_tokens=max_new_tokens,
                stop=[],
            )

        else:
            if inference_type == 'llama':
                return self.get_local_response_llama(query,
                                                    max_length=max_length, 
                                                    max_new_tokens=max_new_tokens,
                                                    temperature=temperature, 
                                                    do_sample=do_sample,
                                                    truncation=truncation)
            
            elif inference_type == 'qwen':
                return self.get_local_response_qwen(query, 
                                                    max_length=max_length, 
                                                    max_new_tokens=max_new_tokens, 
                                                    temperature=temperature, 
                                                    do_sample=do_sample,
                                                    truncation=truncation)

            elif inference_type == 'deepseek':
                return self.get_local_response_deepseek(query,
                                                        max_new_tokens=max_new_tokens,
                                                        temperature=temperature, 
                                                        do_sample=do_sample,
                                                        truncation=truncation)

            elif inference_type == 'mistral':
                return self.get_local_response_mistral(query, 
                                                    max_new_tokens=max_new_tokens,
                                                    temperature=temperature, 
                                                    do_sample=do_sample,
                                                    truncation=truncation)


    def get_api_response(self, query):

        api_type = self.inference_type
        assert api_type in ['qwenapi', '4o-mini', '4o', 'deepseekv3']
        max_length = self.max_length
        # max_new_tokens = self.max_new_token
        temperature = self.temperature
        # do_sample = self.do_sample
        # truncation = self.truncation
        api2model_type = {
            'qwenapi': 'qwen',
            '4o-mini': '4o-mini',
            'deepseekv3': 'deepseek-chat'
        }

        if api_type == 'qwenapi':
            return self.gpt(query, 
                            model=api2model_type[api_type], 
                            temperature=temperature, 
                            max_token=max_length)
        
        elif api_type == '4o-mini':
            return self.gpt(query, 
                            model=api2model_type[api_type],
                            temperature=temperature,
                            max_token=max_length)
        
        elif api_type == 'deepseekv3':
            return self.gpt(query, 
                            model=api2model_type[api_type],
                            temperature=temperature,
                            max_token=max_length)
            


    def gpt(self, 
            query:str, 
            model:str, 
            temperature:float, 
            max_token:int, 
            stop=None):
        
        client = OpenAI()
        client.base_url = _BASE_URL
        output = []
        cnt = 5
        
        while not output and cnt:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query}
                    ],
                    temperature=temperature,
                    max_tokens=max_token,
                    stop=stop,)
                
                self.completion_tokens += completion.usage.completion_tokens
                self.prompt_tokens += completion.usage.prompt_tokens
                output = [completion.choices[0].message.content]
                if output:
                    break
            
            except Exception as e:
                print(f"Error occurred when getting gpt reply!\nError type:{e}\n")
                cnt -= 1
        # pdb.set_trace()
        return output


    def generate_with_vLLM_model(self,
                                query,
                                temperature=0.4,
                                top_p=0.9,
                                top_k=10,
                                repetition_penalty=1.0,
                                n=1,
                                max_tokens=16000,
                                stop=[],):  
        
        cnt = 2
        ## system prompt 분기
        model = self.model
        inference_type = self.inference_type
        # inference_type 정규화
        if 'qwen' in inference_type:
            inference_type = 'qwen'
        elif 'llama' in inference_type:
            inference_type = 'llama'
        split_response = []

        if inference_type == 'qwen':
            input = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n".format(query=query)
        elif inference_type == 'llama':
            input = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'.format(query=query)
        else:
            print(f"경고: 알 수 없는 inference_type={inference_type}, 기본 qwen 포맷 사용")
            input = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n".format(query=query)

        if model is None:
            print("오류: vLLM 모델이 로드되지 않았습니다 (model is None)")
            return split_response


        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            n=n,
            max_tokens=max_tokens,
            stop=stop,
        )
        output = None
        while cnt:
            try:
                output = model.generate(input, sampling_params, use_tqdm=False)
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1
        
        if output is None:
            print("vLLM 모델 생성 실패. 빈 응답 반환")
            return split_response
            
        for o in output[0].outputs:
            split_response.extend(o.text.split('\n'))
        # pdb.set_trace()
        return split_response



    def get_local_response_qwen(self, query, max_length=2048, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
        cnt = 2
        all_response = ''
        tokenizer = self.tokenizer
        model = self.model

        message = "<|im_start|>system\nPlease reason step by step, and output your final answer.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n".format(query=query)
        data = tokenizer.encode_plus(message, max_length=max_length, truncation=truncation, return_tensors='pt')
        input_ids = data['input_ids'].to('cuda')
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.eos_token_id
        ]
        attention_mask = data['attention_mask'].to('cuda')
        # pdb.set_trace()
        while cnt:
            try:
                output = model.generate(input_ids, 
                                        attention_mask=attention_mask, 
                                        do_sample=do_sample, 
                                        max_new_tokens=max_new_tokens, 
                                        temperature=temperature, 
                                        eos_token_id=terminators, 
                                        pad_token_id=tokenizer.pad_token_id,
                                        )

                ori_string = tokenizer.decode(output[0], skip_special_tokens=False)
                processed_string = ori_string.split('<|im_start|>assistant\n')[-1].strip().split('<|im_end|>')[0].strip()
                response = processed_string.strip()

                # print(f'获得回复:{response}\n')
                all_response = response
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1
        if not cnt:
            return []
        # pdb.set_trace()
        split_response = all_response.split('\n')
        return split_response


    def get_local_response_llama(self, query, max_length=2048, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
        cnt = 2
        all_response = ''
        tokenizer = self.tokenizer
        model = self.model
        terminators = [
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        message = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'.format(query=query)
        data = tokenizer.encode_plus(message, max_length=max_length, truncation=truncation, return_tensors='pt')
        input_ids = data['input_ids'].to('cuda')
        attention_mask = data['attention_mask'].to('cuda')
        while cnt:
            try:
                output = model.generate(input_ids, 
                                        attention_mask=attention_mask, 
                                        do_sample=do_sample, 
                                        max_new_tokens=max_new_tokens, 
                                        temperature=temperature, 
                                        eos_token_id=terminators, 
                                        pad_token_id=tokenizer.eos_token_id)

                ori_string = tokenizer.decode(output[0], skip_special_tokens=False)
                # pdb.set_trace()
                processed_string = ori_string.split('<|end_header_id|>')[-1].strip().split('<|eot_id|>')[0].strip()
                response = processed_string.strip()
                all_response = response
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1
        if not cnt:
            return []
        split_response = all_response.split('\n')
        # pdb.set_trace()
        return split_response


    # get mistral model response
    def get_local_response_mistral(self, query, max_length=1024, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
        cnt = 2
        all_response = ''
        tokenizer = self.tokenizer
        model = self.model

        message = '[INST]' + query + '[/INST]'
        data = tokenizer.encode_plus(message, max_length=max_length, truncation=truncation, return_tensors='pt')
        input_ids = data['input_ids'].to('cuda')
        attention_mask = data['attention_mask'].to('cuda')
        while cnt:
            try:
                output = model.generate(input_ids, 
                                        attention_mask=attention_mask, 
                                        max_new_tokens=max_new_tokens, 
                                        do_sample=do_sample, 
                                        temperature=temperature, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        pad_token_id=tokenizer.pad_token_id)

                ori_string = tokenizer.decode(output[0])
                processed_string = ori_string.split('[/INST]')[1].strip()
                response = processed_string.split('</s>')[0].strip()

                print(f'obtain response:{response}\n')
                all_response = response
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1
        if not cnt:
            return []
        all_response = all_response.split('The answer is:')[0].strip()  # intermediate steps should not always include a final answer
        ans_count = all_response.split('####')
        if len(ans_count) >= 2:
            all_response = ans_count[0] + 'Therefore, the answer is:' + ans_count[1]
        all_response = all_response.replace('[SOL]', '').replace('[ANS]', '').replace('[/ANS]', '').replace('[INST]', '').replace('[/INST]', '').replace('[ANSW]', '').replace('[/ANSW]', '')  # remove unique answer mark for mistral
        split_response = all_response.split('\n')
        return split_response

    def get_local_response_deepseek(self, query, max_length=2048, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
        cnt = 2
        all_response = ''
        tokenizer = self.tokenizer
        model = self.model

        message = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'.format(query=query)
        data = tokenizer.encode_plus(message, max_length=max_length, truncation=truncation, return_tensors='pt')
        input_ids = data['input_ids'].to('cuda')
        attention_mask = data['attention_mask'].to('cuda')
        while cnt:
            try:
                output = model.generate(input_ids, 
                                        attention_mask=attention_mask, 
                                        do_sample=do_sample, 
                                        max_new_tokens=max_new_tokens, 
                                        temperature=temperature, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        pad_token_id=tokenizer.pad_token_id)

                ori_string = tokenizer.decode(output[0], skip_special_tokens=False)
                processed_string = ori_string.split('<|end_header_id|>')[2].strip().split('<|eot_id|>')[0].strip()
                response = processed_string.split('<|end_of_text|>')[0].strip()

                # print(f'获得回复:{response}\n')
                all_response = response
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1
        if not cnt:
            return []
        # split_response = all_response.split("Assistant:")[-1].strip().split('\n')
        split_response = all_response.split('\n')
        return split_response