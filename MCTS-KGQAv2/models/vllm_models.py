# Licensed under the MIT license.
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import math
import os
import pdb

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def load_vLLM_model(model_ckpt, seed=42, tensor_parallel_size=1, half_precision=False, max_num_seqs=256,
                    gpu_memory_utilization=0.7):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    if half_precision:
        llm = LLM(
            model=model_ckpt,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            gpu_memory_utilization=gpu_memory_utilization
        )
    else:
        llm = LLM(
            model=model_ckpt,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            gpu_memory_utilization=gpu_memory_utilization
        )

    return tokenizer, llm


def generate_with_vLLM_model(
    model,
    input,
    temperature=0.3,
    top_p=0.95,
    top_k=-1,
    repetition_penalty=1.0,
    n=1,
    max_tokens=16000,
    stop=[],
    inference_type='qwen',
):  
    ## 进行 system prompt的区别
    if inference_type == 'qwen':
        input = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n".format(query=input)
    elif inference_type == 'llama':
        input = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'.format(query=input)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        max_tokens=max_tokens,
        stop=stop,
    )

    output = model.generate(input, sampling_params, use_tqdm=False)
    
    return output


if __name__ == "__main__":
    model_ckpt = "/mnt/models/Qwen2___5-7B-Instruct"
    tokenizer, model = load_vLLM_model(model_ckpt, seed=42, tensor_parallel_size=1, half_precision=False)
    input = "What is the meaning of life?"
    output = generate_with_vLLM_model(model, input)
    pdb.set_trace()
    print(output[0].outputs[0].text)
