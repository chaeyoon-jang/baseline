import pdb
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_inference_model_llama(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return inference_tokenizer, inference_model




def get_local_response_llama(tokenizer, model, query, max_length=2048, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
    cnt = 2
    all_response = ''
    tokenizer = tokenizer
    model = model
    terminators = [
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    message = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'.format(query=query)
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
    pdb.set_trace()
    return split_response





if __name__ == '__main__':
    input = '\n## Instruction:\nAssume you are a **reasoning expert**. You will receive an encyclopedic question, several sub-questions that help solve the main problem, a multi-hop evidence path proposed by a student for the reasoning problem (which needs to be extended), a new **candidate entity** that can extend this path, and relevant attribute information about the candidate entity. Your task is to carefully examine and consider what information is needed to reason about the question and sub-questions, and evaluate whether the relevant attribute information of the candidate entity can assist in deriving answers to these questions. You need to evaluate the relevance of the candidate entity to the reasoning problem and its corresponding sub-questions based on the attribute information surrounding the candidate entity, and provide a score indicating the likelihood that this candidate entity can derive the answers to the question and sub-questions.\nNoted: Please provide a score for the candidate entity\'s inference of the main question and its corresponding sub-questions (should be a decimal between 0 and 1) and explain your reasons. During this process, you should follow the guidelines below:\n## Guidelines:\n1. **Reliability of Information Surrounding the Candidate Entity**: Please carefully consider whether the attribute information surrounding the candidate entity can assist it in inferring the main question and sub-questions.\n2. **Feasibility of Adding the Candidate Entity to the Reasoning Path**: Please thoroughly understand the semantics contained in the question and the existing evidence path, and think about whether incorporating the candidate entity into the evidence path would be helpful for the reasoning problem.\n## Input Format:\n**Question**:\nThe input question\n**Subquestions**:\nThe several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]\n**History path**:\nentity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n\n**Candidate entity**:\nThe candidate entity to be scored\n**Information surrounding the candidate entity**:\nentity1 r1 is entity2 \\n\n...\n## Output Format:\nThe format of the output is the following XML:\nPlease output strictly in XML format, without any additional analysis.\n```\n<entity> The same entity in **Candidate entity** </entity>\n<score> The confidence score 0.0-1.0 of this candidate entity to reason the question and the corresponding subquestions. </score>\n<reason> Provide the reasoning for the score you assigned to the candidate entity for inferring the question. </reason>\n```\nNote that the entity within <entity> </entity> should strictly same as the **Candidate entity** in the input.\n## Example 1\n**Question**:\nWhat does jamaican people speak?\n**Subquestions**:\n[\'Identify the main languages spoken by Jamaican people?\', \'Research the history and development of the Jamaican language?\', \'Determine the percentage of Jamaican people who speak each language?\']\n**History path**:\njamaican -> location.location.nearby_airports -> Norman Manley International Airport\n**Candidate entity**:\nNorman Manley International Airport\n**Information surrounding the candidate entity**:\nNorman Manley International Airport is serves.as primary international airports.\\n\nNorman Manley International Airport is constructed in the 1960s.\\n\n## Output:\n<entity> Norman Manley International Airport </entity>\n<score> 0.12 </score>\n<reason> The information about Norman Manley International Airport and related details are completely unrelated to jamaican\'s languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>\n## Example 2\n**Question**:\nWhich nation has the Alta Verapaz Department and is in Central America?\n**Subquestions**:\n[\'Identify the countries located in Central America?\', \'Determine which country contains the Alta Verapaz Department?\', \'Confirm the geographical location of the Alta Verapaz Department within its country?\']\n**History path**:\nAlta Verapaz Department -> location.location.contains -> Cobán\n**Candidate entity**:\nCobán\n**Information surrounding the candidate entity**:\nCobán was established by Spanish conquistadors.\\n\nCobán speak Spanish.\\n\n## Output:\n<entity> Cobán </entity>\n<score> 0.31 </score>\n<reason> The path formed by the candidate entity Cobán indicates that the Alta Verapaz Department contains Cobán. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this candidate entity successfully reasoning the question is 0.41.</reason>\n## Example 3\n**Question**:\nWhere did the "Country Nation World Tour" concert artist go to college?\n**Subquestions**:\n[\'Identify the artist associated with the "Country Nation World Tour"\', \'Research the educational background of the concert artist\', \'Find out which colleges or universities the artist attended\']\n**History path**:\n"Country Nation World Tour" -> music.concert_tour.artist -> Brad Paisley\n**Candidate entity**:\nBrad Paisley\n**Information surrounding the candidate entity**:\nBrad Paisley born in West Virginia, USA.\\n\nBrad Paisley is married to Kimberly.\\n\nBrad Paisley is graduated from Belmont University.\\n\n## Output:\n<entity> Brad Paisley </entity>\n<score> 0.62 </score>\n<reason> The path formed by the candidate entity Brad Paisley indicates the identity of the performer of the "Country Nation World Tour". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 \'Identify the artist associated with the "Country Nation World Tour"\'. Additionally, the information surrounding the candidate entity includes details about his educational background, which will help to reason the answer. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.62. </reason>\n## Example 4\n**Question**:\nWho is the daughter of the artist who had a concert tour called I Am... World Tour?\n**Subquestions**:\n[\'Identify the artist who had the "I Am... World Tour"\', \'Research the family of the identified artist\', \'Find out if the artist has a daughter and her name\']\n**History path**:\nI Am... World Tour -> music.artist.concert_tours -> Beyoncé Knowles -> people.person.children -> Blue Ivy\n**Candidate entity**:\nBlue Ivy\n**Information surrounding the candidate entity**:\nBlue Ivy born in USA.\\n\nBlue Ivy born on January 7, 2012.\\n\nBlue Ivy garnered attention for her artistic talents.\\n\n## Output:\n<entity> Blue Ivy </entity>\n<score> 0.92 </score>\n<reason> The path formed by the candidate entity Blue Ivy indicates that the artist who held the "I Am... World Tour" is Beyoncé Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 \'Identify the artist who had the "I Am... World Tour"\' and subquestion 2 \'Find out if the artist has a daughter and her name\'. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.92. </reason>\n## Input:\n**Question**:\nwhat did james k polk do before he was president?\n**Subquestions**:\n[\'What were James K. Polk\'s early life and education like?\', \'What were James K. Polk\'s career goals and aspirations before becoming president?\', \'What were James K. Polk\'s notable positions and roles before becoming president?\']\n**History path**:\nJames K. Polk -> government.government_position_held.office_holder -> m.04j60kc\n**Candidate entity**:\nm.04j60kc\n**Information surrounding the candidate entity**:\nm.04j60kc government.government_position_held.district_represented is Tennessee\'s 6th congressional district\nm.04j60kc government.government_position_held.governmental_body is United States House of Representatives\nm.04j60kc government.government_position_held.office_position_or_title is United States Representative\nm.04j60kc government.government_position_held.office_holder is James K. Polk\n## Output:\n'
    
    
    # model_dir = '/workspace/LLaMA-Factory/models/Llama3-1-8B-Instruct'
    model_dir = '/workspace/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct'
    tokenozer, model = get_inference_model_llama(model_dir=model_dir)
    output = get_local_response_llama(tokenizer=tokenozer,
                                      model=model,
                                      query=input)
    