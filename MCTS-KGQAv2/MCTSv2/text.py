import sys
import numpy as np
import torch
sys.path.append('..')
from similarities import BertSimilarity
import pdb
# 1.Compute cosine similarity between two sentences.
sentences = ['如何更换花呗绑定银行卡',
             '花呗更改绑定银行卡',
             '花呗更改绑定银行卡']
corpus = [
    '花呗更改绑定银行卡',
    '我什么时候开通了花呗',
    '俄罗斯警告乌克兰反对欧盟协议',
    '暴风雨掩埋了东北部；新泽西16英寸的降雪',
    '中央情报局局长访问以色列叙利亚会谈',
    '人在巴基斯坦基地的炸弹袭击中丧生',
]
model = BertSimilarity(model_name_or_path="/workspace/LLaMA-Factory/models/text2vec-base-multilingual")
print(model)
# similarity_score = model.similarity(sentences[0], sentences[1])
# print(f"{sentences[0]} vs {sentences[1]}, score: {float(similarity_score):.4f}")

print('-' * 50 + '\n')
# 2.Compute similarity between two list
similarity_scores = model.similarity(sentences, corpus) # 2, 6
pdb.set_trace()

weight_list = [0.5]

weight_list = weight_list + [0.5/(len(sentences)-1)] * (len(sentences) - 1) 
weight_vector = torch.tensor(weight_list).reshape(3, 1)
# weight_vector = np.array(weight_list) # 3
score = torch.matmul(similarity_scores.T, weight_vector)
print(similarity_scores.t.matmul(weight_vector) ) # 3, 6



pdb.set_trace()
for i in range(len(sentences)):
    for j in range(len(corpus)):
        print(f"{sentences[i]} vs {corpus[j]}, score: {similarity_scores.numpy()[i][j]:.4f}")

print('-' * 50 + '\n')
# 3.Semantic Search
model.add_corpus(corpus)
res = model.most_similar(queries=sentences, topn=3)
print(res)
for q_id, id_score_dict in res.items():
    print('query:', sentences[q_id])
    print("search top 3:")
    for corpus_id, s in id_score_dict.items():
        print(f'\t{model.corpus[corpus_id]}: {s:.4f}')

print('-' * 50 + '\n')
print(model.search(sentences[0], topn=3))



questions = ['what did james k polk do before he was president?', 
             'Identify James K. Polk\'s career or roles prior to becoming president', 
             'Determine the timeline of James K. Polk\'s life and career', 
             'Research the political and historical context of the United States during James K. Polk\'s pre-presidential years']
entity_list = ['Politician', 
               'Farmer',
               'Lawyer']


