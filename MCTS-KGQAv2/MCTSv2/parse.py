import json


def dump_json(path, datas):
    with open(path, 'w', encoding='utf-8') as f:
        for data in datas:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


# path = '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/webquestions/webquestion_legal_topic_prediction.json'
# path2 = '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/webquestions/webquestion_legal_topic_prediction2.json'
# data_list = []
# with open(path, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         data = json.loads(line)
#         topic_entity = data['topic_entity']
#         topic_flag = data['topic_flag']
#         topic_pre = data['topic_pre']
#         if not topic_flag:
#             for pre in topic_pre:
#                 for ans in topic_entity:
#                     if ans.lower() in pre.lower() or pre.lower() in ans.lower():
#                         topic_flag = True
#         data['topic_flag'] = topic_flag
#         data_list.append(data)
# dump_json(path2, data_list)
            
            
path_list = ['/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1-450_alltree.json',
             '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_450-900_alltree.json',
             '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1350-1800_alltree.json',
             '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_1800-2250_alltree.json',
             '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_2250-2700_alltree.json',
             '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_2700-3150_alltree.json',
             '/workspace/longxiao/KGQA/MCTS-KGQAv2/outputs/cwq/mcts/qwen14b/qwen14b-2-7-3_3150-3600_alltree.json']
qid_list = []
for path in path_list:
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            qid = data['qid']
            qid_list.append(qid)
duplicate_list = list(set([x for x in qid_list if qid_list.count(x) > 1]))
print(duplicate_list)