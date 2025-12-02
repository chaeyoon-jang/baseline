import json
import pdb



acc_list = []
hit_list = []
f1_list = []
precission_list = []
recall_list = []
path_list = ['/workspace/xxxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1-320_alltree/qwenv1_eval_result.jsonl',
             '/workspace/xxxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_320-640_alltree/qwenv1_eval_result.jsonl',
             '/workspace/xxxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_640-960_alltree/qwenv1_eval_result.jsonl',
             '/workspace/xxxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_960-1280_alltree/qwenv1_eval_result.jsonl',
             '/workspace/xxxxxx/KGQA/MCTS-KGQAv2/outputs/webqsp/mcts/qwen14b/qwen14b-2-7-3_1280-1600_alltree/qwenv1_eval_result.jsonl',]

for path in path_list:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            f1_score, hit_score, acc_score, precission_score, recall_score  = data['f1'], data['hit'], data["acc"], data["precission"],  data["recall"]
            f1_list.append(f1_score), precission_list.append(precission_score), recall_list.append(recall_score), hit_list.append(hit_score), acc_list.append(recall_score)
# result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list)) + " F1: " + str(sum(f1_list) * 100 / len(f1_list)) + " Precision: " + str(sum(precission_list) * 100 / len(precission_list)) + " Recall: " + str(sum(recall_list) * 100 / len(recall_list))

result_str = "数据总数量：" + str(len(hit_list)) + ",  hit总数量：" + str(sum(hit_list)) + ",  Hit: " + str(sum(hit_list) * 100 / len(hit_list))
print(result_str)