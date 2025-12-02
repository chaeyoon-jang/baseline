import json



datapath = '/workspace/longxiao/KGQA/MCTS-KGQAv4/outputs/graliqa/qwen14b_0-100_prediction.json'

correct_num = 0
with open(datapath, 'r') as f:
    lines = f.readlines()
    length = len(lines)
    for line in lines:
        data = json.loads(line)
        pre = data['prediction']
        answers = data['answer']
        is_true = False
        for answer in answers:
            for p in pre:
                if p.lower() in answer.lower() or answer.lower() in p.lower():
                    is_true = True
        if is_true:
            correct_num += 1
print(f"正确的{correct_num}/总数{length}", correct_num/length)