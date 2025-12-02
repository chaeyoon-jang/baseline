from src.mcts_freebase_latest import *

import random

def get_entity_adj_info(entity,sample_num=100):
    """
    Args:
        entity:mid
        sample_num: 随机保留的1hop实体
        双向检索，但是格式固定
        return: target_relation_entity_list: [(relation, neighbor), ...]
        target_triple_text_list: [entity + ' ' + relation + ' is ' + neighbor, ...]
    """
    target_relation_entity_list, target_triple_text_list = [], []
    try:
        _1hop_rel = relation_search(entity)
        for rel in _1hop_rel:
            forward_entities = entity_search(entity, rel, head=True)
            backward_entities = entity_search(entity, rel, head=False)
            entities = forward_entities + backward_entities
            if len(entities) > sample_num:
                _1hop_entities = random.sample(entities, sample_num)
            else:
                _1hop_entities = entities

            for ent in _1hop_entities:
                if check_ent(ent):
                    target_relation_entity_list.append((rel,id2entity_name_or_type(ent)))
                    triple_text = entity + ' ' + rel + ' is ' + id2entity_name_or_type(ent)
                    target_triple_text_list.append(triple_text)
                else:
                    target_relation_entity_list.append((rel,ent))
                    triple_text = entity + ' ' + rel + ' is ' + ent
                    target_triple_text_list.append(triple_text)
        return target_relation_entity_list, target_triple_text_list
    except Exception as e:
        print(f"Error is {e}")
        return target_relation_entity_list, target_triple_text_list

    
    
#测试代码
if __name__ == '__main__':
    with open("val.txt", "w",buffering=1) as f:
        original_stdout = sys.stdout  
        sys.stdout = f
        entity = "m.078w2"
        sample_num = 10 
        target_relation_entity_list, target_triple_text_list = get_entity_adj_info(entity, sample_num)
        #为了方便测试，逐行打印了，实际返回的是list
        for item in target_relation_entity_list:
            print(item)
        for item in target_triple_text_list:
            print(item)
        sys.stdout = original_stdout  