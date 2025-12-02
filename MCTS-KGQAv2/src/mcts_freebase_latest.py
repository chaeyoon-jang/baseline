from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
from collections import deque
import sys
import json
import re
SPARQLPATH = "http://localhost:3001/sparql"  

#retrive all relations of head ent
sparql_head_relations = """PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?relation WHERE { ns:%s ?relation ?x . FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/")) }"""
#retrive all relations of tail ent
sparql_tail_relations = """PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?relation WHERE { ?x ?relation ns:%s . FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/")) }"""
#latest entity search
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?tailEntity WHERE { ?tailEntity ns:%s ns:%s . FILTER((isIRI(?tailEntity) && STRSTARTS(STR(?tailEntity), "http://rdf.freebase.com/ns/")) || (isLiteral(?tailEntity) && LANG(?tailEntity) = "en") || (isLiteral(?tailEntity) && (DATATYPE(?tailEntity) = xsd:date || DATATYPE(?tailEntity) = xsd:dateTime || DATATYPE(?tailEntity) = xsd:gYear || DATATYPE(?tailEntity) = xsd:gYearMonth || DATATYPE(?tailEntity) = xsd:gMonth || DATATYPE(?tailEntity) = xsd:gMonthDay || DATATYPE(?tailEntity) = xsd:gDay || DATATYPE(?tailEntity) = xsd:time))) }"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?tailEntity WHERE { ns:%s ns:%s ?tailEntity . FILTER((isIRI(?tailEntity) && STRSTARTS(STR(?tailEntity), "http://rdf.freebase.com/ns/")) || (isLiteral(?tailEntity) && LANG(?tailEntity) = "en") || (isLiteral(?tailEntity) && (DATATYPE(?tailEntity) = xsd:date || DATATYPE(?tailEntity) = xsd:dateTime || DATATYPE(?tailEntity) = xsd:gYear || DATATYPE(?tailEntity) = xsd:gYearMonth || DATATYPE(?tailEntity) = xsd:gMonth || DATATYPE(?tailEntity) = xsd:gMonthDay || DATATYPE(?tailEntity) = xsd:gDay || DATATYPE(?tailEntity) = xsd:time))) }"""
#@en
sparql_head_entities_extract_en = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n  ?tailEntity ns:%s ns:%s .\n  FILTER(langMatches(lang(?tailEntity), "en"))\n}"""
sparql_tail_entities_extract_en = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\nFILTER(langMatches(lang(?tailEntity), "en"))\n}"""
#id2name
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""
#id2name@en
sparql_id_en = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n    FILTER(LANG(?tailEntity) = "en")\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n    FILTER(LANG(?tailEntity) = "en")\n  }\n}"""


def abandon_rels(relation):
    '''
    Filter 部分关系（可选）
    '''
    if relation == "type.object.type" in relation:
        return True

# def check_ent(entity):
#     #检查entity是否以m.或g.开头
#     if entity.startswith("m.") or entity.startswith("g."):
#         return True
#     else:
#         return False

valid_prefixes = {"m.", "g."}  

def check_ent(entity):
    '''
    check 实体是否为mid形式
    '''
    return any(entity.startswith(prefix) for prefix in valid_prefixes)


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print(results["results"]["bindings"])
    return results["results"]["bindings"]

def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]

def id2entity_name_or_type(entity_id):
    """
    id2name,限定实体名称语言为英语
    """
    sparql_query = sparql_id_en % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return entity_id
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']



# def relation_search(entity_id):
#     """
#     retrive all relations of the entity
#     """
#     sparql_relations_extract_head = sparql_head_relations % (entity_id)
#     head_relations = execurte_sparql(sparql_relations_extract_head)
#     head_relations = replace_relation_prefix(head_relations)
    
#     sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
#     tail_relations = execurte_sparql(sparql_relations_extract_tail)
#     tail_relations = replace_relation_prefix(tail_relations)


#     head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
#     tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]

#     head_relations = list(set(head_relations))
#     tail_relations = list(set(tail_relations))
#     total_relations = head_relations+tail_relations
#     total_relations.sort()
#     tail_relations.sort()
#     # print(total_relations)
#     # print("num of rel:"+str(len(total_relations)))
#     return total_relations
#     # return tail_relations

checked_int = set()
    
def relation_search(entity_id):
    """
    检索实体的所有关系
    """
    # if entity_id in checked_int:
    #     pass
    # else:
    #     if check_ent(entity_id):
    #         checked_int.add(entity_id)
    #     else:
    #         return []
    
    if entity_id not in checked_int:
        if not check_ent(entity_id):
            return []  
        checked_int.add(entity_id) 
         
    try:

        sparql_relations_extract_head = sparql_head_relations % (entity_id)
        head_relations = execurte_sparql(sparql_relations_extract_head)
        head_relations = replace_relation_prefix(head_relations)
        

        sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
        tail_relations = execurte_sparql(sparql_relations_extract_tail)
        tail_relations = replace_relation_prefix(tail_relations)


        head_relations = [rel for rel in head_relations if not abandon_rels(rel)]
        tail_relations = [rel for rel in tail_relations if not abandon_rels(rel)]
        tail_relations.sort()


        total_relations = sorted(set(head_relations + tail_relations))
        return total_relations

    except Exception as e:
        print(f"Error retrieving relations for entity {entity_id}: {e}")
        return []


def entity_search(entity, relation, head=True):
    '''
    通过ent和关系检索另一ent
    这里需要两次调用,分别制定head为True和False
    '''
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = execurte_sparql(head_entities_extract)
    entity_ids = replace_entities_prefix(entities)
    return entity_ids


def BFS_answer_search(topic_ent_list, answer_ent):
    '''
    BFS验证,以graliqa为例
    '''
    if not isinstance(topic_ent_list, list):
        raise ValueError("topic_ent_list must be a list.")
    
    for single_topic_ent in topic_ent_list:
        print(f"Starting BFS for topic entity: {single_topic_ent}")
        ent_queue = deque([(single_topic_ent, [single_topic_ent], 0)])
        visited = set()
        visited.add(single_topic_ent)
        
        while ent_queue:
            current_ent, path_so_far, hops = ent_queue.popleft()
            
            # current ent id2name
            # current_ent_name = id2entity_name_or_type(current_ent)
            
            if current_ent in answer_ent and hops <= 3:
                print("Found Answer!")
                return single_topic_ent, current_ent, path_so_far
            
            if hops > 3:
                break
            
            print(f"Processing entity: {current_ent} , Path so far: {path_so_far}, Hops: {hops}")
            relations = relation_search(current_ent)
            for relation in relations:
                tail_entities_forward = entity_search(current_ent, relation, head=True)
                tail_entities_backward = entity_search(current_ent, relation, head=False)
                tail_entities = tail_entities_forward + tail_entities_backward
                for tail_entity in tail_entities:
                    if tail_entity not in visited:
                        visited.add(tail_entity)
                        ent_queue.append((tail_entity, path_so_far + [relation] + [tail_entity], hops + 1))
        
        print(f"No Answer Found for topic entity: {single_topic_ent}")
    
    print("No Answer Found for all topic entities!")
    return None

#验证代码
if __name__ == "__main__":

    output_file = "graliqa_val.txt"

    with open(output_file, "w") as f:
        original_stdout = sys.stdout  
        sys.stdout = f  


        # print("len of grali_ids", len(grali_ids))
        
        grali_data = []
        with open("graliqa.jsonl", "r") as f_json:
            for line in f_json:
                data = json.loads(line)
                grali_data.append(data)
                    
        # grali_data.sort(key=lambda x: grali_ids.index(x["qid"]))
        # grali_data.sort(key=lambda x: x["qid"])

        print(f"Reading {len(grali_data)} questions from GraliQA dataset.")

        for data in grali_data:
            print("Processing:", data["qid"])
            topic_ent = list(data["topic_entity"].keys())
            answer_ent = set()
            
            if isinstance(data["answer"], list):
                # answer_ent.update(data["answer"])
                for answer in data["answer"]:
                    answer_ent.add(answer["answer_argument"])
                    if answer["answer_type"] != "Entity":
                        print("Not Entity Type", data["qid"])
                        print("\n")
                        answer_ent = set()
                        continue
            else :
                answer_ent.add(data["answer_argument"])
                if answer["answer_type"] != "Entity":
                        print("Not Entity Type", data["qid"])
                        print("\n")
                        answer_ent = set()
                        continue
                    
            # elif isinstance(data["answer"], str):
            #     answer_ent.add(data["answer"])
            # else:
            #     raise ValueError(f"Unexpected type for 'answer': {type(data['answer'])}")
            
            print(f"topic_ent of {data['qid']}:", topic_ent, len(topic_ent))
            print(f"answer_ent of {data['qid']}:", answer_ent, len(answer_ent))
            if not answer_ent:
                print(f"Skipping {data['qid']} because its answer type is not Entity\n")
                # print(f"No Answer Found for QuestionId: {data['qid']}\n")
                continue
            result = BFS_answer_search(topic_ent, answer_ent)
            if result:
                print(f"Answer of QuestionId: {data['qid']} is {result}\n")
            else:
                print(f"No Answer Found for QuestionId: {data['qid']}\n")
            f.flush()

        print("All questions processed.")
        sys.stdout = original_stdout  
