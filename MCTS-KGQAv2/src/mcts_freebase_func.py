from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import pdb
SPARQLPATH = "http://localhost:3001/sparql"  

sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_head_entities_extract_en = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n  ?tailEntity ns:%s ns:%s .\n  FILTER(langMatches(lang(?tailEntity), "en"))\n}"""
sparql_tail_entities_extract_en = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\nFILTER(langMatches(lang(?tailEntity), "en"))\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""
sparql_id_en = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n    FILTER(LANG(?tailEntity) = "en")\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n    FILTER(LANG(?tailEntity) = "en")\n  }\n}"""

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

    
#id2name without limit
def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id_en % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return entity_id
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']
    
#id2name len
def id2entity_name_or_type_len(entity_id):
    sparql_query = sparql_id_en % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return entity_id
    else:
        return len(results["results"]["bindings"])

#retrive entity with head ent and rel
def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = execurte_sparql(head_entities_extract)
    print(len(entities))
    entity_ids = replace_entities_prefix(entities)
    return entity_ids


def relation_search(entity_id):
    """
    retrive all relations of the entity
    """
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)


    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()
    print(total_relations)
    return tail_relations


#@en
def entity_search_en(entity, relation, head=True):
    """
    retrive entity with head ent and rel, and limit type of the entity name is English
    """
    if head:
        tail_entities_extract = sparql_tail_entities_extract_en% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract_en% (relation, entity)
        entities = execurte_sparql(head_entities_extract)

    # print(len(entities))
    entity_ids = replace_entities_prefix(entities)
    return entity_ids


def get_one_entity_all_adj_entity(entity, relations2score_dict: dict) -> dict:
    """
    return all tail entities of the entity and rel and trans the format 
    relations2score_dict: {relation1: score, relation2, score}
    """
    relation_entity2score_dict = defaultdict(dict)
    for relation, score in relations2score_dict.items():
        tail_entity = entity_search_en(entity, relation, head=True)
        for tail_ent in tail_entity:
            relation_entity2score_dict[(entity, relation)] = {tail_ent: score}
    return relation_entity2score_dict
    

def id2entity_name_or_type(entity_id):
    """
    id2name, limit type of the entity name is English
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





