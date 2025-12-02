


# """
# Please provide as few highly relevant relations as possible to the question and its subobjectives from the following relations (separated by semicolons).
# Here is an example:
# Q: Name the president of the country whose main spoken language was Brahui in 1980?
# Subobjectives: ['Identify the countries where the main spoken language is Brahui', 'Find the president of each country', 'Determine the president from 1980']
# Topic Entity: Brahui Language
# Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
# The output is: 
# ['language.human_language.main_country','language.human_language.countries_spoken_in','base.rosetta.languoid.parent']

# Now you need to directly output relations highly related to the following question and its subobjectives in list format without other information or notes.
# Q: 
# <reasons>
# Your analysis and reasoning supporting your choice.
# </reasons>

# <scores>
# The score (0.0-1.0) to supporting your choice.
# </scores>

# <choiced_relation>
# Your choiced relationships
# </choiced_relation>
# """



# filter_and_score_edges_prompt = """
# # Instruction:
# Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, related topic entities, a multi-hop path that helps derive the answer to the question, and several retrieved relationships that need to be filtered. Your task is to carefully consider the information needed to reason about the question and the semantics of the existing reasoning path. Choose which relationships, after the entity at the end of the path, can better assist in reasoning the answer to the question based on the current reasoning path. Please output your reasons and scores and choice (The score should be a decimal between 0 and 1).
# Noted: the number of the choiced relationship is no more than 3. If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when choosing the relationships.

# # Guidelines
# Please format the input and the output in the following structured format:
# 1. The format of the input is
# ### Question:
# The input question

# ### Topic entity:
# The related topic entities

# ### Multi-hop path:
# The multi-hop path invovle entity and relation connected by arrows 

# ### Several retrieved relationships:
# The several retrieved relationships separated by semicolons (;)
# 2. The format of the output is
# Please provide your reasons, scores, and choiced relation in in JSON format, which can be parsed in Python:
# ```
# {{   
#     {{"choice": "xxxx", "reason": "xxxx", "score": 0.0-1.0}},
#     {{"choice": "xxxx", "reason": "xxxx", "score": 0.0-1.0}},
#     {{"choice": "xxxx", "reason": "xxxx", "score": 0.0-1.0}},
# }}
# ```

# # Example
# ### Question:
# Name the president of the country whose main spoken language was Brahui in 1980?
# ### Topic entity:
# Brahui Language
# ### Multi-hop path:
# ### Several retrieved relationships:
# language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region; language.human_language.main_country
# ### output:
# ```
# {{   
#     {{"choice": "language.human_language.main_country", "reason": "This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.", "score": 0.4}},
#     {{"choice": "language.human_language.countries_spoken_in", "reason": "This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.", "score": 0.3}},
#     {{"choice": "base.rosetta.languoid.parent", "reason": "This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.", "score": 0.2}},
# }}
# ```
# # Input
# ### Question:
# {question}
# ### Topic entity:
# {topic_entity}

# ### Multi-hop path:
# {path}

# ### Several retrieved relationships:
# {relation}

# # Your Response
# **output**:
# """


# filter_and_score_edges_prompt = """
# # Instruction:
# Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, related topic entities, a multi-hop path that helps derive the answer to the question, and several retrieved relationships that need to be filtered. Your task is to carefully consider the information needed to reason about the question and the semantics of the existing reasoning path. Choose which relationships, after the entity at the end of the path, can better assist in reasoning the answer to the question based on the current reasoning path. Please output your reasons and scores and choice (The score should be a decimal between 0 and 1).
# Noted: the number of the choosed relationship is no more than 5. If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when choosing the relationships.

# # Guidelines
# Please format the input and the output in the following structured format:
# 1. The format of the input is
# ### Question:
# The input question
# ### Topic entity:
# The related topic entities
# ### Multi-hop path:
# The multi-hop path invovle entity and relation connected by arrows 
# ### Several retrieved relationships:
# The several retrieved relationships separated by semicolons (;)
# ### Budget
# The budget amount of the chosen relationship.

# 2. The number of the choosed relationship is no more than budget. reset counter between <count> and </count> to {budget}.
# You are allowed to select {budget} relationship (starting budget), keep track of it by counting down within tags <count> </count>, STOP SELECTING MORE RELATIONSHIPS when hitting 0.
# Please provide your count, reasons, scores, and choiced relation in the following XML format.

# <count> [starting budget] </count>
# <choice> Your choiced relation 1. </choice>
# <score> The confidence score 0.0-1.0 to relation 1. </score>

# <count> [remaining budget] </count>
# <choice> Your choiced relation 2. </choice>
# <score> The confidence score 0.0-1.0 to relation 2. </score>
# ...
# <count> 1 </count>
# <choice> Your choiced relation {{budget}}. </choice>
# <score> The confidence score 0.0-1.0 to relation 1. </score>


# # Example
# ### Question:
# Name the president of the country whose main spoken language was Brahui in 1980?
# ### Topic entity:
# Brahui Language
# ### Multi-hop path:
# ### Several retrieved relationships:
# language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region; language.human_language.main_country
# ### Budget
# 3 
# ### output:
# <count> 3 </count>
# <choice> language.human_language.main_country </choice>
# <score> 0.8 </score>;
# <count> 2 </count>
# <choice> language.human_language.countries_spoken_in. </choice>
# <score> 0.6 </score>;
# <count> 1 </count>
# <choice> base.rosetta.languoid.parent. </choice>
# question. </reason>
# <score> 0.3 </score>

# # Input
# ### Question:
# {question}
# ### Topic entity:
# {topic_entity}
# ### Multi-hop path:
# {path}
# ### Several retrieved relationships:
# {relation}
# ### Budget
# {budget}
# # Your Response
# ### output:
# """



reweight_value_prompt="""
Assuming you are a **reasoning expert**. You will receive an encyclopedic question, related topic entity, and a candidate multi-hop path for reasoning about the question (partially or fully). Your task is to carefully consider the relevance of this candidate path to reasoning about the question. Please output the relative likelihood (a score between 0.0 and 1.0) for this path to infer the question, and your reasons. In this process, you should carefully consider the following guidelines:

# Guidelines:
1. Whether the existing path is on the correct reasoning track; please evaluate the value of the path from a long-term perspective.
2. Consider the semantics contained in the question and the corresponding thematic entities, and determine whether the existing paths contain key information necessary for reasoning.
3. The format of the input is:
   ### Question: 
   The input question
   ### Topic entity: 
   The related topic entities
   ### Candidate multi-hop path:
   entity1 -> relation1 -> entity2 -> relation2 -> entity3 \\n
4. The format of the output is the following XML:
   <path> entity1 -> relation1 -> entity2 -> relation2 -> entity3 </path>
   <score> The confidence score 0.0-1.0 of this path to reason the question. </score>
   <reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
Note that the path within <path> </path> should strictly match the path in the input.
# Input:
### Question:
{question}
### Topic entity:
{topic_entity}
### Candidate multi-hop path:
{candidate_path}
# Output:
"""



filter_and_score_edges_prompt = """
## Instruction:
Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, related topic entities, a multi-hop path that helps derive the answer to the question, and several retrieved relationships that need to be filtered to help infer the question. Your task is to carefully consider the information needed to reason about the question and the semantics of the existing reasoning path. Choose which relationships, after the entity at the end of the path, can better assist in reasoning the answer to the question based on the current reasoning path. 
Noted: If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when choosing the relationships. Please output the relationship you have chosen that is most likely to infer the answer, along with the score (should be a decimal between 0 and 1) and the reason for this score.
## Guidelines
1. The format of the input is:
   **Question**: 
   The input question
   **Topic entity**: 
   The related topic entities
   **Multi-hop path**: 
   The multi-hop path invovle entity and relationship connected by arrows 
   **Several retrieved relationships**: 
   The several retrieved relationships separated by semicolons (;)
   **Budget**: 
   The budget amount of the chosen relationship.
2. The number of the choosed relationship is no more than budget. reset counter between <count> and </count> to {budget}.
3. You are allowed to select {budget} relationships (starting budget), keep track of it by counting down within tags <count> </count>, STOP GENERATING MORE RELATIONSHIPS when hitting 0.
4. Please provide your count, reasons, scores, and choiced relationship in the following XML format.
Noted: The relationship you choose must be the one most likely to help infer the answer contained in the **Several retrieved relationships**. Output your chosen relationship exactly as it is.
   <count> [starting budget] </count>
   <choice> The relationship 1 you choose </choice>
   <reason> Provide the reasons for the score you assigned to the relationship 1 for helping infer the question. </reason>
   <score> The confidence score 0.0-1.0 to choose this relation </score>
   <count> [remaining budget] </count>
   <choice> The relationship 2 you choose </choice>
   <reason> Provide the reasons for the score you assigned to the relationship 2 for helping infer the question. </reason>
   <score> The confidence score 0.0-1.0 to choose this relationship </score>
   ...
   <count> 1 </count>
   <choice> The relationship {{budget}} you choose </choice>
   <reason> Provide the reasons for the score you assigned to the relationship 3 for helping infer the question. </reason>
   <score> The confidence score 0.0-1.0 to choose this relationship </score>
## Example
**Question**:
Name the president of the country whose main spoken language was Brahui in 1980?
**Topic entity**:
Brahui Language
**Multi-hop path**:
**Several retrieved relationships**:
language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region; language.human_language.main_country
**Budget**:
3 
## Output:
```
<count> 3 </count>
<choice> language.human_language.main_country </choice>
<reason> This relationship is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980. </reason>
<score> 0.83 </score>
<count> 2 </count>
<choice> language.human_language.countries_spoken_in </choice>
<reason> This relationship is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president. </reason>
<score> 0.62 </score>
<count> 1 </count>
<choice> base.rosetta.languoid.parent </choice>
<reason> This relationship is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. </reason>
<score> 0.23 </score>
```
## Input:
**Question**:
{question}
**Topic entity**:
{topic_entity}
**Multi-hop path**:
{path}
**Several retrieved relationships**:
{relation}
**Budget**:
{budget}
## Output:
"""









################################## 12.11 #####################################
# filter_and_score_edges_prompt = """
# ## Instruction:
# Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, related topic entities, a multi-hop path that helps derive the answer to the question, and several retrieved relationships that need to be filtered to help infer the question. Your task is to carefully consider the information needed to reason about the question and the semantics of the existing reasoning path. Choose the top {{budget}} relationships that are most likely to assist in reasoning the answer to the question based on the multi-hop path.
# Noted: If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when choosing the relationships. Please output the relationship you have chosen that is most likely to infer the answer, along with the score (should be a decimal between 0 and 1) and the reason for this score.
# ## Guidelines
# 1. The format of the input is:
#    **Question**: 
#    The input question
#    **Topic entity**: 
#    The related topic entities
#    **Multi-hop path**: 
#    The multi-hop path invovle entity and relationship connected by arrows 
#    **Several retrieved relationships**: 
#    The several retrieved relationships separated by semicolons (;)
#    **Budget**: 
#    The number of the chosen relationship most likely to infer the result.
# 2. The number of the choosed relationship is no more than budget. reset counter between <count> and </count> to {{budget}}.
# 3. You are allowed to select {{budget}} relationships (starting budget), keep track of it by counting down within tags <count> </count>, STOP GENERATING MORE RELATIONSHIPS when hitting 0.
# 4. Please provide your count, reasons, scores, and choiced relationship in the following XML format.
# Noted: The relationship you choose must be the one most likely to help infer the answer contained in the **Several retrieved relationships**. Output your chosen relationship exactly as it is.
#    <count> [starting budget] </count>
#    <choice> The relationship 1 you choose </choice>
#    <reason> Provide the reasons for the score you assigned to the relationship 1 for helping infer the question. </reason>
#    <score> The confidence score 0.0-1.0 to choose this relation </score>
#    <count> [remaining budget] </count>
#    <choice> The relationship 2 you choose </choice>
#    <reason> Provide the reasons for the score you assigned to the relationship 2 for helping infer the question. </reason>
#    <score> The confidence score 0.0-1.0 to choose this relationship </score>
#    ...
#    <count> 1 </count>
#    <choice> The relationship {{budget}} you choose </choice>
#    <reason> Provide the reasons for the score you assigned to the relationship 3 for helping infer the question. </reason>
#    <score> The confidence score 0.0-1.0 to choose this relationship </score>
# ## Example
# **Question**:
# Name the president of the country whose main spoken language was Brahui in 1980?
# **Topic entity**:
# Brahui Language
# **Multi-hop path**:
# **Several retrieved relationships**:
# language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region; language.human_language.main_country
# **Budget**:
# 3 
# ## Output:
# ```
# <count> 3 </count>
# <choice> language.human_language.main_country </choice>
# <reason> This relationship is the most highly relevant relation in the **Several retrieved relationships**, as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980. </reason>
# <score> 0.92 </score>
# <count> 2 </count>
# <choice> language.human_language.countries_spoken_in </choice>
# <reason> This relationship is the second relevant relation, as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president. </reason>
# <score> 0.85 </score>
# <count> 1 </count>
# <choice> base.rosetta.languoid.parent </choice>
# <reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. </reason>
# <score> 0.79 </score>
# ```
# ## Input:
# **Question**:
# {question}
# **Topic entity**:
# {topic_entity}
# **Multi-hop path**:
# {path}
# **Several retrieved relationships**:
# {relation}
# **Budget**:
# {budget}
# ## Output:
# """

# reweight_value_prompt="""
# ## Instruction:
# Assume you are a **reasoning expert**. You will receive an encyclopedic question, related topic entities, and a multi-hop evidence path that a student has thought of for reasoning about the question (the path may be partial or complete). Your task is to carefully review the information needed to reason about the question and the logic contained in the multi-hop evidence path provided by the student. You need to assess the relevance of this multi-hop evidence path to reasoning about the question. 
# Note: Please output the score for this evidence path to infer the question, and provide your reasons. In this process, you should carefully consider the following guidelines:

# ## Guidelines:
# 1. **Current Feasibility of the Approach**: Please carefully consider the semantics contained in the question and the corresponding topic entities, and determine whether the student's current evidence path includes the key information necessary for reasoning.
# 2. **Future Feasibility of the Approach**: Please think carefully from a long-term perspective about what additional information is needed to reason about the question, and whether the existing evidence path is on the correct track. Evaluate the value of this path from a long-term standpoint!

# ## Input Format:
# **Question**: 
# The input question
# **Topic entity**: 
# The related topic entities
# **Candidate multi-hop evidence path**:
# entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n

# ## Output Format
# The format of the output is the following XML:
# ```
# <path> The same path in **Candidate multi-hop evidence path**: entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... </path>
# <score> The confidence score 0.0-1.0 of this path to reason the question. </score>
# <reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
# ```
# Note that the path within <path> </path> should strictly same as the path in the input.
# ## Example 1
# **Question**:
# What does jamaican people speak?
# **Topic entity**:
# jamaican
# **Candidate multi-hop evidence path**:
# jamaican -> location.location.nearby_airports -> Norman Manley International Airport
# ## Output:
# <path> jamaican -> location.location.nearby_airports -> Norman Manley International Airport </path>
# <score> 0.15 </score>
# <reason> The airports near jamaican are completely unrelated to their languages, and the subsequent path also makes it difficult to derive an answer. </reason>
# ## Example 2
# **Question**:
# Which nation has the Alta Verapaz Department and is in Central America?
# **Topic entity**:
# Alta Verapaz Department
# **Candidate multi-hop evidence path**:
# Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
# ## Output:
# <path> Alta Verapaz Department -> location.location.contains -> Cob\u00e1n </path>
# <score> 0.31 </score>
# <reason> This path indicates the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. So, I believe the score of this path successfully reasoning the question is 0.31.</reason>
# ## Example 3
# **Question**:
# Where did the \"Country Nation World Tour\" concert artist go to college?
# **Topic entity**:
# \"Country Nation World Tour\"
# **Candidate multi-hop evidence path**:
# \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
# ## Output:
# <path> \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley </path>
# <score> 0.62 </score>
# <reason> This evidence path provides the identity of the performer of \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, taking one more step down the path has a high probability of revealing which college the performer attended, leading to the correct answer. So, I believe the score of this path successfully reasoning the question is 0.62. </reason>
# ## Example 4
# **Question**:
# Who is the daughter of the artist who had a concert tour called I Am... World Tour?
# **Topic entity**:
# I Am... World Tour
# **Candidate multi-hop evidence path**:
# I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
# ## Output:
# <path> I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy </path>
# <score> 0.92 </score>
# <reason> This path shows through reasoning that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. </reason>

# ## Input:
# **Question**:
# {question}
# **Topic entity**:
# {topic_entity}
# **Candidate multi-hop evidence path**:
# {candidate_path}
# ## Output:
# """










# v2

filter_and_score_edges_prompt = """
## Instruction:
Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, a multi-hop path that helps derive the answer to the question, and a set of several retrieved relationships that need to be filtered (which need to assist in inferring the question and the subquestions). Your task is to carefully consider the information needed to reason about the question or the subquestions, and, based on the semantics of the existing reasoning path, select the top {{budget}} relationships from the set that are most likely to help infer the answer to the question and subquestions.
Noted: If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when selecting the relationships. Please output the relationships you have selected that are most likely to infer the answer to the subquestions, along with the score (should be a decimal between 0 and 1) and your reasons for this score.
## Guidelines
1. The format of the input is:
   **Question**: 
   The input question
   **Subquestions**:
   The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
   **Topic entity**: 
   The related topic entities
   **Multi-hop path**: 
   The multi-hop path invovle entity and relationship connected by arrows 
   **Several retrieved relationships**: 
   A set of several retrieved relationships separated by semicolons (;)
   **Budget**: 
   The number of the selected relationships that are most likely to infer the result.
2. The number of the selected relationships are no more than {{budget}}. reset counter between <count> and </count> to {{budget}}.
3. You are allowed to select {{budget}} relationships (starting budget), keep track of it by counting down within tags <count> </count>, STOP GENERATING MORE RELATIONSHIPS when hitting 0.
4. You need to carefully think about which relationships can help reason through the subquestions.
5. Please provide your count, reasons, scores, and selected relationships in the following XML format.
Noted: The relationships you select must be consistent with that in **Several retrieved relationships**. Please output relationships your select exactly as they are.
   <count> [starting budget] </count>
   <choice> The relationship you select that is most likely to infer the question and subquestions. </choice>
   <reason> Provide the reasons for the score you assigned to the relationship 1 for helping infer the question and subquestions. </reason>
   <score> The confidence score 0.0-1.0 to select this relation </score>
   <count> [remaining budget] </count>
   <choice> The 2-th relationship you select that is likely to infer the question and subquestions. </choice>
   <reason> Provide the reasons for the score you assigned to the relationship 2 for helping infer the question and subquestions. </reason>
   <score> The confidence score 0.0-1.0 to select this relationship </score>
   ...
   <count> 1 </count>
   <choice> The {{budget}}-th relationship you select that is likely to infer the question and subquestions. </choice>
   <reason> Provide the reasons for the score you assigned to the relationship {{budget}} for helping infer the question and subquestions. </reason>
   <score> The confidence score 0.0-1.0 to select this relationship </score>
## Example
**Question**:
Name the president of the country whose main spoken language was Brahui in 1980?
**Subquestions**:
['Identify the country where Brahui was the main spoken language in 1980?', 'Find out who the president of that country was in 1980?', 'Research the historical context of the Brahui language and its speakers in 1980?']
**Topic entity**:
Brahui Language
**Multi-hop path**:
**Several retrieved relationships**:
language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region; language.human_language.main_country
**Budget**:
3 
## Output:
```
<count> 3 </count>
<choice> language.human_language.main_country </choice>
<reason> This relationship is the most highly relevant relation in the **Several retrieved relationships**, as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980. language.human_language.main_country is highly related on the subquestion 1 'Identify the country where Brahui was the main spoken language in 1980?'.</reason>
<score> 0.92 </score>
<count> 2 </count>
<choice> language.human_language.countries_spoken_in </choice>
<reason> This relationship is the second relevant relation, as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president. language.human_language.countries_spoken_in is highly related on the subquestion 2 'Identify the country where Brahui was the main spoken language in 1980?', and can help to reason this subquestion. </reason>
<score> 0.85 </score>
<count> 1 </count>
<choice> base.rosetta.languoid.parent </choice>
<reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. It is highly related on subquestion 3 'Research the historical context of the Brahui language and its speakers in 1980?' </reason>
<score> 0.79 </score>
```
## Input:
**Question**:
{question}
**Subquestions**:
{subquestions}
**Topic entity**:
{topic_entity}
**Multi-hop path**:
{path}
**Several retrieved relationships**:
{relation}
**Budget**:
{budget}
## Output:
"""





# filter_and_score_edges_prompt = """
# ## Instruction:
# Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, a multi-hop path that helps derive the answer to the question, and a set of several retrieved relationships that need to be filtered (which need to assist in inferring the question and the subquestions). Your task is to carefully consider the information needed to reason about the question or the subquestions, and, based on the semantics of the existing reasoning path, select the top {{budget}} relationships from the set that are most likely to help infer the answer to the question and subquestions.
# Noted: If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when selecting the relationships. Please output the relationships you have selected that are most likely to infer the answer to the subquestions, along with the score (should be a decimal between 0 and 1) and your reasons for this score.
# ## Guidelines
# 1. **Clarify the required information**: Identify the key information needed to answer the question, and select the relationships most relevant to the key information.
# 2. **Consider the information of subquestions**: You need to carefully think about which relationships can help reason through the subquestions.
# ## The format of the input is:
# **Question**: 
# The input question
# **Subquestions**:
# The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
# **Topic entity**: 
# The related topic entities
# **Multi-hop path**: 
# The multi-hop path invovle entity and relationship connected by arrows 
# **Several retrieved relationships**: 
# A set of several retrieved relationships separated by semicolons (;)
# **Budget**: 
# The number of the selected relationships that are most likely to infer the result.
# Noted: 
# 1. The relationships you select must be consistent with that in **Several retrieved relationships**. Please output relationships your select exactly as they are.
# 2. The number of the selected relationships are no more than {{budget}}. reset counter between <count> and </count> to {{budget}}.
# 3. You are allowed to select {{budget}} relationships (starting budget), keep track of it by counting down within tags <count> </count>, STOP GENERATING MORE RELATIONSHIPS when hitting 0.
# 4. You need to carefully think about which relationships can help reason through the subquestions.
# ## Please provide your count, reasons, scores, and selected relationships in the following XML format:
# <count> [starting budget] </count>
# <choice> The relationship you select that is most likely to infer the question and subquestions. </choice>
# <reason> Provide the reasons for the score you assigned to the relationship 1 for helping infer the question and subquestions. </reason>
# <score> The confidence score 0.0-1.0 to select this relation </score>
# <count> [remaining budget] </count>
# <choice> The 2-th relationship you select that is likely to infer the question and subquestions. </choice>
# <reason> Provide the reasons for the score you assigned to the relationship 2 for helping infer the question and subquestions. </reason>
# <score> The confidence score 0.0-1.0 to select this relationship </score>
# ...
# <count> 1 </count>
# <choice> The {{budget}}-th relationship you select that is likely to infer the question and subquestions. </choice>
# <reason> Provide the reasons for the score you assigned to the relationship {{budget}} for helping infer the question and subquestions. </reason>
# <score> The confidence score 0.0-1.0 to select this relationship </score>
# ## Example
# **Question**:
# Name the president of the country whose main spoken language was Brahui in 1980?
# **Subquestions**:
# ['Identify the country where Brahui was the main spoken language in 1980?', 'Find out who the president of that country was in 1980?', 'Research the historical context of the Brahui language and its speakers in 1980?']
# **Topic entity**:
# Brahui Language
# **Multi-hop path**:
# **Several retrieved relationships**:
# language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region; language.human_language.main_country
# **Budget**:
# 3 
# ## Output:
# ```
# <count> 3 </count>
# <choice> language.human_language.main_country </choice>
# <reason> This relationship is the most highly relevant relation in the **Several retrieved relationships**, as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980. language.human_language.main_country is highly related on the subquestion 1 'Identify the country where Brahui was the main spoken language in 1980?'.</reason>
# <score> 0.92 </score>
# <count> 2 </count>
# <choice> language.human_language.countries_spoken_in </choice>
# <reason> This relationship is the second relevant relation, as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president. language.human_language.countries_spoken_in is highly related on the subquestion 2 'Identify the country where Brahui was the main spoken language in 1980?', and can help to reason this subquestion. </reason>
# <score> 0.85 </score>
# <count> 1 </count>
# <choice> base.rosetta.languoid.parent </choice>
# <reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. It is highly related on subquestion 3 'Research the historical context of the Brahui language and its speakers in 1980?' </reason>
# <score> 0.79 </score>
# ```
# ## Input:
# **Question**:
# {question}
# **Subquestions**:
# {subquestions}
# **Topic entity**:
# {topic_entity}
# **Multi-hop path**:
# {path}
# **Several retrieved relationships**:
# {relation}
# **Budget**:
# {budget}
# ## Output:
# """


# score_candidate_entity="""
# ## Instruction:
# Assume you are a **reasoning expert**. You will receive an encyclopedic question, several sub-questions that help solve the main problem, a multi-hop evidence path proposed by a student for the reasoning problem (which needs to be extended), a new **candidate entity** that can extend this path, and relevant attribute information about the candidate entity. Your task is to carefully examine and consider what information is needed to reason about the question and sub-questions, and evaluate whether the relevant attribute information of the candidate entity can assist in deriving answers to these questions. You need to evaluate the relevance of the candidate entity to the reasoning problem and its corresponding sub-questions based on the attribute information surrounding the candidate entity, and provide a score indicating the likelihood that this candidate entity can derive the answers to the question and sub-questions.
# Noted: Please provide a score for the candidate entity's inference of the main question and its corresponding sub-questions (should be a decimal between 0 and 1) and explain your reasons. During this process, you should follow the guidelines below:
# ## Guidelines:
# 1. **Reliability of Information Surrounding the Candidate Entity**: Please carefully consider whether the attribute information surrounding the candidate entity can assist it in inferring the main question and sub-questions.
# 2. **Feasibility of Adding the Candidate Entity to the Reasoning Path**: Please thoroughly understand the semantics contained in the question and the existing evidence path, and think about whether incorporating the candidate entity into the evidence path would be helpful for the reasoning problem.
# ## Input Format:
# **Question**:
# The input question
# **Subquestions**:
# The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
# **History path**:
# entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
# **Candidate entity**:
# The candidate entity to be scored
# **Information surrounding the candidate entity**:
# entity1 r1 is entity2 \\n
# ...
# ## Output Format:
# The format of the output is the following XML:
# ```
# <entity> The same entity in **Candidate entity** </entity>
# <score> The confidence score 0.0-1.0 of this candidate entity to reason the question and the corresponding subquestions. </score>
# <reason> Provide the reasoning for the score you assigned to the candidate entity for inferring the question. </reason>
# ```
# Note that the entity within <entity> </entity> should strictly same as the **Candidate entity** in the input.
# ## Example 1
# **Question**:
# What does jamaican people speak?
# **Subquestions**:
# ['Identify the main languages spoken by Jamaican people?', 'Research the history and development of the Jamaican language?', 'Determine the percentage of Jamaican people who speak each language?']
# **History path**:
# jamaican -> location.location.nearby_airports -> Norman Manley International Airport
# **Candidate entity**:
# Norman Manley International Airport
# **Information surrounding the candidate entity**:
# Norman Manley International Airport is serves.as primary international airports.\\n
# Norman Manley International Airport is constructed in the 1960s.\\n
# ## Output:
# <entity> Norman Manley International Airport </entity>
# <score> 0.12 </score>
# <reason> The information about Norman Manley International Airport and related details are completely unrelated to jamaican's languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>
# ## Example 2
# **Question**:
# Which nation has the Alta Verapaz Department and is in Central America?
# **Subquestions**:
# ['Identify the countries located in Central America?', 'Determine which country contains the Alta Verapaz Department?', 'Confirm the geographical location of the Alta Verapaz Department within its country?']
# **History path**:
# Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
# **Candidate entity**:
# Cob\u00e1n
# **Information surrounding the candidate entity**:
# Cob\u00e1n was established by Spanish conquistadors.\\n
# Cob\u00e1n speak Spanish.\\n
# ## Output:
# <entity> Cob\u00e1n </entity>
# <score> 0.31 </score>
# <reason> The path formed by the candidate entity Cob\u00e1n indicates that the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this candidate entity successfully reasoning the question is 0.41.</reason>
# ## Example 3
# **Question**:
# Where did the \"Country Nation World Tour\" concert artist go to college?
# **Subquestions**:
# ['Identify the artist associated with the \"Country Nation World Tour\"', 'Research the educational background of the concert artist', 'Find out which colleges or universities the artist attended']
# **History path**:
# \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
# **Candidate entity**:
# Brad Paisley
# **Information surrounding the candidate entity**:
# Brad Paisley born in West Virginia, USA.\\n
# Brad Paisley is married to Kimberly.\\n
# Brad Paisley is graduated from Belmont University.\\n
# ## Output:
# <entity> Brad Paisley </entity>
# <score> 0.62 </score>
# <reason> The path formed by the candidate entity Brad Paisley indicates the identity of the performer of the \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 'Identify the artist associated with the \"Country Nation World Tour\"'. Additionally, the information surrounding the candidate entity includes details about his educational background, which will help to reason the answer. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.62. </reason>
# ## Example 4
# **Question**:
# Who is the daughter of the artist who had a concert tour called I Am... World Tour?
# **Subquestions**:
# ['Identify the artist who had the \"I Am... World Tour\"', 'Research the family of the identified artist', 'Find out if the artist has a daughter and her name']
# **History path**:
# I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
# **Candidate entity**:
# Blue Ivy
# **Information surrounding the candidate entity**:
# Blue Ivy born in USA.\\n
# Blue Ivy born on January 7, 2012.\\n
# Blue Ivy garnered attention for her artistic talents.\\n
# ## Output:
# <entity> Blue Ivy </entity>
# <score> 0.92 </score>
# <reason> The path formed by the candidate entity Blue Ivy indicates that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 'Identify the artist who had the \"I Am... World Tour\"' and subquestion 2 'Find out if the artist has a daughter and her name'. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.92. </reason>
# ## Input:
# **Question**:
# {question}
# **Subquestions**:
# {subquestions}
# **History path**:
# {history_path}
# **Candidate entity**:
# {candidate_entity}
# **Information surrounding the candidate entity**:
# {neighbor_info}
# ## Output:
# """


# reweight_value_prompt="""
# ## Instruction:
# Assume you are a **reasoning expert**. You will receive an encyclopedic question, several sub-questions that help solve the main question, related topic entities, and a multi-hop evidence path that a student has proposed for reasoning the question (the path may be incomplete or complete). Your task is to carefully review and contemplate the information needed to reason about the question and subquestions, as well as the logic contained in the multi-hop evidence path provided by the student. You need to assess the relevance of this multi-hop evidence path to reasoning about the question and its corresponding subquestions, and provide a score indicating the likelihood of this evidence path leading to the answers for the question and subquestions.
# Note: Please output the score for this evidence path to infer the question and corresponding subquestions (should be a decimal between 0 and 1), and provide your reasons. In this process, you should cfollow the guidelines below:
# ## Guidelines:
# 1. **Current Feasibility of the Path**: Please carefully consider the semantics contained in the question and subquestions, as well as the semantics in the corresponding topic entities. Evaluate whether the student's current evidence path contains the key information needed to reason the question and subquestions.
# 2. **Future Feasibility of the Path**: Please think carefully from a long-term perspective about what additional information is needed to reason about the question and corresponding subquestions. Assess whether the student's current evidence path is on the right track. Is it possible to infer the answers to the question and subquestions after several hops? Evaluate the value of this path from a long-term perspective!
# ## Input Format:
# **Question**: 
# The input question
# **Subquestions**:
# The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
# **Topic entity**: 
# The related topic entities
# **Candidate multi-hop evidence path**:
# entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
# ## Output Format
# The format of the output is the following XML:
# ```
# <path> The same path in **Candidate multi-hop evidence path**: entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... </path>
# <score> The confidence score 0.0-1.0 of this path to reason the question and the corresponding subquestions. </score>
# <reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
# ```
# Note that the path within <path> </path> should strictly same as the path in the input.
# ## Example 1
# **Question**:
# What does jamaican people speak?
# **Subquestions**:
# ['Identify the main languages spoken by Jamaican people?', 'Research the history and development of the Jamaican language?', 'Determine the percentage of Jamaican people who speak each language?']
# **Topic entity**:
# jamaican
# **Candidate multi-hop evidence path**:
# jamaican -> location.location.nearby_airports -> Norman Manley International Airport
# ## Output:
# <path> jamaican -> location.location.nearby_airports -> Norman Manley International Airport </path>
# <score> 0.15 </score>
# <reason> The airports near Jamaica are completely unrelated to its languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>
# ## Example 2
# **Question**:
# Which nation has the Alta Verapaz Department and is in Central America?
# **Subquestions**:
# ['Identify the countries located in Central America?', 'Determine which country contains the Alta Verapaz Department?', 'Confirm the geographical location of the Alta Verapaz Department within its country?']
# **Topic entity**:
# Alta Verapaz Department
# **Candidate multi-hop evidence path**:
# Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
# ## Output:
# <path> Alta Verapaz Department -> location.location.contains -> Cob\u00e1n </path>
# <score> 0.31 </score>
# <reason> This path indicates the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this path successfully reasoning the question is 0.31.</reason>
# ## Example 3
# **Question**:
# Where did the \"Country Nation World Tour\" concert artist go to college?
# **Subquestions**:
# ['Identify the artist associated with the \"Country Nation World Tour\"', 'Research the educational background of the concert artist', 'Find out which colleges or universities the artist attended']
# **Topic entity**:
# \"Country Nation World Tour\"
# **Candidate multi-hop evidence path**:
# \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
# ## Output:
# <path> \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley </path>
# <score> 0.62 </score>
# <reason> This evidence path provides the identity of the performer of \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 'Identify the artist associated with the \"Country Nation World Tour\"'. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this path successfully reasoning the question is 0.62. </reason>
# ## Example 4
# **Question**:
# Who is the daughter of the artist who had a concert tour called I Am... World Tour?
# **Subquestions**:
# ['Identify the artist who had the \"I Am... World Tour\"', 'Research the family of the identified artist', 'Find out if the artist has a daughter and her name']
# **Topic entity**:
# I Am... World Tour
# **Candidate multi-hop evidence path**:
# I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
# ## Output:
# <path> I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy </path>
# <score> 0.92 </score>
# <reason> This path shows through reasoning that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 'Identify the artist who had the \"I Am... World Tour\"' and subquestion 2 'Find out if the artist has a daughter and her name'. </reason>
# ## Input:
# **Question**:
# {question}
# **Subquestions**:
# {subquestions}
# **Topic entity**:
# {topic_entity}
# **Candidate multi-hop evidence path**:
# {candidate_path}
# ## Output:
# """
