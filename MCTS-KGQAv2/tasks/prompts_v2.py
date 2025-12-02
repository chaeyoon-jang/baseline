
prompt = """

"""



convert2path_prompt = """
{}
"""



intension_compose ="""
## Instrution:
Assume you are an **expert** in analyzing complex problems. You will receive a complex question and the related topic entities. Your task is to carefully think through and understand the information needed to answer the complex question, and consider what key information about the topic entities is required to assist in solving the problem. Based on this analysis, you should systematically break down a complex question into several smaller, simpler subquestions to better reason through and answer the original question.
## Guidelines:
1. **Clarify the required information**: Identify the key information needed to answer the question.
2. **Logical breakdown**: Divide the question into several subquestions based on logical relationships.
3. **Ensure comprehensive coverage**: Make sure the subquestions cover all the key points of the original question.
## The format of the input is:
**Question**: 
The input question
**Topic entity**: 
The related topic entities
## Please provide all the subquestions derived from the original question in the following XML format, ordered by likelihood from highest to lowest.
<subquestion> The first sub-question derived from the breakdown. </subquestion>
<subquestion> The second sub-question derived from the breakdown. </subquestion>
...
<subquestion> The N-th sub-question derived from the breakdown. </subquestion>
## Example
**Question**:
Which of the countries in the Caribbean has the smallest country calling code?
**Topic entity**:
Caribbean
## Output
```
<subquestion> Maybe we should search the countries in the Caribbean </subquestion>
<subquestion> Search the country calling code for each Caribbean country </subquestion>
<subquestion> Compare the country calling codes to find the smallest one </subquestion>
```
## Input:
**Question**:
{question}
**Topic entity**:
{topic_entity}
## Output:
"""


rank_edges_prompt ="""
## Instruction:
Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, related topic entities, a multi-hop path that helps derive the answer to the question, and a set of relationships that need to be ranked. Your task is to carefully consider the information needed to reason about the question, the semantics of the existing reasoning path, and the semantic relevance of the relationships to be ranked. Please think carefully about the likelihood of these relationships inferring the answer to the question and rank them in descending order of likelihood.
Note: If the multi-hop path is empty, do not consider the reasoning information from the multi-hop path when selecting the relationships. Please output these candidate relationships in descending order of likelihood!
## Guidelines
1. The format of the input is:
   **Question**: 
   The input question
   **Topic entity**: 
   The related topic entities
   **Multi-hop path**: 
   The multi-hop path invovle entity and relationship connected by arrows 
   **Several relationships to be ranked**: 
   A set of several retrieved relationships separated by semicolons (;), the total number of relationships is N. 
2. Please provide the ranking of all relationships in the following XML format, ordered by likelihood from highest to lowest.
Noted: The relationships you select must be consistent with that in **Several relationships to be ranked**. Please output your ranking of all relationships accurately and completely.
   <rank> 1 </rank>
   <choice> The relationship that is most likely to infer the question. </choice>
   <rank> 2 </rank>
   <choice> The 2-th relationship that is likely to infer the question. </choice>
   <rank> 3 </rank>
   <choice> The 3-th relationship that is likely to infer the question. </choice>
   ...
   <rank> N </rank>
   <choice> The N-th relationship that is likely to infer the question. </choice>
## Example
**Question**:
Name the president of the country whose main spoken language was Brahui in 1980?
**Topic entity**:
Brahui Language
**Multi-hop path**:
**Several relationships to be ranked**:
language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; base.rosetta.languoid.document; base.rosetta.languoid.local_name; language.human_language.region; language.human_language.main_country
## Output:
```
<rank> 1 </rank>
<choice> language.human_language.main_country </choice>
<rank> 2 </rank>
<choice> language.human_language.countries_spoken_in </choice>
<rank> 3 </rank>
<choice> language.human_language.language_family </choice>
<rank> 4 </rank>
<choice> language.human_language.iso_639_3_code </choice>
<rank> 5 </rank>
<choice> language.human_language.region </choice>
<rank> 6 </rank>
<choice> language.human_language.writing_system </choice>
<rank> 7 </rank>
<choice> base.rosetta.languoid.parent </choice>
<rank> 8 </rank>
<choice> base.rosetta.languoid.languoid_class </choice>
<rank> 9 </rank>
<choice> base.rosetta.languoid.local_name </choice>
<rank> 10 </rank>
<choice> base.rosetta.languoid.document </choice>
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
## Output:
"""

########################### qwen ################################################################################################################################
filter_and_score_edges_prompt = """
## Instruction:
Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, a multi-hop path that helps derive the answer to the question, and a set of several retrieved relationships that need to be filtered (which need to assist in inferring the question and the subquestions). Your task is to carefully consider the information needed to reason about the question or the subquestions, and, based on the semantics of the existing reasoning path, select the top {{budget}} relationships from the set that are most likely to help infer the answer to the question and subquestions.
Noted: If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when selecting the relationships. Please output the relationships you have selected that are most likely to infer the answer to the subquestions, along with the score (should be a decimal between 0 and 1) and your reasons for this score.
## Guidelines
1. **Clarify the required information**: Identify the key information needed to answer the question, and select the relationships most relevant to the key information.
2. **Consider the information of subquestions**: You need to carefully think about which relationships can help reason through the subquestions.
## The format of the input is:
**Question**: 
The input question
**Subquestions**:
The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
**Topic entity**: 
The related topic entities
**Multi-hop path**: 
The multi-hop path invovles entity and relationship connected by arrows 
**Several retrieved relationships**: 
A set of several retrieved relationships separated by semicolons (;)
**Budget**: 
The number of the selected relationships that are most likely to infer the result.
Noted: 
1. The relationships you select must be consistent with that in **Several retrieved relationships**. Please output relationships your select exactly as they are.
2. The number of the selected relationships are no more than {{budget}}. reset counter between <count> and </count> to {{budget}}.
3. You are allowed to select {{budget}} relationships (starting budget), keep track of it by counting down within tags <count> </count>, STOP GENERATING MORE RELATIONSHIPS when hitting 0.
4. You need to carefully think about which relationships can help derive the subquestions.
## Please provide your count, reasons, scores, and selected relationships in the following XML format:
<count> [starting budget] </count>
<choice> The relationship you select that is most likely to infer the question and subquestions. </choice>
<score> The confidence score 0.0-1.0 to select this relation </score>
<reason> Provide the reasons for the score you assigned to the relationship 1 for helping infer the question and subquestions. </reason>
<count> [remaining budget] </count>
<choice> The 2-th relationship you select that is likely to infer the question and subquestions. </choice>
<score> The confidence score 0.0-1.0 to select this relationship </score>
<reason> Provide the reasons for the score you assigned to the relationship 2 for helping infer the question and subquestions. </reason>
...
<count> 1 </count>
<choice> The {{budget}}-th relationship you select that is likely to infer the question and subquestions. </choice>
<score> The confidence score 0.0-1.0 to select this relationship </score>
<reason> Provide the reasons for the score you assigned to the relationship {{budget}} for helping infer the question and subquestions. </reason>
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
<score> 0.92 </score>
<reason> This relationship is the most highly relevant relation in the **Several retrieved relationships**, as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980. language.human_language.main_country is highly related on the subquestion 1 'Identify the country where Brahui was the main spoken language in 1980?'. </reason>
<count> 2 </count>
<choice> language.human_language.countries_spoken_in </choice>
<score> 0.85 </score>
<reason> This relationship is the second relevant relation, as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president. language.human_language.countries_spoken_in is highly related on the subquestion 2 'Identify the country where Brahui was the main spoken language in 1980?', and can help to reason this subquestion. </reason>
<count> 1 </count>
<choice> base.rosetta.languoid.parent </choice>
<score> 0.79 </score>
<reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. It is highly related on subquestion 3 'Research the historical context of the Brahui language and its speakers in 1980?' </reason>
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

score_candidate_entity="""
## Instruction:
Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main problem, a multi-hop evidence path proposed by a student for the reasoning problem (which needs to be extended), a new **candidate entity** that can extend this path, and relevant attribute information about the candidate entity. Your task is to carefully examine and consider what information is needed to reason about the question and subquestions, and evaluate whether the relevant attribute information of the candidate entity can assist in deriving answers to these questions. You need to evaluate the relevance of the candidate entity to the reasoning problem and its corresponding subquestions based on the attribute information surrounding the candidate entity, and provide a score indicating the likelihood that this candidate entity can derive the answers to the question and subquestions.
Noted: Please provide a score for the candidate entity's inference of the main question and its corresponding subquestions (should be a decimal between 0 and 1) and explain your reasons. During this process, you should follow the guidelines below:
## Guidelines:
1. **Reliability of Information Surrounding the Candidate Entity**: Please carefully consider whether the attribute information surrounding the candidate entity can assist it in inferring the main question and subquestions.
2. **Feasibility of Adding the Candidate Entity to the Reasoning Path**: Please thoroughly understand the semantics contained in the question and the existing evidence path, and think about whether incorporating the candidate entity into the evidence path would be helpful for the reasoning problem.
## Input Format:
**Question**:
The input question
**Subquestions**:
The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
**History path**:
entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
**Candidate entity**:
The candidate entity to be scored
**Information surrounding the candidate entity**:
entity1 r1 is entity2 \\n
...
## Please output your response strictly following the XML format below:
```
<entity> The same entity in **Candidate entity** </entity>
<score> The confidence score 0.0-1.0 of this candidate entity to reason the question and the corresponding subquestions. </score>
<reason> Provide the reasoning for the score you assigned to the candidate entity for inferring the question. </reason>
```
Note that the entity within <entity> </entity> should strictly same as the **Candidate entity** in the input.
## Example 1
**Question**:
What does jamaican people speak?
**Subquestions**:
['Identify the main languages spoken by Jamaican people?', 'Research the history and development of the Jamaican language?', 'Determine the percentage of Jamaican people who speak each language?']
**History path**:
jamaican -> location.location.nearby_airports -> Norman Manley International Airport
**Candidate entity**:
Norman Manley International Airport
**Information surrounding the candidate entity**:
Norman Manley International Airport is serves.as primary international airports.\\n
Norman Manley International Airport is constructed in the 1960s.\\n
## Output:
<entity> Norman Manley International Airport </entity>
<score> 0.12 </score>
<reason> The information about Norman Manley International Airport and related details are completely unrelated to jamaican's languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>
## Example 2
**Question**:
Which nation has the Alta Verapaz Department and is in Central America?
**Subquestions**:
['Identify the countries located in Central America?', 'Determine which country contains the Alta Verapaz Department?', 'Confirm the geographical location of the Alta Verapaz Department within its country?']
**History path**:
Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
**Candidate entity**:
Cob\u00e1n
**Information surrounding the candidate entity**:
Cob\u00e1n was established by Spanish conquistadors.\\n
Cob\u00e1n speak Spanish.\\n
## Output:
<entity> Cob\u00e1n </entity>
<score> 0.31 </score>
<reason> The path formed by the candidate entity Cob\u00e1n indicates that the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this candidate entity successfully reasoning the question is 0.41.</reason>
## Example 3
**Question**:
Where did the \"Country Nation World Tour\" concert artist go to college?
**Subquestions**:
['Identify the artist associated with the \"Country Nation World Tour\"', 'Research the educational background of the concert artist', 'Find out which colleges or universities the artist attended']
**History path**:
\"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
**Candidate entity**:
Brad Paisley
**Information surrounding the candidate entity**:
Brad Paisley born in West Virginia, USA.\\n
Brad Paisley is married to Kimberly.\\n
Brad Paisley is graduated from Belmont University.\\n
## Output:
<entity> Brad Paisley </entity>
<score> 0.62 </score>
<reason> The path formed by the candidate entity Brad Paisley indicates the identity of the performer of the \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 'Identify the artist associated with the \"Country Nation World Tour\"'. Additionally, the information surrounding the candidate entity includes details about his educational background, which will help to reason the answer. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.62. </reason>
## Example 4
**Question**:
Who is the daughter of the artist who had a concert tour called I Am... World Tour?
**Subquestions**:
['Identify the artist who had the \"I Am... World Tour\"', 'Research the family of the identified artist', 'Find out if the artist has a daughter and her name']
**History path**:
I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
**Candidate entity**:
Blue Ivy
**Information surrounding the candidate entity**:
Blue Ivy born in USA.\\n
Blue Ivy born on January 7, 2012.\\n
Blue Ivy garnered attention for her artistic talents.\\n
## Output:
<entity> Blue Ivy </entity>
<score> 0.92 </score>
<reason> The path formed by the candidate entity Blue Ivy indicates that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 'Identify the artist who had the \"I Am... World Tour\"' and subquestion 2 'Find out if the artist has a daughter and her name'. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.92. </reason>
## Input:
**Question**:
{question}
**Subquestions**:
{subquestions}
**History path**:
{history_path}
**Candidate entity**:
{candidate_entity}
**Information surrounding the candidate entity**:
{neighbor_info}
## Output:
"""



filter_candidate_entity="""
## Instruction:
Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main problem, a multi-hop evidence path proposed by a student for the reasoning problem (which needs to be extended), a new **candidate entity** that can extend this path, and relevant attribute information about the candidate entity. Your task is to carefully examine and consider what information is needed to reason about the question and subquestions, and evaluate whether the relevant attribute information of the candidate entity can assist in deriving answers to these questions. You need to evaluate the relevance of the candidate entity to the reasoning problem and its corresponding subquestions based on the attribute information surrounding the candidate entity, and provide a score indicating the likelihood that this candidate entity can derive the answers to the question and subquestions.
Noted: Please provide a score for the candidate entity's inference of the main question and its corresponding subquestions (should be a decimal between 0 and 1) and explain your reasons. During this process, you should follow the guidelines below:
## Guidelines:
1. **Reliability of Information Surrounding the Candidate Entity**: Please carefully consider whether the attribute information surrounding the candidate entity can assist it in inferring the main question and subquestions.
2. **Feasibility of Adding the Candidate Entity to the Reasoning Path**: Please thoroughly understand the semantics contained in the question and the existing evidence path, and think about whether incorporating the candidate entity into the evidence path would be helpful for the reasoning problem.
## Input Format:
**Question**:
The input question
**Subquestions**:
The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
**History path**:
entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
**Candidate entites with neighbor information**:
<data>
   <entity> candidate entity_1 </entity>
   <neighbor> neighbor information of the candidate entity_1 </neighbor>
</data>
<data>
   <entity> candidate entity_2 </entity>
   <neighbor> neighbor information of the candidate entity_2 </neighbor>
</data>
...
<data>
   <entity> candidate entity_n </entity>
   <neighbor> neighbor information of the candidate entity_n </neighbor>
</data>

## Please output your response strictly following the XML format below:
2. The number of the selected relationships are no more than {{budget}}. reset counter between <count> and </count> to {{budget}}.
3. You are allowed to select {{budget}} relationships (starting budget), keep track of it by counting down within tags <count> </count>, STOP GENERATING MORE RELATIONSHIPS when hitting 0.
```
<entity> The same entity in **Candidate entity** </entity>
<score> The confidence score 0.0-1.0 of this candidate entity to reason the question and the corresponding subquestions. </score>
<reason> Provide the reasoning for the score you assigned to the candidate entity for inferring the question. </reason>
```
Note that the entity within <entity> </entity> should strictly same as the **Candidate entity** in the input.
## Example 1
**Question**:
What does jamaican people speak?
**Subquestions**:
['Identify the main languages spoken by Jamaican people?', 'Research the history and development of the Jamaican language?', 'Determine the percentage of Jamaican people who speak each language?']
**History path**:
jamaican -> location.location.nearby_airports -> Norman Manley International Airport
**Candidate entity**:
Norman Manley International Airport
**Information surrounding the candidate entity**:
Norman Manley International Airport is serves.as primary international airports.\\n
Norman Manley International Airport is constructed in the 1960s.\\n
## Output:
<entity> Norman Manley International Airport </entity>
<score> 0.12 </score>
<reason> The information about Norman Manley International Airport and related details are completely unrelated to jamaican's languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>
## Example 2
**Question**:
Which nation has the Alta Verapaz Department and is in Central America?
**Subquestions**:
['Identify the countries located in Central America?', 'Determine which country contains the Alta Verapaz Department?', 'Confirm the geographical location of the Alta Verapaz Department within its country?']
**History path**:
Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
**Candidate entity**:
Cob\u00e1n
**Information surrounding the candidate entity**:
Cob\u00e1n was established by Spanish conquistadors.\\n
Cob\u00e1n speak Spanish.\\n
## Output:
<entity> Cob\u00e1n </entity>
<score> 0.31 </score>
<reason> The path formed by the candidate entity Cob\u00e1n indicates that the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this candidate entity successfully reasoning the question is 0.41.</reason>
## Example 3
**Question**:
Where did the \"Country Nation World Tour\" concert artist go to college?
**Subquestions**:
['Identify the artist associated with the \"Country Nation World Tour\"', 'Research the educational background of the concert artist', 'Find out which colleges or universities the artist attended']
**History path**:
\"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
**Candidate entity**:
Brad Paisley
**Information surrounding the candidate entity**:
Brad Paisley born in West Virginia, USA.\\n
Brad Paisley is married to Kimberly.\\n
Brad Paisley is graduated from Belmont University.\\n
## Output:
<entity> Brad Paisley </entity>
<score> 0.62 </score>
<reason> The path formed by the candidate entity Brad Paisley indicates the identity of the performer of the \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 'Identify the artist associated with the \"Country Nation World Tour\"'. Additionally, the information surrounding the candidate entity includes details about his educational background, which will help to reason the answer. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.62. </reason>
## Example 4
**Question**:
Who is the daughter of the artist who had a concert tour called I Am... World Tour?
**Subquestions**:
['Identify the artist who had the \"I Am... World Tour\"', 'Research the family of the identified artist', 'Find out if the artist has a daughter and her name']
**History path**:
I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
**Candidate entity**:
Blue Ivy
**Information surrounding the candidate entity**:
Blue Ivy born in USA.\\n
Blue Ivy born on January 7, 2012.\\n
Blue Ivy garnered attention for her artistic talents.\\n
## Output:
<entity> Blue Ivy </entity>
<score> 0.92 </score>
<reason> The path formed by the candidate entity Blue Ivy indicates that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 'Identify the artist who had the \"I Am... World Tour\"' and subquestion 2 'Find out if the artist has a daughter and her name'. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.92. </reason>
## Input:
**Question**:
{question}
**Subquestions**:
{subquestions}
**History path**:
{history_path}
**Candidate entity**:
{candidate_entity}
**Information surrounding the candidate entity**:
{neighbor_info}
## Output:
"""


reweight_value_prompt="""
## Instruction:
Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, and a multi-hop evidence path that a student has proposed for reasoning the question (the path may be incomplete or complete). Your task is to carefully review and contemplate the information needed to reason about the question and subquestions, as well as the logic contained in the multi-hop evidence path provided by the student. You need to assess the relevance of this multi-hop evidence path to reasoning about the question and its corresponding subquestions, and provide a score indicating the likelihood of this evidence path leading to the answers for the question and subquestions.
Note: Please output the score for this evidence path to infer the question and corresponding subquestions (should be a decimal between 0 and 1), and provide your reasons. In this process, you should cfollow the guidelines below:
## Guidelines:
1. **Current Feasibility of the Path**: Please carefully consider the semantics contained in the question and subquestions, as well as the semantics in the corresponding topic entities. Evaluate whether the student's current evidence path contains the key information needed to reason the question and subquestions.
2. **Future Feasibility of the Path**: Please think carefully from a long-term perspective about what additional information is needed to reason about the question and corresponding subquestions. Assess whether the student's current evidence path is on the right track. Is it possible to infer the answers to the question and subquestions after several hops? Evaluate the value of this path from a long-term perspective!
## Input Format:
**Question**: 
The input question
**Subquestions**:
The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
**Topic entity**: 
The related topic entities
**Candidate multi-hop evidence path**:
entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
## Please output your response strictly following the XML format below:
```
<path> The same path in **Candidate multi-hop evidence path**: entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... </path>
<score> The confidence score 0.0-1.0 of this path to reason the question and the corresponding subquestions. </score>
<reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
```
Noted:
1. The path within <path> </path> should strictly same as the path in the input.
2. If the entity at the end of the path starts with 'm.', such as 'm.0527d13', this entity will not be the final answer, and the score should not be very high, as further deductions are required.
## Example 1
**Question**:
What does jamaican people speak?
**Subquestions**:
['Identify the main languages spoken by Jamaican people?', 'Research the history and development of the Jamaican language?', 'Determine the percentage of Jamaican people who speak each language?']
**Topic entity**:
jamaican
**Candidate multi-hop evidence path**:
jamaican -> location.location.nearby_airports -> Norman Manley International Airport
## Output:
<path> jamaican -> location.location.nearby_airports -> Norman Manley International Airport </path>
<score> 0.15 </score>
<reason> The airports near Jamaica are completely unrelated to its languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>
## Example 2
**Question**:
Which nation has the Alta Verapaz Department and is in Central America?
**Subquestions**:
['Identify the countries located in Central America?', 'Determine which country contains the Alta Verapaz Department?', 'Confirm the geographical location of the Alta Verapaz Department within its country?']
**Topic entity**:
Alta Verapaz Department
**Candidate multi-hop evidence path**:
Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
## Output:
<path> Alta Verapaz Department -> location.location.contains -> Cob\u00e1n </path>
<score> 0.31 </score>
<reason> This path indicates the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this path successfully reasoning the question is 0.31.</reason>
## Example 3
**Question**:
Where did the \"Country Nation World Tour\" concert artist go to college?
**Subquestions**:
['Identify the artist associated with the \"Country Nation World Tour\"', 'Research the educational background of the concert artist', 'Find out which colleges or universities the artist attended']
**Topic entity**:
\"Country Nation World Tour\"
**Candidate multi-hop evidence path**:
\"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
## Output:
<path> \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley </path>
<score> 0.62 </score>
<reason> This evidence path provides the identity of the performer of \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 'Identify the artist associated with the \"Country Nation World Tour\"'. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this path successfully reasoning the question is 0.62. </reason>
## Example 4
**Question**:
Who is the daughter of the artist who had a concert tour called I Am... World Tour?
**Subquestions**:
['Identify the artist who had the \"I Am... World Tour\"', 'Research the family of the identified artist', 'Find out if the artist has a daughter and her name']
**Topic entity**:
I Am... World Tour
**Candidate multi-hop evidence path**:
I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
## Output:
<path> I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy </path>
<score> 0.92 </score>
<reason> This path shows through reasoning that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 'Identify the artist who had the \"I Am... World Tour\"' and subquestion 2 'Find out if the artist has a daughter and her name'. </reason>
## Input:
**Question**:
{question}
**Subquestions**:
{subquestions}
**Topic entity**:
{topic_entity}
**Candidate multi-hop evidence path**:
{candidate_path}
## Output:
"""



answer_generate_prompt="""
## Instruction:
Assume you are a **reasoning expert** in analyzing complex questions. You will receive a complex question, several subquestions related to the original question, relevant topic entities, a serise of known correct historical reasoning paths, and a multi-hop evidence path that a student has proposed for reasoning the question. Your task is to carefully review the information needed to reason through the question or subquestions, as well as the logic and reasoning information contained in the student's multi-hop evidence path. You need to determine whether the existing path information allows you to answer the question (the entity at the end of the path being the answer to the inferred question).
Note: Please provide your final judgment, outputting "Yes" or "No," along with the reasoning for your judgment. During this process, you should follow the guidelines below:
## Guidelines:
1. **Completeness of the original question solution**: Please carefully consider the semantics contained in the question. Evaluate whether the student's current evidence path includes the key information needed to reason through the questions. Can the entity at the end of the path serve as the final answer?
2. **Completeness of the subquestions solution**: Please carefully consider the semantics contained in the series of subquestions. Evaluate whether the student's current evidence path includes the key information needed to reason through the subquestions. Can the entity at the end of the path serve as the final answer?
## Input Format:
**Question**:
The input question
**Subquestions**:
The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
**Topic entity**:
The related topic entities
**Historical paths**: 
The known correct historical reasoning paths.
**Student's multi-hop evidence path**:
entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
## Please output your response strictly following the XML format below:
```
<response> Your judgment on whether the path in **Student's multi-hop evidence path** can answer the question: "Yes" or "No" </response>
<reason> Provide the reasons for your judgment. </reason>
```
Noted: The output in between <response> and </response> must be "Yes" or "No".
## Example 1
**Question**:
who did michael j fox marry?
**Subquestions**:
["What is the name of Michael J. Fox's current spouse?", "Has Michael J. Fox been married more than once? If so, who were his previous spouses?", "When did Michael J. Fox get married to his current or most recent spouse?"]
**Topic entity**:
Michael J. Fox
**Historical path**: 
**Student's multi-hop evidence path**:
Michael J. Fox -> people.person.spouse_s -> m.02kkmrn -> people.marriage.spouse -> Tracy Pollan
## Output:
```
<response> Yes </response>
<reason> The provided path follows a logical sequence to infer the answer: 1. Michael J. Fox is linked to m.02kkmrn via the relationship people.person.spouse_s, indicating a spouse. 2. m.02kkmrn is linked to Tracy Pollan via the relationship people.marriage.spouse, confirming that Tracy Pollan is the spouse of Michael J. Fox. The final entity in the path, Tracy Pollan, directly answers the question "Who did Michael J. Fox marry?" Therefore, the path successfully infers the correct answer.  </reason>
```
## Example 2
**Question**:
what language do the maasai tribe speak?
**Subquestions**:
["What is the primary language spoken by the Maasai people?", "Are there any secondary languages spoken by the Maasai people?", "How widespread is the primary language among the Maasai people?"]
**Topic entity**:
Maasai people
**Historical path**: 
**Student's multi-hop evidence path**:
Maasai people -> base.ontologies.ontology_instance.equivalent_instances -> m.09dvqxc
## Output:
<response> No </response>
<reason> The provided path does not contain sufficient information to infer the language spoken by the Maasai tribe. Here's why: The path links Maasai people to m.09dvqxc via the relationship base.ontologies.ontology_instance.equivalent_instances, which suggests that m.09dvqxc is an equivalent instance of the Maasai people. However, the path does not provide any explicit information about the language spoken by the Maasai tribe. Without a relationship or entity that directly or indirectly connects to a language (e.g., "speaks" or "language"), the path cannot answer the question.To infer the language, the path would need to include a connection to a language entity, such as Maa (the language spoken by the Maasai). Since this is missing, the path cannot answer the question.</reason>
## Input:
**Question**:
{question}
**Subquestions**:
{subquestions}
**Topic entity**:
{topic_entity}
**Historical paths**:
{history_info}
**Student's multi-hop evidence path**:
{candidate_path}
## Output:
"""



# During this process, you should follow the guidelines below:
# ## Guidelines:
# 1. **Completeness of the original question solution**: Please carefully consider the semantics contained in the question. Evaluate whether the student's current evidence path includes the key information needed to reason through the questions. Can the entity at the end of the path serve as the final answer? or provide key information for reasoning about the question.

answer_generate_promptv2="""
## Instruction:
Assume you are a **reasoning expert** in analyzing complex questions. You will receive a complex question, relevant topic entities and a multi-hop evidence path that a student has proposed for reasoning the question. Your task is to determine whether the path in **Student's multi-hop evidence path** can help you to answer the question or provide information for reasoning about the question (the entity at the end of the path being the answer to the inferred question).
Note: Please provide your final judgment, outputting "Yes" or "No," along with the reasoning for your judgment. 
## Input Format:
**Question**:
The input question
**Topic entity**:
The related topic entities
**Student's multi-hop evidence path**:
entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
## Please output your response strictly following the XML format below:
```
<response> Your judgment on whether the path in **Student's multi-hop evidence path** can answer the question or provide key information for reasoning about the question: "Yes" or "No" </response>
<reason> Provide the reasons for your judgment. </reason>
```
Noted: The output in between <response> and </response> must be "Yes" or "No".
## Example 1
**Question**:
who did michael j fox marry?
**Topic entity**:
Michael J. Fox
**Student's multi-hop evidence path**:
Michael J. Fox -> people.person.spouse_s -> m.02kkmrn -> people.marriage.spouse -> Tracy Pollan
## Output:
```
<response> Yes </response>
<reason> The provided path follows a logical sequence to infer the answer: 1. Michael J. Fox is linked to m.02kkmrn via the relationship people.person.spouse_s, indicating a spouse. 2. m.02kkmrn is linked to Tracy Pollan via the relationship people.marriage.spouse, confirming that Tracy Pollan is the spouse of Michael J. Fox. The final entity in the path, Tracy Pollan, directly answers the question "Who did Michael J. Fox marry?" Therefore, the path successfully infers the correct answer.  </reason>
```
## Example 2
**Question**:
what language do the maasai tribe speak?
**Topic entity**:
Maasai people
**Student's multi-hop evidence path**:
Maasai people -> base.ontologies.ontology_instance.equivalent_instances -> m.09dvqxc
## Output:
<response> No </response>
<reason> The provided path does not contain sufficient information to infer the language spoken by the Maasai tribe. Here's why: The path links Maasai people to m.09dvqxc via the relationship base.ontologies.ontology_instance.equivalent_instances, which suggests that m.09dvqxc is an equivalent instance of the Maasai people. However, the path does not provide any explicit information about the language spoken by the Maasai tribe. Without a relationship or entity that directly or indirectly connects to a language (e.g., "speaks" or "language"), the path cannot answer the question.To infer the language, the path would need to include a connection to a language entity, such as Maa (the language spoken by the Maasai). Since this is missing, the path cannot answer the question. </reason>
## Example 3
**Question**:
who is keyshia cole dad? 
**Topic entity**:
keyshia cole
**Student's multi-hop evidence path**:
Keyshia Cole -> people.person.parents -> Francine Lons -> common.topic.notable_types -> Person -> freebase.type_profile.strict_included_types -> Deceased Person -> common.topic.notable_types -> Sal Gibson
## Output:
<response> Yes </response>
<reason> Keyshia Cole: American singer. people.person.parents: Indicates a parent-child relationship. Francine Lons: Keyshia Cole's mother. common.topic.notable_types -> Person: Indicates that Francine Lons is a person. freebase.type_profile.strict_included_types -> Deceased Person: Indicates that Francine Lons is deceased. common.topic.notable_types -> Sal Gibson: Indicates that Sal Gibson is related to Francine Lons, likely as her spouse or partner. From the path, it can be inferred that: Sal Gibson is Keyshia Cole's father. </reason>

## Input:
**Question**:
{question}
**Topic entity**:
{topic_entity}
**Student's multi-hop evidence path**:
{candidate_path}
## Output:
"""


answer_generate_promptv3="""
## Instruction:
Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, and a multi-hop evidence path that a student has proposed for reasoning the question (the path may be incomplete or complete). Your task is to carefully review and contemplate the information needed to reason about the question and subquestions, as well as the logic contained in the multi-hop evidence path provided by the student. You need to assess the relevance of this multi-hop evidence path to reasoning about the question and its corresponding subquestions, and provide a score indicating the likelihood of this evidence path leading to the answers for the question and subquestions.
Note: Please output the score for this evidence path to infer the question and corresponding subquestions (should be a decimal between 0 and 1), and provide your reasons. In this process, you should cfollow the guidelines below:
## Guidelines:
1. **Current Feasibility of the Path**: Please carefully consider the semantics contained in the question and subquestions, as well as the semantics in the corresponding topic entities. Evaluate whether the student's current evidence path contains the key information needed to reason the question and subquestions.
2. **Future Feasibility of the Path**: Please think carefully from a long-term perspective about what additional information is needed to reason about the question and corresponding subquestions. Assess whether the student's current evidence path is on the right track. Is it possible to infer the answers to the question and subquestions after several hops? Evaluate the value of this path from a long-term perspective!
## Input Format:
**Question**: 
The input question
**Subquestions**:
The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
**Topic entity**: 
The related topic entities
**Candidate multi-hop evidence path**:
entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
## Please output your response strictly following the XML format below:
```
<path> The same path in **Candidate multi-hop evidence path**: entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... </path>
<score> The confidence score 0.0-1.0 of this path to reason the question and the corresponding subquestions. </score>
<reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
```
Noted:
1. The path within <path> </path> should strictly same as the path in the input.
2. If the entity at the end of the path starts with 'm.', such as 'm.0527d13', this entity will not be the final answer, and the score should not be very high, as further deductions are required.
## Example 1
**Question**:
What does jamaican people speak?
**Subquestions**:
['Identify the main languages spoken by Jamaican people?', 'Research the history and development of the Jamaican language?', 'Determine the percentage of Jamaican people who speak each language?']
**Topic entity**:
jamaican
**Candidate multi-hop evidence path**:
jamaican -> location.location.nearby_airports -> Norman Manley International Airport
## Output:
<path> jamaican -> location.location.nearby_airports -> Norman Manley International Airport </path>
<score> 0.15 </score>
<reason> The airports near Jamaica are completely unrelated to its languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>
## Example 2
**Question**:
Which nation has the Alta Verapaz Department and is in Central America?
**Subquestions**:
['Identify the countries located in Central America?', 'Determine which country contains the Alta Verapaz Department?', 'Confirm the geographical location of the Alta Verapaz Department within its country?']
**Topic entity**:
Alta Verapaz Department
**Candidate multi-hop evidence path**:
Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
## Output:
<path> Alta Verapaz Department -> location.location.contains -> Cob\u00e1n </path>
<score> 0.31 </score>
<reason> This path indicates the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this path successfully reasoning the question is 0.31.</reason>
## Example 3
**Question**:
Where did the \"Country Nation World Tour\" concert artist go to college?
**Subquestions**:
['Identify the artist associated with the \"Country Nation World Tour\"', 'Research the educational background of the concert artist', 'Find out which colleges or universities the artist attended']
**Topic entity**:
\"Country Nation World Tour\"
**Candidate multi-hop evidence path**:
\"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
## Output:
<path> \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley </path>
<score> 0.62 </score>
<reason> This evidence path provides the identity of the performer of \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 'Identify the artist associated with the \"Country Nation World Tour\"'. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this path successfully reasoning the question is 0.62. </reason>
## Example 4
**Question**:
Who is the daughter of the artist who had a concert tour called I Am... World Tour?
**Subquestions**:
['Identify the artist who had the \"I Am... World Tour\"', 'Research the family of the identified artist', 'Find out if the artist has a daughter and her name']
**Topic entity**:
I Am... World Tour
**Candidate multi-hop evidence path**:
I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
## Output:
<path> I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy </path>
<score> 0.92 </score>
<reason> This path shows through reasoning that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 'Identify the artist who had the \"I Am... World Tour\"' and subquestion 2 'Find out if the artist has a daughter and her name'. </reason>
## Input:
**Question**:
{question}
**Subquestions**:
{subquestions}
**Topic entity**:
{topic_entity}
**Candidate multi-hop evidence path**:
{candidate_path}
## Output:
"""



answer_generate_promptv3="""
## Instruction:
Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, and a multi-hop evidence path that a student has proposed for reasoning the question. Your task is to carefully review and contemplate the information needed to reason about the question and subquestions, as well as the logic contained in the multi-hop evidence path provided by the student. You need to assess the relevance of this multi-hop evidence path to reasoning about the question and its corresponding subquestions, and provide a score indicating the likelihood of this evidence path leading to the answers for the question and subquestions.
Note: Please output the score for this evidence path to infer the question and corresponding subquestions (should be a decimal between 0 and 1), and provide your reasons. In this process, you should cfollow the guidelines below:
## Guidelines:
1. **Current Feasibility of the Path**: Please carefully consider the semantics contained in the question and subquestions, as well as the semantics in the corresponding topic entities. Evaluate whether the student's current evidence path contains the key information needed to reason the question and subquestions.
2. **Future Feasibility of the Path**: Please think carefully from a long-term perspective about what additional information is needed to reason about the question and corresponding subquestions. Assess whether the student's current evidence path is on the right track. Is it possible to infer the answers to the question and subquestions after several hops? Evaluate the value of this path from a long-term perspective!
## Input Format:
**Question**: 
The input question
**Subquestions**:
The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
**Topic entity**: 
The related topic entities
**Candidate multi-hop evidence path**:
entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
## Please output your response strictly following the XML format below:
```
<path> The same path in **Candidate multi-hop evidence path**: entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... </path>
<score> The confidence score 0.0-1.0 of this path to reason the question and the corresponding subquestions. </score>
<reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
```
Noted:
1. The path within <path> </path> should strictly same as the path in the input.
2. If the entity at the end of the path starts with 'm.', such as 'm.0527d13', this entity will not be the final answer, and the score should not be very high, as further deductions are required.
## Example 1
**Question**:
What does jamaican people speak?
**Subquestions**:
['Identify the main languages spoken by Jamaican people?', 'Research the history and development of the Jamaican language?', 'Determine the percentage of Jamaican people who speak each language?']
**Topic entity**:
jamaican
**Candidate multi-hop evidence path**:
jamaican -> location.location.nearby_airports -> Norman Manley International Airport
## Output:
<path> jamaican -> location.location.nearby_airports -> Norman Manley International Airport </path>
<score> 0.15 </score>
<reason> The airports near Jamaica are completely unrelated to its languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>
## Example 2
**Question**:
Which nation has the Alta Verapaz Department and is in Central America?
**Subquestions**:
['Identify the countries located in Central America?', 'Determine which country contains the Alta Verapaz Department?', 'Confirm the geographical location of the Alta Verapaz Department within its country?']
**Topic entity**:
Alta Verapaz Department
**Candidate multi-hop evidence path**:
Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
## Output:
<path> Alta Verapaz Department -> location.location.contains -> Cob\u00e1n </path>
<score> 0.31 </score>
<reason> This path indicates the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this path successfully reasoning the question is 0.31.</reason>
## Example 3
**Question**:
Where did the \"Country Nation World Tour\" concert artist go to college?
**Subquestions**:
['Identify the artist associated with the \"Country Nation World Tour\"', 'Research the educational background of the concert artist', 'Find out which colleges or universities the artist attended']
**Topic entity**:
\"Country Nation World Tour\"
**Candidate multi-hop evidence path**:
\"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
## Output:
<path> \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley </path>
<score> 0.62 </score>
<reason> This evidence path provides the identity of the performer of \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 'Identify the artist associated with the \"Country Nation World Tour\"'. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this path successfully reasoning the question is 0.62. </reason>
## Example 4
**Question**:
Who is the daughter of the artist who had a concert tour called I Am... World Tour?
**Subquestions**:
['Identify the artist who had the \"I Am... World Tour\"', 'Research the family of the identified artist', 'Find out if the artist has a daughter and her name']
**Topic entity**:
I Am... World Tour
**Candidate multi-hop evidence path**:
I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
## Output:
<path> I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy </path>
<score> 0.92 </score>
<reason> This path shows through reasoning that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 'Identify the artist who had the \"I Am... World Tour\"' and subquestion 2 'Find out if the artist has a daughter and her name'. </reason>
## Input:
**Question**:
{question}
**Subquestions**:
{subquestions}
**Topic entity**:
{topic_entity}
**Candidate multi-hop evidence path**:
{candidate_path}
## Output:
"""



answer_generate_promptv4="""
## Instruction:
Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, and a series of candidate reasoning paths proposed by students. Your task is to carefully review and contemplate the information needed to reason about the question and subquestions, as well as the logic contained in the multi-hop evidence path provided by the student. You need to assess the relevance of this multi-hop evidence path to reasoning about the question and its corresponding subquestions, and select several evidence paths that you believe are most likely to lead to the answers for the question and subquestions and provide your reasons for your selection.
Note: You must select several paths from the **Candidate multi-hop evidence paths** that are relatively more likely to lead to the answer through reasoning. In this process, you should follow the guidelines below:
## Guidelines:
1. The paths you select must be exactly the same as the original paths in **Candidate multi-hop evidence paths**.
## Input Format:
**Question**: 
The input question
**Subquestions**:
The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
**Topic entity**: 
The related topic entities
**Candidate multi-hop evidence paths**:
entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
entity2 -> relation1 -> entity3 -> relation4 -> entity3 \\n
...
## Please output your response strictly following the format below:
```
<path> The same path in **Candidate multi-hop evidence path** entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... </path>
<reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
<path> The same path in **Candidate multi-hop evidence path** entity2 -> relation1 -> entity3 -> relation4 -> entity3 </path>
<reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
...
```
Noted: The path within <path> </path> should strictly same as the path in the **Candidate multi-hop evidence path**.

## Example 1
**Question**:
who was richard nixon married to?
**Subquestions**:
['Who is Richard Nixon?', 'Was Richard Nixon married?', 'Who was Richard Nixon married to?']
**Topic entity**:
Richard Nixon
**Candidate multi-hop evidence path**:
The Mission Inn Hotel & Spa", "Richard Nixon -> people.person.spouse_s -> m.02h98gq -> people.marriage.location_of_ceremony -> The Mission Inn Hotel & Spa \\n
Richard Nixon -> people.person.spouse_s -> m.02h98gq -> people.marriage.type_of_union -> Marriage -> type.property.expected_type -> Spouse (or domestic partner) -> freebase.valuenotation.has_value -> Joe McCain \\n
Spouse (or domestic partner)", "Richard Nixon -> people.person.spouse_s -> m.02h98gq -> people.marriage.type_of_union -> Marriage -> rdf-schema#domain -> Spouse -> type.property.reverse_property -> Spouse (or domestic partner) \\n
Richard Nixon -> people.person.spouse_s -> m.02h98gq -> people.marriage.type_of_union -> Marriage -> type.property.expected_type -> Spouse (or domestic partner) -> type.property.reverse_property -> Spouse \\n
Richard Nixon -> people.person.spouse_s -> m.02h98gq -> people.marriage.type_of_union -> Marriage -> rdf-schema#domain -> Spouse -> rdf-schema#range -> Person \\n
Richard Nixon -> people.person.spouse_s -> m.02h98gq -> people.marriage.type_of_union -> Marriage -> type.property.expected_type -> Spouse (or domestic partner) -> freebase.valuenotation.is_reviewed -> Richard Nixon \\n
Richard Nixon -> government.us_president.vice_president -> Dwight D. Eisenhower -> people.person.profession -> Politician -> common.topic.notable_types -> Pat Nixon \\n
Richard Nixon -> people.person.spouse_s -> m.02h98gq -> people.marriage.type_of_union -> Marriage -> type.property.expected_type -> Spouse (or domestic partner) \\n
Richard Nixon -> base.kwebbase.kwtopic.connections_from -> richard milhous nixon vice-president to dwight david eisenhower -> base.kwebbase.kwtopic.connections_to -> Dwight D. Eisenhower -> people.person.profession -> Politician -> common.topic.notable_types -> Pat Nixon \\n
## Output:
```
<path> Richard Nixon -> government.us_president.vice_president -> Dwight D. Eisenhower -> people.person.profession -> Politician -> common.topic.notable_types -> Pat Nixon </path>
<reason> This path, compared to other paths, is more likely to lead to the final answer. </reason>
<path> Richard Nixon -> base.kwebbase.kwtopic.connections_from -> richard milhous nixon vice-president to dwight david eisenhower -> base.kwebbase.kwtopic.connections_to -> Dwight D. Eisenhower -> people.person.profession -> Politician -> common.topic.notable_types -> Pat Nixon </path>
<reason> This path, compared to other paths, is more likely to lead to the final answer. Because it provides information about Richard Nixon's spouse. </reason>
```
## Input:
**Question**:
{question}
**Subquestions**:
{subquestions}
**Topic entity**:
{topic_entity}
**Candidate multi-hop evidence paths**:
{candidate_path}
## Output:
"""

# ########################### qwen ################################################################################################################################





# ############################ llama3 ######################################################################################################################
# filter_and_score_edges_prompt = """
# ## Instruction:
# Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, a multi-hop path that helps derive the answer to the question, and a set of several retrieved relationships that need to be filtered (which need to assist in inferring the question and the subquestions). Your task is to carefully consider the information needed to reason about the question or the subquestions, and, based on the semantics of the existing reasoning path, select the top {{budget}} relationships from the set that are most likely to help infer the answer to the question and subquestions.
# Noted: If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when selecting the relationships. Please output the relationships you have selected that are most likely to infer the answer to the subquestions, along with the score (should be a decimal between 0 and 1) and your reasons for this score.
# ## Guidelines
# 1. **Clarify the required information**: Identify the key information needed to answer the question, and select the relationships most relevant to the key information.
# 2. **Consider the information of subquestions**: You need to carefully think about which relationships can help reason through the subquestions.
# ## The Input Format is as below:
# ###Question: 
# The input question
# ###Subquestions:
# The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
# ###Topic entity: 
# The related topic entities
# ###Multi-hop path: 
# The multi-hop path invovle entity and relationship connected by arrows 
# ###Several retrieved relationships: 
# A set of several retrieved relationships separated by semicolons (;)
# ###Budget: 
# The number of the selected relationships that are most likely to infer the result.
# Noted: 
# 1. The relationships you select must be consistent with that in ###Several retrieved relationships. Please output relationships your select exactly as they are.
# 2. The number of the selected relationships are no more than {{budget}}. reset counter between <count> and </count> to {{budget}}.
# 3. You are allowed to select {{budget}} relationships (starting budget), keep track of it by counting down within tags <count> </count>, STOP GENERATING MORE RELATIONSHIPS when hitting 0.
# 4. You need to carefully think about which relationships can help reason through the subquestions.
# ## Please provide your count, reasons, scores, and selected relationships in the following XML format:
# Noted: Please output strictly in XML format, without any additional analysis.
# <count> [starting budget] </count>
# <choice> The relationship you select that is most likely to infer the question and subquestions. </choice>
# <score> The confidence score 0.0-1.0 to select this relation </score>
# <reason> Provide the reasons for the score you assigned to the relationship 1 for helping infer the question and subquestions. </reason>
# <count> [remaining budget] </count>
# <choice> The 2-th relationship you select that is likely to infer the question and subquestions. </choice>
# <score> The confidence score 0.0-1.0 to select this relationship </score>
# <reason> Provide the reasons for the score you assigned to the relationship 2 for helping infer the question and subquestions. </reason>
# ...
# <count> 1 </count>
# <choice> The {{budget}}-th relationship you select that is likely to infer the question and subquestions. </choice>
# <score> The confidence score 0.0-1.0 to select this relationship </score>
# <reason> Provide the reasons for the score you assigned to the relationship {{budget}} for helping infer the question and subquestions. </reason>
# ## Example
# ###Question:
# Name the president of the country whose main spoken language was Brahui in 1980?
# ###Subquestions:
# ['Identify the country where Brahui was the main spoken language in 1980?', 'Find out who the president of that country was in 1980?', 'Research the historical context of the Brahui language and its speakers in 1980?']
# ###Topic entity:
# Brahui Language
# ###Multi-hop path:
# ###Several retrieved relationships:
# language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region; language.human_language.main_country
# ###Budget:
# 3 
# ## Output:
# ```
# <count> 3 </count>
# <choice> language.human_language.main_country </choice>
# <score> 0.92 </score>
# <reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. It is highly related on subquestion 3 'Research the historical context of the Brahui language and its speakers in 1980?' </reason>
# <count> 2 </count>
# <choice> language.human_language.countries_spoken_in </choice>
# <score> 0.85 </score>
# <reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. It is highly related on subquestion 3 'Research the historical context of the Brahui language and its speakers in 1980?' </reason>
# <count> 1 </count>
# <choice> base.rosetta.languoid.parent </choice>
# <score> 0.79 </score>
# <reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. It is highly related on subquestion 3 'Research the historical context of the Brahui language and its speakers in 1980?' </reason>
# ```
# ## Input:
# ###Question:
# {question}
# ###Subquestions:
# {subquestions}
# ###Topic entity:
# {topic_entity}
# ###Multi-hop path:
# {path}
# ###Several retrieved relationships:
# {relation}
# ###Budget:
# {budget}
# ## Output:
# """

# score_candidate_entity="""
# ## Instruction:
# Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main problem, a multi-hop evidence path proposed by a student for the reasoning problem (which needs to be extended), a new **candidate entity** that can extend this path, and relevant attribute information about the candidate entity. Your task is to carefully examine and consider what information is needed to reason about the question and subquestions, and evaluate whether the relevant attribute information of the candidate entity can assist in deriving answers to these questions. You need to evaluate the relevance of the candidate entity to the reasoning problem and its corresponding subquestions based on the attribute information surrounding the candidate entity, and provide a score indicating the likelihood that this candidate entity can derive the answers to the question and subquestions.
# Noted: Please provide a score for the candidate entity's inference of the main question and its corresponding subquestions (should be a decimal between 0 and 1) and explain your reasons. During this process, you should follow the guidelines below:
# ## Guidelines:
# 1. **Reliability of Information Surrounding the Candidate Entity**: Please carefully consider whether the attribute information surrounding the candidate entity can assist it in inferring the main question and subquestions.
# 2. **Feasibility of Adding the Candidate Entity to the Reasoning Path**: Please thoroughly understand the semantics contained in the question and the existing evidence path, and think about whether incorporating the candidate entity into the evidence path would be helpful for the reasoning problem.
# ## The Input Format is as below:
# ###Question:
# The input question
# ###Subquestions:
# The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
# ###History path:
# entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
# ###Candidate entity:
# The candidate entity to be scored
# ###Information surrounding the candidate entity:
# entity1 r1 is entity2 \\n
# ...
# ## Please output your response strictly following the XML format below:
# Noted: Please output strictly in XML format, without any additional analysis.
# ```
# <entity> The same entity in **Candidate entity** </entity>
# <score> The confidence score 0.0-1.0 of this candidate entity to reason the question and the corresponding subquestions. </score>
# <reason> Provide the reasoning for the score you assigned to the candidate entity for inferring the question. </reason>
# ```
# Note that the entity within <entity> </entity> should strictly same as the **Candidate entity** in the input.
# ## Example 1
# ###Question:
# What does jamaican people speak?
# ###Subquestions:
# ['Identify the main languages spoken by Jamaican people?', 'Research the history and development of the Jamaican language?', 'Determine the percentage of Jamaican people who speak each language?']
# ###History path:
# jamaican -> location.location.nearby_airports -> Norman Manley International Airport
# ###Candidate entity:
# Norman Manley International Airport
# ###Information surrounding the candidate entity:
# Norman Manley International Airport is serves.as primary international airports.\\n
# Norman Manley International Airport is constructed in the 1960s.\\n
# ## Output:
# <entity> Norman Manley International Airport </entity>
# <score> 0.12 </score>
# <reason> The information about Norman Manley International Airport and related details are completely unrelated to jamaican's languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>
# ## Example 2
# ###Question:
# Which nation has the Alta Verapaz Department and is in Central America?
# ###Subquestions:
# ['Identify the countries located in Central America?', 'Determine which country contains the Alta Verapaz Department?', 'Confirm the geographical location of the Alta Verapaz Department within its country?']
# ###History path:
# Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
# ###Candidate entity:
# Cob\u00e1n
# ###Information surrounding the candidate entity:
# Cob\u00e1n was established by Spanish conquistadors.\\n
# Cob\u00e1n speak Spanish.\\n
# ## Output:
# <entity> Cob\u00e1n </entity>
# <score> 0.31 </score>
# <reason> The path formed by the candidate entity Cob\u00e1n indicates that the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this candidate entity successfully reasoning the question is 0.41.</reason>
# ## Example 3
# ###Question:
# Where did the \"Country Nation World Tour\" concert artist go to college?
# ###Subquestions:
# ['Identify the artist associated with the \"Country Nation World Tour\"', 'Research the educational background of the concert artist', 'Find out which colleges or universities the artist attended']
# ###History path:
# \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
# ###Candidate entity:
# Brad Paisley
# ###Information surrounding the candidate entity:
# Brad Paisley born in West Virginia, USA.\\n
# Brad Paisley is married to Kimberly.\\n
# Brad Paisley is graduated from Belmont University.\\n
# ## Output:
# <entity> Brad Paisley </entity>
# <score> 0.62 </score>
# <reason> The path formed by the candidate entity Brad Paisley indicates the identity of the performer of the \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 'Identify the artist associated with the \"Country Nation World Tour\"'. Additionally, the information surrounding the candidate entity includes details about his educational background, which will help to reason the answer. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.62. </reason>
# ## Example 4
# ###Question:
# Who is the daughter of the artist who had a concert tour called I Am... World Tour?
# ###Subquestions:
# ['Identify the artist who had the \"I Am... World Tour\"', 'Research the family of the identified artist', 'Find out if the artist has a daughter and her name']
# ###History path:
# I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
# ###Candidate entity:
# Blue Ivy
# ###Information surrounding the candidate entity:
# Blue Ivy born in USA.\\n
# Blue Ivy born on January 7, 2012.\\n
# Blue Ivy garnered attention for her artistic talents.\\n
# ## Output:
# <entity> Blue Ivy </entity>
# <score> 0.92 </score>
# <reason> The path formed by the candidate entity Blue Ivy indicates that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 'Identify the artist who had the \"I Am... World Tour\"' and subquestion 2 'Find out if the artist has a daughter and her name'. Therefore, I believe the score of this candidate entity successfully reasoning the question is 0.92. </reason>
# ## Input:
# ###Question:
# {question}
# ###Subquestions:
# {subquestions}
# ###History path:
# {history_path}
# ###Candidate entity:
# {candidate_entity}
# ###Information surrounding the candidate entity**:
# {neighbor_info}
# ## Output:
# """


# reweight_value_prompt="""
# ## Instruction:
# Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, and a multi-hop evidence path that a student has proposed for reasoning the question (the path may be incomplete or complete). Your task is to carefully review and contemplate the information needed to reason about the question and subquestions, as well as the logic contained in the multi-hop evidence path provided by the student. You need to assess the relevance of this multi-hop evidence path to reasoning about the question and its corresponding subquestions, and provide a score indicating the likelihood of this evidence path leading to the answers for the question and subquestions.
# Note: Please output the score for this evidence path to infer the question and corresponding subquestions (should be a decimal between 0 and 1), and provide your reasons. In this process, you should cfollow the guidelines below:
# ## Guidelines:
# 1. **Current Feasibility of the Path**: Please carefully consider the semantics contained in the question and subquestions, as well as the semantics in the corresponding topic entities. Evaluate whether the student's current evidence path contains the key information needed to reason the question and subquestions.
# 2. **Future Feasibility of the Path**: Please think carefully from a long-term perspective about what additional information is needed to reason about the question and corresponding subquestions. Assess whether the student's current evidence path is on the right track. Is it possible to infer the answers to the question and subquestions after several hops? Evaluate the value of this path from a long-term perspective!
# ## The Input Format is as below:
# ###Question: 
# The input question
# ###Subquestions:
# The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
# ###Topic entity: 
# The related topic entities
# ###Candidate multi-hop evidence path:
# entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
# ## Please output your response strictly following the XML format below:
# Noted: Please output strictly in XML format, without any additional analysis.
# ```
# <path> The same path in **Candidate multi-hop evidence path**: entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... </path>
# <score> The confidence score 0.0-1.0 of this path to reason the question and the corresponding subquestions. </score>
# <reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
# ```
# Note that the path within <path> </path> should strictly same as the path in the input.
# ## Example 1
# ###Question:
# What does jamaican people speak?
# ###Subquestions:
# ['Identify the main languages spoken by Jamaican people?', 'Research the history and development of the Jamaican language?', 'Determine the percentage of Jamaican people who speak each language?']
# ###Topic entity:
# jamaican
# ###Candidate multi-hop evidence path:
# jamaican -> location.location.nearby_airports -> Norman Manley International Airport
# ## Output:
# <path> jamaican -> location.location.nearby_airports -> Norman Manley International Airport </path>
# <score> 0.15 </score>
# <reason> The airports near Jamaica are completely unrelated to its languages and do not help in answering the question or subquestions, while the subsequent path also makes it difficult to derive an answer. </reason>
# ## Example 2
# ###Question:
# Which nation has the Alta Verapaz Department and is in Central America?
# ###Subquestions:
# ['Identify the countries located in Central America?', 'Determine which country contains the Alta Verapaz Department?', 'Confirm the geographical location of the Alta Verapaz Department within its country?']
# ###Topic entity:
# Alta Verapaz Department
# ###Candidate multi-hop evidence path:
# Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
# ## Output:
# <path> Alta Verapaz Department -> location.location.contains -> Cob\u00e1n </path>
# <score> 0.31 </score>
# <reason> This path indicates the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. And it do not help in answering the question or subquestions. So, I believe the score of this path successfully reasoning the question is 0.31.</reason>
# ## Example 3
# ###Question:
# Where did the \"Country Nation World Tour\" concert artist go to college?
# ###Subquestions:
# ['Identify the artist associated with the \"Country Nation World Tour\"', 'Research the educational background of the concert artist', 'Find out which colleges or universities the artist attended']
# ###Topic entity:
# \"Country Nation World Tour\"
# ###Candidate multi-hop evidence path:
# \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
# ## Output:
# <path> \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley </path>
# <score> 0.62 </score>
# <reason> This evidence path provides the identity of the performer of \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, it contains the most of the information needed to answer subquestion 1 'Identify the artist associated with the \"Country Nation World Tour\"'. Taking one more step down this path has a high probability of revealing which college the performer attended, leading to the correct answer. Therefore, I believe the score of this path successfully reasoning the question is 0.62. </reason>
# ## Example 4
# ###Question:
# Who is the daughter of the artist who had a concert tour called I Am... World Tour?
# ###Subquestions:
# ['Identify the artist who had the \"I Am... World Tour\"', 'Research the family of the identified artist', 'Find out if the artist has a daughter and her name']
# ###Topic entity:
# I Am... World Tour
# **Candidate multi-hop evidence path**:
# I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
# ## Output:
# <path> I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy </path>
# <score> 0.92 </score>
# <reason> This path shows through reasoning that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. This evidence path contains most of the information needed to answer subquestion 1 'Identify the artist who had the \"I Am... World Tour\"' and subquestion 2 'Find out if the artist has a daughter and her name'. </reason>
# ## Input:
# ###Question:
# {question}
# ###Subquestions:
# {subquestions}
# ###Topic entity:
# {topic_entity}
# ###Candidate multi-hop evidence path:
# {candidate_path}
# ## Output:
# """
# ########################### llama3 #######################################################################################################################################



# ############################ llama3.1 ######################################################################################################################
# filter_and_score_edges_prompt = """
# ## Instruction:
# Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, a multi-hop path that helps derive the answer to the question, and a set of several retrieved relationships that need to be filtered (which need to assist in inferring the question and the subquestions). Your task is to carefully consider the information needed to reason about the question or the subquestions, and, based on the semantics of the existing reasoning path, select the top {{budget}} relationships from the set that are most likely to help infer the answer to the question and subquestions.
# Noted: If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when selecting the relationships. Please output the relationships you have selected that are most likely to infer the answer to the subquestions, along with the score (should be a decimal between 0 and 1) and your reasons for this score.
# ## Guidelines
# 1. **Clarify the required information**: Identify the key information needed to answer the question, and select the relationships most relevant to the key information.
# 2. **Consider the information of subquestions**: You need to carefully think about which relationships can help reason through the subquestions.
# ## The Input Format is as below:
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
# 1. The relationships you select must be consistent with that in ###Several retrieved relationships. Please output relationships your select exactly as they are.
# 2. The number of the selected relationships are no more than {{budget}}. reset counter between <count> and </count> to {{budget}}.
# 3. You are allowed to select {{budget}} relationships (starting budget), keep track of it by counting down within tags <count> </count>, STOP GENERATING MORE RELATIONSHIPS when hitting 0.
# 4. You need to carefully think about which relationships can help reason through the subquestions.
# ## Please provide your count, reasons, scores, and selected relationships in the following XML format:
# Noted: Please output strictly in XML format, without any additional analysis.
# <count> [starting budget] </count>
# <choice> The relationship you select that is most likely to infer the question and subquestions. </choice>
# <reason> Provide the reasons for the score you assigned to the relationship 1 for helping infer the question and subquestions. </reason>
# <score> The confidence score 0.0-1.0 to select this relationship </score>
# <count> [remaining budget] </count>
# <choice> The 2-th relationship you select that is likely to infer the question and subquestions. </choice>
# <reason> Provide the reasons for the score you assigned to the relationship 2 for helping infer the question and subquestions. </reason>
# <score> The confidence score 0.0-1.0 to select this relationship </score>
# ...
# <count> 1 </count>
# <choice> The {{budget}}-th relationship you select that is likely to infer the question and subquestions. </choice>
# <reason> Provide the reasons for the score you assigned to the relationship 2 for helping infer the question and subquestions. </reason>
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
# <reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. It is highly related on subquestion 3 'Research the historical context of the Brahui language and its speakers in 1980?' </reason>
# <score> 0.92 </score>
# <count> 2 </count>
# <choice> language.human_language.countries_spoken_in </choice>
# <reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. It is highly related on subquestion 3 'Research the historical context of the Brahui language and its speakers in 1980?' </reason>
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
# Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main problem, a multi-hop evidence path proposed by a student for the reasoning problem (which needs to be extended), a new **candidate entity** that can extend this path, and relevant attribute information about the candidate entity. Your task is to carefully examine and consider what information is needed to reason about the question and subquestions, and evaluate whether the relevant attribute information of the candidate entity can assist in deriving answers to these questions. You need to evaluate the relevance of the candidate entity to the reasoning problem and its corresponding subquestions based on the attribute information surrounding the candidate entity, and provide a score indicating the likelihood that this candidate entity can derive the answers to the question and subquestions.
# Noted: Please provide a score for the candidate entity's inference of the main question and its corresponding subquestions (should be a decimal between 0 and 1) and explain your reasons. During this process, you should follow the guidelines below:
# ## Guidelines:
# 1. **Reliability of Information Surrounding the Candidate Entity**: Please carefully consider whether the attribute information surrounding the candidate entity can assist it in inferring the main question and subquestions.
# 2. **Feasibility of Adding the Candidate Entity to the Reasoning Path**: Please thoroughly understand the semantics contained in the question and the existing evidence path, and think about whether incorporating the candidate entity into the evidence path would be helpful for the reasoning problem.
# ## The Input Format is as below:
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
# ## Please output your response strictly following the XML format below:
# Noted: Please output strictly in XML format, without any additional analysis.
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
# Assume you are a **reasoning expert**. You will receive an encyclopedic question, several subquestions that help solve the main question, related topic entities, and a multi-hop evidence path that a student has proposed for reasoning the question (the path may be incomplete or complete). Your task is to carefully review and contemplate the information needed to reason about the question and subquestions, as well as the logic contained in the multi-hop evidence path provided by the student. You need to assess the relevance of this multi-hop evidence path to reasoning about the question and its corresponding subquestions, and provide a score indicating the likelihood of this evidence path leading to the answers for the question and subquestions.
# Note: Please output the score for this evidence path to infer the question and corresponding subquestions (should be a decimal between 0 and 1), and provide your reasons. In this process, you should cfollow the guidelines below:
# ## Guidelines:
# 1. **Current Feasibility of the Path**: Please carefully consider the semantics contained in the question and subquestions, as well as the semantics in the corresponding topic entities. Evaluate whether the student's current evidence path contains the key information needed to reason the question and subquestions.
# 2. **Future Feasibility of the Path**: Please think carefully from a long-term perspective about what additional information is needed to reason about the question and corresponding subquestions. Assess whether the student's current evidence path is on the right track. Is it possible to infer the answers to the question and subquestions after several hops? Evaluate the value of this path from a long-term perspective!
# ## The Input Format is as below:
# **Question**: 
# The input question
# **Subquestions**:
# The several subquestions derived from the original question. [the subquestion 1, the subquestion 2, the subquestion 3]
# **Topic entity**: 
# The related topic entities
# **Candidate multi-hop evidence path**:
# entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n
# ## Please output your response strictly following the XML format below:
# Noted: Please output strictly in XML format, without any additional analysis.
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
# ########################### llama3.1 #######################################################################################################################################
