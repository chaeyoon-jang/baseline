
prompt = """

"""



convert2path_prompt = """
{}
"""



intension_compose ="""
## Instrution:
Assume you are an expert in reasoning about encyclopedia questions. You will receive an encyclopedia question along with the related topic entities. Your task is to carefully review and contemplate the information needed to reason through the question, especially identifying the key information required about the topic entities. Based on this, you should break down the encyclopedia question into several sub-questions.
## Guidelines:
1. The format of the input is:
   **Question**: 
   The input question
   **Topic entity**: 
   The related topic entities
2. Please provide all the sub-questions derived from the original question in the following XML format, ordered by likelihood from highest to lowest.
Noted: The relationships you select must be consistent with that in **Several relationships to be ranked**. Please output your ranking of all relationships accurately and completely.
   <subquestion> The first sub-question derived from the breakdown. </subquestion>
   <subquestion> The 2-th sub-question derived from the breakdown. </subquestion>
   ...
   <subquestion> The N-th sub-question derived from the breakdown. </subquestion>
## Example
**Question**:
Which of the countries in the Caribbean has the smallest country calling code?
**Topic entity**:
Caribbean
## Output
```
<subquestion> Search the countries in the Caribbean </subquestion>
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


filter_and_score_edges_prompt = """
## Instruction:
Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, related topic entities, a multi-hop path that helps derive the answer to the question, and a set of several retrieved relationships that need to be filtered (which need to assist in inferring the question). Your task is to carefully consider the information needed to reason about the question and the semantics of the existing reasoning path, and select the top {{budget}} relationships from the set that are most likely to help infer the answer to the question.
Noted: If the multi-hop path is empty, then do not consider the reasoning information from the multi-hop path when selecting the relationships. Please output the relationships you have selected that are most likely to infer the answer, along with the score (should be a decimal between 0 and 1) and your reasons for this score.
## Guidelines
1. The format of the input is:
   **Question**: 
   The input question
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
4. Please provide your count, reasons, scores, and selected relationships in the following XML format.
Noted: The relationships you select must be consistent with that in **Several retrieved relationships**. Please output relationships your select exactly as they are.
   <count> [starting budget] </count>
   <choice> The relationship you select that is most likely to infer the question. </choice>
   <reason> Provide the reasons for the score you assigned to the relationship 1 for helping infer the question. </reason>
   <score> The confidence score 0.0-1.0 to select this relation </score>
   <count> [remaining budget] </count>
   <choice> The 2-th relationship you select that is likely to infer the question. </choice>
   <reason> Provide the reasons for the score you assigned to the relationship 2 for helping infer the question. </reason>
   <score> The confidence score 0.0-1.0 to select this relationship </score>
   ...
   <count> 1 </count>
   <choice> The {{budget}}-th relationship you select that is likely to infer the question.</choice>
   <reason> Provide the reasons for the score you assigned to the relationship {{budget}} for helping infer the question. </reason>
   <score> The confidence score 0.0-1.0 to select this relationship </score>
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
<reason> This relationship is the most highly relevant relation in the **Several retrieved relationships**, as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980. </reason>
<score> 0.92 </score>
<count> 2 </count>
<choice> language.human_language.countries_spoken_in </choice>
<reason> This relationship is the second relevant relation, as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president. </reason>
<score> 0.85 </score>
<count> 1 </count>
<choice> base.rosetta.languoid.parent </choice>
<reason> This relationship is the third relevant relation, it still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question. </reason>
<score> 0.79 </score>
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



reweight_value_prompt="""
## Instruction:
Assume you are a **reasoning expert**. You will receive an encyclopedic question, related topic entities, and a multi-hop evidence path that a student has thought of for reasoning about the question (the path may be partial or complete). Your task is to carefully review the information needed to reason about the question and the logic contained in the multi-hop evidence path provided by the student. You need to assess the relevance of this multi-hop evidence path to reasoning about the question. 
Note: Please output the score for this evidence path to infer the question, and provide your reasons. In this process, you should carefully consider the following guidelines:

## Guidelines:
1. **Current Feasibility of the Approach**: Please carefully consider the semantics contained in the question and the corresponding topic entities, and determine whether the student's current evidence path includes the key information necessary for reasoning.
2. **Future Feasibility of the Approach**: Please think carefully from a long-term perspective about what additional information is needed to reason about the question, and whether the existing evidence path is on the correct track. Evaluate the value of this path from a long-term standpoint!

## Input Format:
**Question**: 
The input question
**Topic entity**: 
The related topic entities
**Candidate multi-hop evidence path**:
entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... \\n

## Output Format
The format of the output is the following XML:
```
<path> The same path in **Candidate multi-hop evidence path**: entity1 -> relation1 -> entity2 -> relation2 -> entity3 -> ... </path>
<score> The confidence score 0.0-1.0 of this path to reason the question. </score>
<reason> Provide the reasoning for the score you assigned to the path for inferring the question. </reason>
```
Note that the path within <path> </path> should strictly same as the path in the input.
## Example 1
**Question**:
What does jamaican people speak?
**Topic entity**:
jamaican
**Candidate multi-hop evidence path**:
jamaican -> location.location.nearby_airports -> Norman Manley International Airport
## Output:
<path> jamaican -> location.location.nearby_airports -> Norman Manley International Airport </path>
<score> 0.15 </score>
<reason> The airports near jamaican are completely unrelated to their languages, and the subsequent path also makes it difficult to derive an answer. </reason>
## Example 2
**Question**:
Which nation has the Alta Verapaz Department and is in Central America?
**Topic entity**:
Alta Verapaz Department
**Candidate multi-hop evidence path**:
Alta Verapaz Department -> location.location.contains -> Cob\u00e1n
## Output:
<path> Alta Verapaz Department -> location.location.contains -> Cob\u00e1n </path>
<score> 0.31 </score>
<reason> This path indicates the Alta Verapaz Department contains Cob\u00e1n. However, this information is not closely related to deducing the country where Alta Verapaz Department is located. So, I believe the score of this path successfully reasoning the question is 0.31.</reason>
## Example 3
**Question**:
Where did the \"Country Nation World Tour\" concert artist go to college?
**Topic entity**:
\"Country Nation World Tour\"
**Candidate multi-hop evidence path**:
\"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley
## Output:
<path> \"Country Nation World Tour\" -> music.concert_tour.artist -> Brad Paisley </path>
<score> 0.62 </score>
<reason> This evidence path provides the identity of the performer of \"Country Nation World Tour\". Although it is not sufficient to answer the question directly, taking one more step down the path has a high probability of revealing which college the performer attended, leading to the correct answer. So, I believe the score of this path successfully reasoning the question is 0.62. </reason>
## Example 4
**Question**:
Who is the daughter of the artist who had a concert tour called I Am... World Tour?
**Topic entity**:
I Am... World Tour
**Candidate multi-hop evidence path**:
I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy
## Output:
<path> I Am... World Tour -> music.artist.concert_tours -> Beyonc\u00e9 Knowles -> people.person.children -> Blue Ivy </path>
<score> 0.92 </score>
<reason> This path shows through reasoning that the artist who held the "I Am... World Tour" is Beyonc\u00e9 Knowles, and her daughter is Blue Ivy. The information contained in the path is very confident to infer the question. </reason>

## Input:
**Question**:
{question}
**Topic entity**:
{topic_entity}
**Candidate multi-hop evidence path**:
{candidate_path}
## Output:
"""
