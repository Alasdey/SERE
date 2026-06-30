from typing import Any


def pattern_prompt_inference(text: str, source: str, target: str) -> str:
    prompt = f'''
###
TEXT: {text},
EVENT X: {source},
EVENT Y: {target},
QUESTION: Determine whether there is a causal relationship between EVENT X ({source}) and EVENT Y ({target}).
###
NOTE: You should give the step-by-step reason first, and then put the final answer in JSON format: {{"pattern": THE CAUSAL PATTERN YOU FIND. IF NO PATTERN RULES ARE MET, GIVE "No"}}
###
- Step-by-Step Rules:
You must answer the following questions ONE-BY-ONE in your step-by-step reasoning:
1. Please write the two events (X and Y) in natural language.
2. Analyze and determine whether X and Y have direct causal relationship, and meet the causal pattern rule "Direct". If so, answer causal pattern as "Direct"; If not, continue to analyze.
3. Determine which indirect causal pattern given below the given input and events satisfy. Note: If X and Y have the indirect causal relationship, they must satisfy to one of the following patterns.
4. Consider whether there are mediators between events X and Y: write down other events (or entities) that relates to X, and other events (or entities) that relates to Y, and determine whether there is any intersection between the events (or entities) that relate to both events. Note: Mediators can be given explicitly from the input text. If not given, you can also use common sense to think about whether there are implicit mediators.
5. Finally, analyze all the following patterns ONE-BY-ONE to determine whether the given text and events satisfy.
###
- Pattern Rules:
    - Direct: If the text explicitly states a causal relationship between X and Y without involving any mediating event (Z), then the causal connection is "Direct." This means that X directly influences Y, or Y directly influences X, with no intermediary mentioned. You should give this pattern "Direct" in your answer.

    - Coreference of X: In the text, if an event with the same or similar meaning as the X can be found, and this similar event has a causal relationship with Y, then there is a causal relationship between X and Y. You should give this pattern "Coreference" in your answer.

    - Coreference of Y: In the text, if an event with the same or similar meaning as the Y can be found, and this similar event has a causal relationship with X, then there is a causal relationship between X and Y. You should give this pattern "Coreference" in your answer.

    - Collider: In the text, If there are one or multiple mediators (Z) that both events X and Y have causal relationship to:
        Then consider specific rule: First, it satisfies that it is related to X and Y respectively, then it satisfies that X acts on Z and Y acts on Z, i.e. X -> Z and Y -> Z. Therefore, it can be concluded that there is a causal relationship between X and Y. You should give this pattern "Collider" in your answer.

    - Fork: In text, if there are one or multiple mediators (Z) that both events X and Y have causal relationship to:
         Then consider specific rule: First, it satisfies that it is related to X and Y respectively, then it satisfies that Z acts on X and Z acts on Y, i.e. Z -> X and Z -> Y. Therefore, it can be concluded that there is a causal relationship between X and Y. You should give this pattern "Fork" in your answer.

    - Chain: In the text, if there are at least one or multiple mediators (Z) that both events X and Y have causal relationship to:
        Then consider specific rules: First, a mediator satisfies that it is related to X and Y respectively, then it satisfies that X acts on Z and Z acts on Y, i.e. they form a causal chain structure: X -> Z -> Y (or inversely, Y -> Z -> X). Then, it can be concluded that there is a causal relationship between X and Y. You should give this pattern "Chain" in your answer.
        NOTE: "Chain" is satisfied only when there is at least one mediator Z. If not, "Chain" CAN NOT be matched.
###
**VERY IMPORTANT** If one pattern rule is met, you DON'T need to analyze the remaining rules.
###
EXAMPLE 1:
TEXT: The factory’s decision to shut down immediately resulted in hundreds of workers losing their jobs.
X: shut down
Y: losing
PATTERN: Direct (The text clearly indicates a direct causal link between "factory’s decision to shut down" and "workers losing their jobs.")
EXAMPLE 2:
TEXT: The sudden resignation of the CEO triggered a sharp decline in the company’s stock prices.
X: resignation
Y: decline
PATTERN: Direct (The text explicitly states that the CEO's resignation directly triggered a sharp decline in the company's stock prices, and this causal relationship does not involve any intermediary events.)
EXAMPLE 3:
TEXT: The company was accused of negligence in maintaining its pipelines, which were found to be leaking crude oil into the river. The oil spill caused significant harm to the local ecosystem.
X: negligence in maintaining its pipelines
Y: Harm
PATTERN: Coreference ("The company was accused of negligence in maintaining its pipelines" and "pipelines leaking crude oil into the river" describe the same event in different ways. "Pipelines leaking crude oil" caused "harm to the ecosystem".)
EXAMPLE 4:
TEXT: A dispute over water rights erupted between the two regions, with accusations of illegal diversions being exchanged. The diversion of water led to severe shortages downstream.
X: dispute
Y: Illegal diversions of water
PATTERN: Coreference ("Dispute over water rights" and "illegal diversions of water" describe the same situation from different perspectives. "Illegal diversions of water" caused "severe water shortages.")
EXAMPLE 5:
TEXT: A major tech company introduced aggressive hiring policies, while a spike in tech startups also attracted talent to the industry. The resulting competition for skilled workers drove up average salaries in the tech sector.
X: aggressive hiring policies
Y: spike in tech startups
Z: Competition for skilled workers.
PATTERN: Collider ("Aggressive hiring policies" and "spike in tech startups" both increase competition for skilled workers (X -> Z, Y -> Z), which in turn drives up salaries, indirectly linking X and Y.)
EXAMPLE 6:
TEXT: A government subsidy program boosted the production of electric vehicles, while rising consumer demand also drove manufacturers to increase their output. This surge in production caused a significant strain on the supply of critical battery materials.
X: subsidy
Y: consumer demand
Z: Strain on the supply of critical battery materials
PATTERN: Collider ("Government subsidy program" increases "electric vehicle production" (X -> Z), and "rising consumer demand" also pushes manufacturers to increase production (Y -> Z). Both contribute to the strain on battery materials (Z), linking X and Y indirectly.)
EXAMPLE 7:
TEXT: A global economic slowdown led to a decline in consumer spending and a rise in unemployment rates, as businesses struggled to stay profitable.
X: decline in consumer spending
Y: rise in unemployment rates
PATTERN: Fork ("Global economic slowdown" causes both "a decline in consumer spending" (Z -> X) and "a rise in unemployment rates" (Z -> Y). This forms a Fork structure linking X and Y via Z.)
EXAMPLE 8:
TEXT: A severe drought reduced agricultural output, while also causing water shortages in urban areas, leading to widespread concern about resource sustainability.
X: reduced agricultural output
Y: water shortages
PATTERN: Fork ("Severe drought" leads to "reduced agricultural output" (Z -> X) and "water shortages in urban areas" (Z -> Y). This establishes a Fork pattern with X and Y indirectly related through Z.)
EXAMPLE 9:
TEXT: Heavy deforestation in the region caused soil erosion, which eventually led to a decline in agricultural productivity.
X: deforestation
Y: decline
Z: soil erosion
PATTERN: Chain ("Heavy deforestation" leads to "soil erosion" (X -> Z), and "soil erosion" causes "a decline in agricultural productivity" (Z -> Y). This forms a causal chain)
PATTERN 10:
TEXT: Rising global temperatures caused the melting of polar ice caps, which led to a significant increase in sea levels.
X: rising global temperatures
Y: increase in sea levels
Z: Melting of polar ice caps
PATTERN: Chain ("Rising global temperatures" result in "melting of polar ice caps" (X -> Z), and "melting of polar ice caps" leads to "increased sea levels" (Z -> Y). This satisfies the causal chain structure.)
###
**VERY IMPORTANT** You can only choose from the given patterns: ["Direct", "Coreference", "Collider", "Fork", "Chain", "No"] and cannot invent new patterns.
Give your reasoning path step-by-step (and analyze all the pattern rules ONE-BY-ONE) first and then give your final answer in JSON format: {{"pattern": THE CAUSAL PATTERN YOU FIND. IF NO PATTERN RULES ARE MET, GIVE "No"}}:
'''

    return prompt


def pattern_prompt_train_pos(text: str, source: str, target: str) -> str:
    prompt = f'''
###
TEXT: {text},
EVENT X: {source},
EVENT Y: {target},
QUESTION: There is a causal relationship between EVENT X ({source}) and EVENT Y ({target}). Please follow the following instructions to explain why there exists a causal relationship between X ({source}) and Y ({target}) based on the given text.
###
NOTE: You should give the step-by-step reason first, and then put the final answer in JSON format: {{"pattern": THE CAUSAL PATTERN YOU FIND. DO NOT answer "None" or "No".}}
###
- Step-by-Step Rules:
You must answer the following questions ONE-BY-ONE in your step-by-step reasoning:
1. Please write the two events (X and Y) in natural language.
2. Analyze and determine whether X and Y have direct causal relationship, and meet the causal pattern rule "Direct". If so, answer causal pattern as "Direct"; If not, continue to analyze.
3. Determine which indirect causal pattern given below the given input and events satisfy. Note: If X and Y have the indirect causal relationship, they must satisfy to one of the following patterns.
4. Consider whether there are mediators between events X and Y: write down other events (or entities) that relates to X, and other events (or entities) that relates to Y, and determine whether there is any intersection between the events (or entities) that relate to both events. Note: Mediators can be given explicitly from the input text. If not given, you can also use common sense to think about whether there are implicit mediators.
5. Finally, analyze all the following patterns ONE-BY-ONE to determine whether the given text and events satisfy. DO NOT answer "None" or "No".
###
- Pattern Rules:
    - Direct: If the text explicitly states a causal relationship between X and Y without involving any mediating event (Z), then the causal connection is "Direct." This means that X directly influences Y, or Y directly influences X, with no intermediary mentioned. You should give this pattern "Direct" in your answer.

    - Coreference of X: In the text, if an event with the same or similar meaning as the X can be found, and this similar event has a causal relationship with Y, then there is a causal relationship between X and Y. You should give this pattern "Coreference" in your answer.

    - Coreference of Y: In the text, if an event with the same or similar meaning as the Y can be found, and this similar event has a causal relationship with X, then there is a causal relationship between X and Y. You should give this pattern "Coreference" in your answer.

    - Collider: In the text, If there are one or multiple mediators (Z) that both events X and Y have causal relationship to:
        Then consider specific rule: First, it satisfies that it is related to X and Y respectively, then it satisfies that X acts on Z and Y acts on Z, i.e. X -> Z and Y -> Z. Therefore, it can be concluded that there is a causal relationship between X and Y. You should give this pattern "Collider" in your answer.

    - Fork: In text, if there are one or multiple mediators (Z) that both events X and Y have causal relationship to:
         Then consider specific rule: First, it satisfies that it is related to X and Y respectively, then it satisfies that Z acts on X and Z acts on Y, i.e. Z -> X and Z -> Y. Therefore, it can be concluded that there is a causal relationship between X and Y. You should give this pattern "Fork" in your answer.

    - Chain: In the text, if there are at least one or multiple mediators (Z) that both events X and Y have causal relationship to:
        Then consider specific rules: First, a mediator satisfies that it is related to X and Y respectively, then it satisfies that X acts on Z and Z acts on Y, i.e. they form a causal chain structure: X -> Z -> Y (or inversely, Y -> Z -> X). Then, it can be concluded that there is a causal relationship between X and Y. You should give this pattern "Chain" in your answer.
        NOTE: "Chain" is satisfied only when there is at least one mediator Z. If not, "Chain" CAN NOT be matched.
###
**VERY IMPORTANT** If one pattern rule is met, you DON'T need to analyze the remaining rules. DO NOT answer "None".
###
EXAMPLE 1:
TEXT: The factory’s decision to shut down immediately resulted in hundreds of workers losing their jobs.
X: shut down
Y: losing
PATTERN: Direct (The text clearly indicates a direct causal link between "factory’s decision to shut down" and "workers losing their jobs.")
EXAMPLE 2:
TEXT: The sudden resignation of the CEO triggered a sharp decline in the company’s stock prices.
X: resignation
Y: decline
PATTERN: Direct (The text explicitly states that the CEO's resignation directly triggered a sharp decline in the company's stock prices, and this causal relationship does not involve any intermediary events.)
EXAMPLE 3:
TEXT: The company was accused of negligence in maintaining its pipelines, which were found to be leaking crude oil into the river. The oil spill caused significant harm to the local ecosystem.
X: negligence in maintaining its pipelines
Y: Harm
PATTERN: Coreference ("The company was accused of negligence in maintaining its pipelines" and "pipelines leaking crude oil into the river" describe the same event in different ways. "Pipelines leaking crude oil" caused "harm to the ecosystem".)
EXAMPLE 4:
TEXT: A dispute over water rights erupted between the two regions, with accusations of illegal diversions being exchanged. The diversion of water led to severe shortages downstream.
X: dispute
Y: Illegal diversions of water
PATTERN: Coreference ("Dispute over water rights" and "illegal diversions of water" describe the same situation from different perspectives. "Illegal diversions of water" caused "severe water shortages.")
EXAMPLE 5:
TEXT: A major tech company introduced aggressive hiring policies, while a spike in tech startups also attracted talent to the industry. The resulting competition for skilled workers drove up average salaries in the tech sector.
X: aggressive hiring policies
Y: spike in tech startups
Z: Competition for skilled workers.
PATTERN: Collider ("Aggressive hiring policies" and "spike in tech startups" both increase competition for skilled workers (X -> Z, Y -> Z), which in turn drives up salaries, indirectly linking X and Y.)
EXAMPLE 6:
TEXT: A government subsidy program boosted the production of electric vehicles, while rising consumer demand also drove manufacturers to increase their output. This surge in production caused a significant strain on the supply of critical battery materials.
X: subsidy
Y: consumer demand
Z: Strain on the supply of critical battery materials
PATTERN: Collider ("Government subsidy program" increases "electric vehicle production" (X -> Z), and "rising consumer demand" also pushes manufacturers to increase production (Y -> Z). Both contribute to the strain on battery materials (Z), linking X and Y indirectly.)
EXAMPLE 7:
TEXT: A global economic slowdown led to a decline in consumer spending and a rise in unemployment rates, as businesses struggled to stay profitable.
X: decline in consumer spending
Y: rise in unemployment rates
PATTERN: Fork ("Global economic slowdown" causes both "a decline in consumer spending" (Z -> X) and "a rise in unemployment rates" (Z -> Y). This forms a Fork structure linking X and Y via Z.)
EXAMPLE 8:
TEXT: A severe drought reduced agricultural output, while also causing water shortages in urban areas, leading to widespread concern about resource sustainability.
X: reduced agricultural output
Y: water shortages
PATTERN: Fork ("Severe drought" leads to "reduced agricultural output" (Z -> X) and "water shortages in urban areas" (Z -> Y). This establishes a Fork pattern with X and Y indirectly related through Z.)
EXAMPLE 9:
TEXT: Heavy deforestation in the region caused soil erosion, which eventually led to a decline in agricultural productivity.
X: deforestation
Y: decline
Z: soil erosion
PATTERN: Chain ("Heavy deforestation" leads to "soil erosion" (X -> Z), and "soil erosion" causes "a decline in agricultural productivity" (Z -> Y). This forms a causal chain)
PATTERN 10:
TEXT: Rising global temperatures caused the melting of polar ice caps, which led to a significant increase in sea levels.
X: rising global temperatures
Y: increase in sea levels
Z: Melting of polar ice caps
PATTERN: Chain ("Rising global temperatures" result in "melting of polar ice caps" (X -> Z), and "melting of polar ice caps" leads to "increased sea levels" (Z -> Y). This satisfies the causal chain structure.)
###
**VERY IMPORTANT** You can only choose from the given patterns: ["Direct", "Coreference", "Collider", "Fork", "Chain"] and cannot invent new patterns. DO NOT answer "None" or "No".
Give your reasoning path step-by-step (and analyze all the pattern rules ONE-BY-ONE) first and then give your final answer in JSON format: {{"pattern": THE CAUSAL PATTERN YOU FIND. DO NOT answer "None" or "No".}}:
'''

    return prompt


def causal_inference_prompt_no_examples(text: str, source: str, target: str) -> str:
    """
    Same instructions/voice as predict_by_structured_examples_prompt (no pattern-rule taxonomy),
    but without fewshot examples. Used by utils/cot_synthesis.py so the reasoning it synthesizes
    for fewshot CoT matches what the prediction prompt actually teaches the model.
    """
    prompt = '''Given a text, two events (Event X and Event Y), you need to determine whether there is a causal relationship between the given events X and Y.
###
Instructions:
You should give step-by-step reasoning path before giving the final answer.
###
Text: {text};
Event X: {source};
Event Y: {target};
Give step-by-step reasoning path, and then organize the final answer in JSON format: {{"Answer": "Your answer, the answer must be either 'Yes' or 'No', and nothing else."}}
Your response:
'''

    return prompt.format(text=text, source=source, target=target)


def predict_by_structured_examples_prompt(text: str, source: str, target: str, examples: list[dict[str, Any]], use_cot: bool = False) -> str:
    instruction_prompt = '''Given a text, two events (Event X and Event Y). Based on the related examples, you need to determine whether there is a causal relationship between the given events X and Y. Please follow the instructions below and refer to the provided examples when answering.
###
Instructions:
You should refer to the reasoning process in the examples but not be entirely influenced by them. Whether the events in the examples have a causal relationship DOES NOT affect whether the given events in the provided text have a causal relationship.
You should give step-by-step reasoning path before giving the final answer.
'''

    example_prompt = '''***Example {idx}***
Text: {text};
Event X: {source};
Event Y: {target};
Answer: {{"Answer": "{answer}"}}
'''

    # Used when use_cot=True and the example carries a synthesized 'cot' field (see
    # utils/cot_synthesis.py), so the fewshot demonstrates a reasoning path rather than a bare label.
    example_prompt_cot = '''***Example {idx}***
Text: {text};
Event X: {source};
Event Y: {target};
Reasoning: {cot}
Answer: {{"Answer": "{answer}"}}
'''

    example_template = example_prompt_cot if use_cot else example_prompt

    examples_prompt = ''
    for idx, e in enumerate(examples, start=1):
        examples_prompt += example_template.format(idx=idx,
                                                               text=e['input_text'],
                                                               source=e['source'],
                                                               target=e['target'],
                                                               cot=e.get('cot', ''),
                                                               answer='Yes' if e['ground'] == 1 else 'No') + '\n'

    prompt = '''{instruction_prompt}
###
Here are some examples.
{examples_prompt}
###
Text: {text};
Event X: {source};
Event Y: {target};
Give step-by-step reasoning path, and then organize the final answer in JSON format: {{"Answer": "Your answer, the answer must be either 'Yes' or 'No', and nothing else."}}
Your response:
'''

    return prompt.format(instruction_prompt=instruction_prompt.format(text=text, source=source, target=target),
                         examples_prompt=examples_prompt,
                         text=text,
                         source=source,
                         target=target)

