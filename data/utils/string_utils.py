from typing import Dict
from ast import literal_eval
import re
import random

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

QUERY_TEMPLATE_NOANSWER_TOKENLIMIT = """
{Question}

Think for up to {time_limit} tokens.
""".strip()

QUERY_TEMPLATE = """
{Question}

Put your answer on its own line after "Answer:"
""".strip()

QUERY_TEMPLATE_NOANSWER_STEPSLEFT = """
{Question}

Think for up to {time_limit} steps.
""".strip()

QUERY_TEMPLATE_WITH_TIME_LIMIT = """
{Question}

Solve the problem within {time_limit} steps. Put your answer on its own line after "Answer:"
""".strip()

QUERY_TEMPLATE_WITH_TOKEN_TIME_LIMIT = """
{Question}

Solve the problem within {time_limit} tokens. Put your answer on its own line after "Answer:"
""".strip()


GPQA_FORMAT_1 = """
Answer the following multiple-choice question. Your response must adhere to these rules:
  1. Think step by step to arrive at the correct answer.
  2. Avoid repeating reasoning or steps already stated.
  3. Ensure your response is within the word limit.
  4. Conclude with the final answer in the format: 'Answer: $LETTER' (without quotes), where LETTER is one of ABCD.
  
  {Question}
  
  A) {choice1}
  B) {choice2}
  C) {choice3}
  D) {choice4}
""".strip()

GPQA_FORMAT_2 = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

GPQA_FORMAT_3 = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A} B) {B} C) {C} D) {D}
""".strip()

NO_THINK = [
    "You may not think for this question.",
    "Answer the question without thinking.",
    "Please answer the question without thinking.",
    "Answer directly with no thinking.",
    "Answer without thinking.",
]

THINK_SHORT = [
    "Please think for a short amount of time before answering.",
    "Think only briefly before answering.",
    "Answer after a short amount of thinking.",
    "Think for a short amount of time before answering.",
    "Answer after a short amount of thought.",
    "Answer with a short amount of thinking.",
]

THINK_MEDIUM = [
    "Please think for a moderate amount of time before answering. If you are done early, double check your work.",
    "Think for a moderate amount of time before answering to make sure you get it right.",
    "Answer after a moderate amount of thinking.",
    "Answer after a moderate amount of thought.",
    "Answer with a moderate amount of thinking.",
]

THINK_LONG = [
    "Please think for a long amount of time before answering to ensure accuracy.",
    "Think for a long amount of time before answering.",
    "Answer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double check your work until you are absolutely sure that you have the correct answer.",
    "Answer after a long amount of thought.",
    "Answer with a long amount of thinking.",
]

import numpy as np

def power_law_sample(a=None, b=None, c=None, valid_numbers=None, exponent=-1):
    """
    Samples a new number based on a power-law distribution where 'a' is most likely and 'c' is least likely.
    
    Args:
        a (int): The starting number (must be a multiple of b).
        b (int): The base multiple.
        c (int): The maximum number (must be >= a and a multiple of b).
        exponent (float): The exponent for the power-law distribution (default is -2).

    Returns:
        int: A sampled number that is a multiple of b between a and c.
    """
    if valid_numbers is None:
        # Ensure a, b, and c are valid
        if a % b != 0 or c % b != 0 or a > c:
            raise ValueError("Invalid input: Ensure a and c are multiples of b and a <= c.")
        
        # Generate valid multiples of b between a and c (inclusive)
        valid_numbers = np.arange(a, c + 1, b)
    
    # Compute probabilities for the power-law distribution
    # The probability is proportional to (1 / number^|exponent|)
    probabilities = (valid_numbers ** exponent)
    
    # Normalize the probabilities to sum to 1
    probabilities /= probabilities.sum()
    
    # Sample a number according to the power-law distribution
    sampled_number = np.random.choice(valid_numbers, p=probabilities)
    
    return sampled_number

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_cot_example(
    example: Dict,
    idx=None,
    tokenize=True, 
    tokenizer=None, 
    time_limit=False, 
    model_type="llama",
    step_format="",
    sample_max_steps=False,
):
    """
    Testing:
    ```python
    from datasets import load_dataset; from tokenization import _process_example
    dataset = load_dataset('qfq/cotmath', download_mode='force_redownload')['train']
    example = dataset[0]
    _process_example(example)
    ```
    """
    import numpy as np
    thinking_trajectory = example.get("thinking_trajectory", example.get("thinking_trajectories", example.get("refined_thinking_trajectory",  example.get("cot", example.get("thinking")))))
    question = example.get('orig_problem', example.get('question'))
    answer = example.get("orig_answer", example.get("attempt", example.get("response")))
    if isinstance(thinking_trajectory, str):
        thinking_trajectory = [thinking_trajectory]
    
    if step_format == "tokensnosteps":
        text = "\n".join(thinking_trajectory)
        tokens = tokenizer(text)['input_ids']
        tokens_len = len(tokens)
        time_limit = 2**(tokens_len).bit_length()
        if sample_max_steps:
            # Temporarily set new random seed
            np.random.seed(idx)
            #time_limit = int(power_law_sample(time_limit, 20.0, 1000.0))
            # Allow from time_limit in powers of 2 up to 2048
            valid_numbers = [float(time_limit)] + [float(2**i) for i in range(time_limit.bit_length(), 12) if 2**i <= 2048]
            time_limit = int(power_law_sample(valid_numbers=np.array(valid_numbers)))
            np.random.seed(1234)
        if step_format == "tokensnostepswarnings":
            # Insert warnings
            tokens_inserted_len = 0
            half_warning = time_limit // 2
            if tokens_len > half_warning:
                tokens_to_insert = tokenizer(f"<|imstart|>You have {half_warning} tokens left<|imstart|>")[0]['input_ids']
                tokens.insert(half_warning, tokens_to_insert)
            quarter_warning = time_limit // 4
            if tokens_len > half_warning + quarter_warning:
                tokens_inserted_len += len(tokens_to_insert)
                tokens_to_insert = tokenizer(f"<|imstart|>You have {quarter_warning} tokens left<|imstart|>")[0]['input_ids']
                tokens.insert(half_warning + quarter_warning + tokens_inserted_len, tokens_to_insert)
        prompt = QUERY_TEMPLATE_NOANSWER_TOKENLIMIT.format(Question=question, time_limit=time_limit)
    elif "tokens" in step_format:
        step_toks = [len(tokenizer(x)['input_ids']) for x in thinking_trajectory]

        # If rounding to next multiple of 20
        # time_limit = 20 * ((sum(step_toks) + 19) // 20)
        prompt = QUERY_TEMPLATE_WITH_TOKEN_TIME_LIMIT.format(Question=question, time_limit=time_limit)
        step_toks = [0] + list(np.cumsum(step_toks))
        if step_format == "tokensleft":
            step_toks = [time_limit - x for x in step_toks]
    elif time_limit:
        if len(thinking_trajectory) == 1:
            print("Warning: Single step in thinking trajectory; Splitting into steps on double newlines")
            thinking_trajectory = thinking_trajectory[0].split("\n\n")
            thinking_trajectory = [step.strip() for step in thinking_trajectory if step.strip()]
        
        time_limit = 2**(len(thinking_trajectory) - 1).bit_length()
        # If rounding to next multiple of 20
        #time_limit = 20 * ((len(thinking_trajectory) + 19) // 20)
        if sample_max_steps:
            # Temporarily set new random seed
            np.random.seed(idx)
            #time_limit = int(power_law_sample(time_limit, 20.0, 1000.0))
            # Allow from time_limit in powers of 2 up to 2048
            valid_numbers = [float(time_limit)] + [float(2**i) for i in range(time_limit.bit_length(), 12) if 2**i <= 2048]
            time_limit = int(power_law_sample(valid_numbers=np.array(valid_numbers)))
            np.random.seed(1234)

        if step_format == "stepsleftnoanswer":
            prompt = QUERY_TEMPLATE_NOANSWER_STEPSLEFT.format(Question=question, time_limit=time_limit)
        else:
            prompt = QUERY_TEMPLATE_WITH_TIME_LIMIT.format(Question=question, time_limit=time_limit)
        if "stepsleft" in step_format:
            steps = [time_limit - i for i in range(len(thinking_trajectory))]
    elif step_format == "nostepsnoanswer":
        # Problem: The answer most likely does not follow the prompt template i.e. no A / B / C / D in the response as we did not provide the choices to Gemini
        # if ("gpqa" in example['source_type']): 
        #     metadata = literal_eval(example['metadata'])
        #     choices = [
        #         preprocess(doc["Incorrect Answer 1"]),
        #         preprocess(doc["Incorrect Answer 2"]),
        #         preprocess(doc["Incorrect Answer 3"]),
        #         preprocess(doc["Correct Answer"]),
        #     ]
        #     random.shuffle(choices)
        #     doc = {
        #         "choice1": choices[0],
        #         "choice2": choices[1],
        #         "choice3": choices[2],
        #         "choice4": choices[3],
        #     }
        #     rand = np.random.rand()
        #     if rand < 0.25:
        #         prompt = GPQA_FORMAT_1.format(
        #             Question=question,
        #             **doc
        #         )
        #     elif rand < 0.5:
        #         prompt = GPQA_FORMAT_2.format(
        #             Question=question,
        #             **doc
        #         )
        #     else:
        #         prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)
        # else:
        prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)
    else:
        prompt = QUERY_TEMPLATE.format(Question=question)

    if model_type == "llama":
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
            {
                "role": "assistant", 
                "content": "<|reserved_special_token_0|>\n" + "\n<|reserved_special_token_2|>\n".join(thinking_trajectory) + "\n<|reserved_special_token_1|>\n\nAnswer: " + answer
            }
        ], tokenize=False)
    elif model_type in ("qwen", "qwq"):
        answer = "Answer: " + answer if "Answer:" not in answer else answer
        if step_format == "tokensnosteps":
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {
                    "role": "assistant", 
                    "content": "<|im_start|>think\n" + "\n".join(thinking_trajectory).strip() + "\n<|im_start|>answer\n" + answer.strip()
                }
            ], tokenize=False)
        elif step_format == "numbered":
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {
                    "role": "assistant", 
                    "content": "<|im_start|>" + "\n<|im_start|>".join("step" + str(i) + "\n" + s for i, s in enumerate(thinking_trajectory, start=1)) + "\n<|im_start|>answer\n" + answer
                }
            ], tokenize=False)
        elif step_format == "numberedmax":
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {
                    "role": "assistant", 
                    "content": "<|im_start|>" + "\n<|im_start|>".join("step " + str(i) + "/" + str(time_limit) + "\n" + s for i, s in enumerate(thinking_trajectory, start=1)) + "\n<|im_start|>answer\n" + answer
                }
            ], tokenize=False)
        elif step_format == "numberedtokens":
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {
                    "role": "assistant", 
                    "content": "<|im_start|>" + "\n<|im_start|>".join("step " + str(step_toks[i-1]) + "/" + str(time_limit) + "\n" + s for i, s in enumerate(thinking_trajectory, start=1)) + "\n<|im_start|>answer\n" + answer
                }
            ], tokenize=False)        
        elif step_format == "tokensleft":
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": "<|im_start|>" + "\n<|im_start|>".join(str(step_toks[i]) + " tokens left" + "\n" + s for i, s in enumerate(thinking_trajectory)) + "\n<|im_start|>answer\n" + answer
                }
            ], tokenize=False)
        elif "stepsleft" in step_format:
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": "<|im_start|>" + "\n<|im_start|>".join(str(steps[i]) + " steps left" + "\n" + s for i, s in enumerate(thinking_trajectory)) + "\n<|im_start|>answer\n" + answer
                }
            ], tokenize=False)
        elif "nosteps" in step_format:
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {
                    "role": "assistant", 
                    "content": "<|im_start|>think\n" + "\n".join(thinking_trajectory).strip() + "\n<|im_start|>answer\n" + answer.strip()
                }
            ], tokenize=False)
        else:
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {
                    "role": "assistant", 
                    "content": "<|im_start|>think\n" + "\n<|im_start|>step\n".join(thinking_trajectory) + "\n<|im_start|>answer\n\n" + answer
                }
            ], tokenize=False)


    if tokenize is False: return dict(text=text)
    token_ids = tokenizer(text, add_special_tokens=False)['input_ids']
    return dict(text=text, ids=token_ids, len=len(token_ids))

def extract_content(input_string, start_token, end_token=None):
    start_index = input_string.find(start_token) + len(start_token)
    if end_token is None: return input_string[start_index:].strip()
    end_index = input_string.find(end_token)
    # Return None if neither token is found
    if start_index == -1 or end_index == -1: return None
    return input_string[start_index:end_index].strip()

def remove_special_tokens(input_string):
    """
    Remove the special tokens from the input string.
    """
    if input_string is None: return ""
    special_tokens = [
        '<|reserved_special_token_0|>',
        '<|reserved_special_token_1|>',
        '<|reserved_special_token_2|>'
    ]
    for token in special_tokens:
        input_string = input_string.replace(token, '')
    return input_string

def parse_eidata_string(s):
    """
    Parse strings of the format "qfq/eidata_{any_string}_iter{int}"
    Args:
        s (str): Input string to parse
    Returns:
        tuple: (prefix string, iteration number) if matched, None otherwise
    """
    pattern = r'^(qfq/eidata_.*?_iter)(\d+)$'
    match = re.match(pattern, s)
    if match:
        prefix = match.group(1)
        iteration = int(match.group(2))
        return prefix, iteration
    return None

def parse_completion(completion):
    """
    <step>
    <thought 1>

    <step>
    <thought 2>

    ...
    <step>
    <thought N>

    <answer>
    <final answer>

    Need to parse the completion and extract the thinking trajectory as a list of strings
    :param completion:
    :return:
    """
    # Split into sections and clean up
    thinking_trajectory = completion.split("<step>")
    thinking_trajectory = [step.strip().replace('</step>', '').strip() for step in thinking_trajectory if step.strip()]
    # Extract answer section
    answer = ""
    if "<answer>" in thinking_trajectory[-1]:
        trajectory_part, answer_part = thinking_trajectory[-1].split("<answer>")
        thinking_trajectory[-1] = trajectory_part.strip().replace('</step>', '')
        thinking_trajectory[-1] = thinking_trajectory[-1].replace('## Current Incorrect Thinking Trajectory\n', '')
        answer = answer_part.strip().replace('</answer>', '')
    # Remove any empty steps
    thinking_trajectory = [step for step in thinking_trajectory if step]
    return thinking_trajectory, answer

def format_cot_example(example: Dict, noattempt=False) -> str:
    thinking_trajectory = "\n".join([f"Step {i}\n{step}" for i, step in enumerate(example['thinking_trajectories'])])
    if noattempt:
        question_input =  ("## Question\n" +
                       f"{example['question']}\n\n" +
                       "## Thinking Trajectory\n" +
                       f"{thinking_trajectory}\n\n")
    else:
        question_input =  ("## Question\n" +
                       f"{example['question']}\n\n" +
                       "## Thinking Trajectory\n" +
                       f"{thinking_trajectory}\n\n"
                       "## Attempt\n" +
                       f"{example['attempt']}")
    return question_input