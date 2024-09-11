import pandas as pd
import json
import copy
import tqdm
import copy
from src.model import call_chat_gpt
from src.utils import num_tokens_from_messages
from src.utils import process_generation_to_code
from src.utils import write_to_file
from src.utils import get_function_info


HumanEval_system_message = 'Only generate the code.'

HumanEval_Zeroshot_CoT_prompt = '''
{prompt}

Let's generate the code step by step.
'''

MBPP_system_message = 'Only generate the Python code for the following task.'

MBPP_Zeroshot_CoT_prompt = '''
{prompt}

The function name and input variables should follow this template: {function_name}.

Let's generate the code step by step.
'''

DS1000_system_message = 'Only generate the code.'

DS1000_Zeroshot_CoT_prompt = '''
{prompt}

Let's generate the code step by step.
'''


def generate_prompt(args):
    if args.dataset == 'HumanEval':
        file_path = 'dataset/HumanEval.jsonl'
    elif args.dataset == 'MBPP':
        file_path = 'dataset/MBPP.jsonl'
    elif args.dataset == 'DS1000':
        file_path = 'dataset/DS1000.jsonl'
    data = list(map(json.loads, open(file_path)))

    start = 0 if args.start == 0 else args.start
    end = len(data) if args.end == 0 else args.end

    data_selected = data[start:end]
    messages = []
    
    for per_data in data[start:end]:
        if args.dataset == 'HumanEval':
            prompt = per_data['prompt']
            message =[{'role': 'system', 'content': HumanEval_system_message}, {'role': 'user', 'content': HumanEval_Zeroshot_CoT_prompt.format(prompt=prompt)}]
        elif args.dataset == 'MBPP':
            function_name = get_function_info(per_data['code'])
            prompt = per_data['text']
            message =[{'role': 'system', 'content': MBPP_system_message}, {'role': 'user', 'content': MBPP_Zeroshot_CoT_prompt.format(prompt=prompt, function_name=function_name)}]
        elif args.dataset == 'DS1000':
            prompt = per_data['prompt']
            message =[{'role': 'system', 'content': DS1000_system_message}, {'role': 'user', 'content': DS1000_Zeroshot_CoT_prompt.format(prompt=prompt)}]
        messages.append(message)

    return messages, data_selected

def Zeroshot_CoT_generate_result(args):
    output_path = f'result/{args.dataset}_gpt4o_result/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}.jsonl'
    messages, data = generate_prompt(args)
    for idx, per_data in enumerate(tqdm.tqdm(data)):
        result = copy.copy(per_data)
        response = call_chat_gpt(messages[idx], args)

        input_token, input_cost = num_tokens_from_messages(messages[idx], args.model, True)
        output_token, output_cost = num_tokens_from_messages(response, args.model, False)

        code = process_generation_to_code(response)
        
        result['response_code'] = '\n'.join(code)
        result['input_token'] = input_token
        result['input_cost'] = input_cost
        result['output_token'] = output_token
        result['output_cost'] = output_cost

        write_to_file(result, output_path)

