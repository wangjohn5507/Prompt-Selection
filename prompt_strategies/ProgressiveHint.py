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

HumanEval_system_message = ''

HumanEval_system_message_2 = 'Only generate the code.'

HumanEval_init_prompt = '''
{prompt}
Please complete the task in Python.
'''

HumanEval_hint_prompt = '''
{prompt}
Please complete the task in Python.

The answer is near to:
{hint}
'''

MBPP_system_message = ''

MBPP_system_message_2 = 'Only generate the code.'

MBPP_init_prompt = '''
{prompt}
Please complete the task in Python.
The function name and input variables should follow this template: {function_name}.
'''

MBPP_hint_prompt = '''
{prompt}
Please complete the task in Python.
The function name and input variables should follow this template: {function_name}.

The answer is near to:
{hint}
'''

APPS_system_message = ''

APPS_few_shot_prompt = ''


def generate_prompt(args):
    if args.dataset == 'HumanEval':
        file_path = 'dataset/HumanEval.jsonl'
    elif args.dataset == 'MBPP':
        file_path = 'dataset/MBPP.jsonl'
    elif args.dataset == 'APPS':
        file_path = 'dataset/apps.jsonl'
    data = list(map(json.loads, open(file_path)))

    start = 0 if args.start == 0 else args.start
    end = len(data) if args.end == 0 else args.end

    data_selected = data[start:end]
    messages = []
    
    for per_data in data[start:end]:
        if args.dataset == 'HumanEval':
            prompt = per_data['prompt']
            message =[{'role': 'system', 'content': HumanEval_system_message_2}, {'role': 'user', 'content': HumanEval_init_prompt.format(prompt=prompt)}]
        elif args.dataset == 'MBPP':
            prompt = per_data['text']
            function_name = get_function_info(per_data['code'])
            message =[{'role': 'system', 'content': MBPP_system_message_2}, {'role': 'user', 'content': MBPP_init_prompt.format(prompt=prompt, function_name=function_name)}]
        elif args.dataset == 'APPS':
            prompt = per_data['question']
            message =[{'role': 'system', 'content': APPS_system_message}, {'role': 'user', 'content': APPS_few_shot_prompt.format(prompt=prompt)}]
        messages.append(message)

    return messages, data_selected

def ProgressiveHint_generate_result(args):
    output_path = f'result/{args.dataset}_gpt4o_result/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}.jsonl'
    messages, data = generate_prompt(args)
    for idx, per_data in enumerate(tqdm.tqdm(data)):
        result = copy.copy(per_data)
        
        if args.dataset == 'HumanEval':
            prompt = per_data['prompt']
            response = call_chat_gpt(messages[idx], args)
            print(response)
            code = process_generation_to_code(response)

            input_token, input_cost = num_tokens_from_messages(messages[idx], args.model, True)
            output_token, output_cost = num_tokens_from_messages(response, args.model, False)

            #iterative 1
            hint_message = [{'role': 'system', 'content': HumanEval_system_message_2}, {'role': 'user', 'content': HumanEval_hint_prompt.format(prompt=prompt, hint='\n'.join(code))}]
            print(hint_message)
            response2 = call_chat_gpt(hint_message, args)

            input_token2, input_cost2 = num_tokens_from_messages(hint_message, args.model, True)
            output_token2, output_cost2 = num_tokens_from_messages(response2, args.model, False)

            input_token += input_token2
            input_cost += input_cost2
            output_token += output_token2
            output_cost += output_cost2

            code2 = process_generation_to_code(response2)

            #iterative 2
            hint_message_2 = [{'role': 'system', 'content': HumanEval_system_message_2}, {'role': 'user', 'content': HumanEval_hint_prompt.format(prompt=prompt, hint='\n'.join(code2))}]
            response3 = call_chat_gpt(hint_message_2, args)

            input_token3, input_cost3 = num_tokens_from_messages(hint_message_2, args.model, True)
            output_token3, output_cost3 = num_tokens_from_messages(response3, args.model, False)

            input_token += input_token3
            input_cost += input_cost3
            output_token += output_token3
            output_cost += output_cost3

            code3 = process_generation_to_code(response3)

        if args.dataset == 'MBPP':
            function_name = get_function_info(per_data['code'])
            prompt = per_data['text']
            response = call_chat_gpt(messages[idx], args)
            print(response)
            code = process_generation_to_code(response)

            input_token, input_cost = num_tokens_from_messages(messages[idx], args.model, True)
            output_token, output_cost = num_tokens_from_messages(response, args.model, False)
            #iterative 1
            hint_message = [{'role': 'system', 'content': MBPP_system_message_2}, {'role': 'user', 'content': MBPP_hint_prompt.format(prompt=prompt, function_name=function_name, hint='\n'.join(code))}]
            print(hint_message)
            response2 = call_chat_gpt(hint_message, args)

            input_token2, input_cost2 = num_tokens_from_messages(hint_message, args.model, True)
            output_token2, output_cost2 = num_tokens_from_messages(response2, args.model, False)

            input_token += input_token2
            input_cost += input_cost2
            output_token += output_token2
            output_cost += output_cost2

            code2 = process_generation_to_code(response2)

            #iterative 2
            hint_message_2 = [{'role': 'system', 'content': MBPP_system_message_2}, {'role': 'user', 'content': MBPP_hint_prompt.format(prompt=prompt, function_name=function_name, hint='\n'.join(code2))}]
            response3 = call_chat_gpt(hint_message_2, args)

            input_token3, input_cost3 = num_tokens_from_messages(hint_message_2, args.model, True)
            output_token3, output_cost3 = num_tokens_from_messages(response3, args.model, False)

            input_token += input_token3
            input_cost += input_cost3
            output_token += output_token3
            output_cost += output_cost3

            code3 = process_generation_to_code(response3)
        
        result['response_code'] = '\n'.join(code3)
        result['input_token'] = input_token
        result['input_cost'] = input_cost
        result['output_token'] = output_token
        result['output_cost'] = output_cost

        write_to_file(result, output_path)

