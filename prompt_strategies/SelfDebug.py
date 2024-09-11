import pandas as pd
import json
import copy
import tqdm
import copy
from src.model import call_chat_gpt
from src.utils import num_tokens_from_messages
from src.utils import process_generation_to_code
from src.utils import write_to_file
from src.utils import extract_one_assert, get_function_info
from src.evaluation import check_code, MBPP_check_code, APPS_check_code, DS1000_check_code

code_system_message = ''

MBPP_system_message = ''

DS1000_system_message = ''

HumanEval_SelfDebug_init_prompt = '''
Complete the following task in Python:
{prompt}

Your code should pass the test:
{assertion}
'''

MBPP_SelfDebug_init_prompt = '''
{prompt}

The function name and input variables should follow this template: {function_name}.

Your code should pass the test: 
{test}
'''

DS1000_SelfDebug_init_prompt = '''
{prompt}
'''

SelfDebug_success_prompt = '''
{code}
Is the code above correct? If not, please fix it.
'''

SelfDebug_failed_prompt = '''
{code}
The code above is wrong. Please fix it.
'''

DS1000_failed_prompt = '''
{code}
The code above is wrong. The reason is {reason}. Please fix it.
'''

def generate_init_prompt(args):
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
            unit_test = extract_one_assert(per_data['test'])
            message =[{'role': 'system', 'content': code_system_message}, {'role': 'user', 'content': HumanEval_SelfDebug_init_prompt.format(prompt=prompt, assertion = unit_test)}]
        elif args.dataset == 'MBPP':
            function_name = get_function_info(per_data['code'])
            prompt = per_data['text']
            test = per_data['test_list'][0]
            message =[{'role': 'system', 'content': MBPP_system_message}, {'role': 'user', 'content': MBPP_SelfDebug_init_prompt.format(prompt=prompt, test = test, function_name=function_name)}]
        elif args.dataset == 'DS1000':
            prompt = per_data['prompt']
            # print(DS1000_SelfDebug_init_prompt.format(prompt=prompt))
            message =[{'role': 'system', 'content': DS1000_system_message}, {'role': 'user', 'content': DS1000_SelfDebug_init_prompt.format(prompt=prompt)}]
        messages.append(message)

    return messages, data_selected


def process(code, args, total_input_cost, total_input_token, total_output_cost, total_output_token, is_success):
    if is_success:
        message_process = [{'role': 'system', 'content': code_system_message}, {'role': 'user', 'content': SelfDebug_success_prompt.format(code = code)}]
    else:
        message_process = [{'role': 'system', 'content': code_system_message}, {'role': 'user', 'content': SelfDebug_failed_prompt.format(code = code)}]
    response_process = call_chat_gpt(message_process, args)

    input_token_process, input_cost_process = num_tokens_from_messages(message_process, args.model, True)
    output_token_process, output_cost_process = num_tokens_from_messages(response_process, args.model, False)
    total_input_cost += input_cost_process
    total_input_token += input_token_process
    total_output_cost += output_cost_process
    total_output_token += output_token_process
    

    process_code = process_generation_to_code(response_process)
    process_code = '\n'.join(process_code)

    return process_code, total_input_cost, total_input_token, total_output_cost, total_output_token

def DS1000_process(code, error, args, total_input_cost, total_input_token, total_output_cost, total_output_token, is_success):
    if is_success:
        message_process = [{'role': 'system', 'content': code_system_message}, {'role': 'user', 'content': SelfDebug_success_prompt.format(code = code)}]
    else:
        print(DS1000_failed_prompt.format(code = code, reason = error))
        message_process = [{'role': 'system', 'content': code_system_message}, {'role': 'user', 'content': DS1000_failed_prompt.format(code = code, reason = error)}]
    response_process = call_chat_gpt(message_process, args)

    input_token_process, input_cost_process = num_tokens_from_messages(message_process, args.model, True)
    output_token_process, output_cost_process = num_tokens_from_messages(response_process, args.model, False)
    total_input_cost += input_cost_process
    total_input_token += input_token_process
    total_output_cost += output_cost_process
    total_output_token += output_token_process
    

    process_code = process_generation_to_code(response_process)
    process_code = '\n'.join(process_code)

    return process_code, total_input_cost, total_input_token, total_output_cost, total_output_token



def SelfDebug_generate_result(args):
    output_path = f'result/{args.dataset}_gpt4o_result/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}.jsonl'
    messages, data = generate_init_prompt(args)
    total_input_token = 0
    total_output_cost = 0
    total_input_cost = 0
    total_output_token = 0
    
    for idx, per_data in enumerate(tqdm.tqdm(data)):
        tried = 0
        result = copy.copy(per_data)
        response = call_chat_gpt(messages[idx], args)

        input_token_init, input_cost_init = num_tokens_from_messages(messages[idx], args.model, True)
        output_token_init, output_cost_init = num_tokens_from_messages(response, args.model, False)
        total_input_cost += input_cost_init
        total_input_token += input_token_init
        total_output_cost += output_cost_init
        total_output_token += output_token_init

        code = process_generation_to_code(response)
        code = '\n'.join(code)

        if args.dataset == 'HumanEval':
            test = per_data['test']
            checked = check_code(per_data['prompt'], code, f'def check(candidate):\n    {extract_one_assert(test)}\n', per_data['entry_point'])

            while checked == False and tried < 5:
                print('wrong anser, regenerate code!')
                process_code, total_input_cost, total_input_token, total_output_cost, total_output_token = process(code, args, total_input_cost, total_input_token, total_output_cost, total_output_token, False)
                code = process_code
                tried += 1
                checked = check_code(per_data['prompt'], code, f'def check(candidate):\n    {extract_one_assert(test)}\n', per_data['entry_point'])


            process_code, total_input_cost, total_input_token, total_output_cost, total_output_token = process(code, args, total_input_cost, total_input_token, total_output_cost, total_output_token, True)

            if check_code(per_data['prompt'], process_code, f'def check(candidate):\n    {extract_one_assert(test)}\n', per_data['entry_point']):
                print('Regenerate code success, use regenerate one!')
                result['response_code'] = process_code
            else:
                print('Regenerate code failed, use original code!')
                result['response_code'] = code
        
        elif args.dataset == 'MBPP':
            test = per_data['test_list'][0]
            checked = MBPP_check_code(code, test)

            while checked == False and tried < 5:
                print('wrong anser, regenerate code!')
                process_code, total_input_cost, total_input_token, total_output_cost, total_output_token = process(code, args, total_input_cost, total_input_token, total_output_cost, total_output_token, False)
                code = process_code
                tried += 1
                checked = MBPP_check_code(code, test)

            process_code, total_input_cost, total_input_token, total_output_cost, total_output_token = process(code, args, total_input_cost, total_input_token, total_output_cost, total_output_token, True)

            if MBPP_check_code(process_code, test):
                print('Regenerate code success, use regenerate one!')
                result['response_code'] = process_code
            else:
                print('Regenerate code failed, use original code!')
                result['response_code'] = code

        elif args.dataset == 'DS1000':
            final = code
            test_code = per_data['code_context']
            checked = True
            if DS1000_check_code(final, test_code):
                checked = True
            else:
                checked = False

            while checked == False and tried < 5:
                print('wrong anser, regenerate code!')
                exec(test_code, globals())
                try:
                    exec(f'test_execution(\'{final}\')')
                    print('Success')
                except Exception as e:
                    error = e
                    
                process_code, total_input_cost, total_input_token, total_output_cost, total_output_token = DS1000_process(code, error, args, total_input_cost, total_input_token, total_output_cost, total_output_token, False)
                code = process_code
                tried += 1
                checked = DS1000_check_code(code, test_code)


            process_code, total_input_cost, total_input_token, total_output_cost, total_output_token = process(code, args, total_input_cost, total_input_token, total_output_cost, total_output_token, True)

            if DS1000_check_code(process_code, test_code):
                print('Regenerate code success, use regenerate one!')
                result['response_code'] = process_code
            else:
                print('Regenerate code failed, use original code!')
                result['response_code'] = code
        
        
        result['input_token'] = total_input_token
        result['input_cost'] = total_input_cost
        result['output_token'] = total_output_token
        result['output_cost'] = total_output_cost

        write_to_file(result, output_path)
