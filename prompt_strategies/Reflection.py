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

MBPP_system_message = 'Only generate the code.'

DS1000_system_message = 'Only generate the code.'

Reflection_system_message = ''

MBPP_Reflection_prompt = '''
{prompt}

The function name and input variables should follow this template: {function_name}.
'''

Reflection_prompt = '''
Here is a code snippet:
{code}

Please review this code and suggest any improvements or identify any issues.
'''

HumanEval_Refinement_prompt = '''
Here is a code snippet:
{initial_code}

Based on the following feedback, refine the code:
{reflection}
'''

MBPP_Refinement_prompt = '''
Here is a code snippet:
{initial_code}

Based on the following feedback, refine the code:
{reflection}

Refined code (The function name and input variables should follow this template: {function_name}):
'''

DS1000_Refinement_prompt = '''
Here is a code snippet:
{initial_code}

Based on the following feedback, refine the code:
{reflection}
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
            message =[{'role': 'system', 'content': HumanEval_system_message}, {'role': 'user', 'content': prompt}]
        elif args.dataset == 'MBPP':
            function_name = get_function_info(per_data['code'])
            prompt = per_data['text']
            message =[{'role': 'system', 'content': MBPP_system_message}, {'role': 'user', 'content': MBPP_Reflection_prompt.format(prompt=prompt, function_name=function_name)}]
        elif args.dataset == 'DS1000':
            prompt = per_data['prompt']
            message =[{'role': 'system', 'content': DS1000_system_message}, {'role': 'user', 'content': prompt}]
        messages.append(message)

    return messages, data_selected

def Reflection_generate_result(args):
    output_path = f'result/{args.dataset}_gpt4o_result/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}.jsonl'
    messages, data = generate_prompt(args)
    for idx, per_data in enumerate(tqdm.tqdm(data)):
        result = copy.copy(per_data)
        response = call_chat_gpt(messages[idx], args)

        input_token_init, input_cost_init = num_tokens_from_messages(messages[idx], args.model, True)
        output_token_init, output_cost_init = num_tokens_from_messages(response, args.model, False)

        code = process_generation_to_code(response)
        

        if args.dataset == 'HumanEval':
            for i in range(2):  # Number of iterations for self-reflection
            # i=0
                reflection = [{'role': 'system', 'content': Reflection_system_message}, {'role': 'user', 'content': Reflection_prompt.format(code='\n'.join(code))}]
                reflection_response = call_chat_gpt(reflection, args)
                input_token, input_cost = num_tokens_from_messages(reflection, args.model, True)
                output_token, output_cost = num_tokens_from_messages(reflection_response, args.model, False)
                input_token_init += input_token
                input_cost_init += input_cost
                output_cost_init += output_cost
                output_token_init += output_token
                print(f"\nReflection {i+1}:\n", reflection_response)
                
                # Assume the reflection contains suggestions or improvements
                refined_code = [{'role': 'system', 'content': MBPP_system_message}, {'role': 'user', 'content': HumanEval_Refinement_prompt.format(initial_code='\n'.join(code), reflection=reflection_response)}]
                refined_response = call_chat_gpt(refined_code, args)
                input_token, input_cost = num_tokens_from_messages(refined_code, args.model, True)
                output_token, output_cost = num_tokens_from_messages(refined_response, args.model, False)
                input_token_init += input_token
                input_cost_init += input_cost
                output_cost_init += output_cost
                output_token_init += output_token
    
    
                code = process_generation_to_code(refined_response)  # Update initial code with the refined version
                
                print(f"\nRefined Code {i+1}:\n", code)

        elif args.dataset == 'MBPP':
            function_name = get_function_info(per_data['code'])
            # i=0
            for i in range(2):  # Number of iterations for self-reflection
                reflection = [{'role': 'system', 'content': Reflection_system_message}, {'role': 'user', 'content': Reflection_prompt.format(code='\n'.join(code))}]
                reflection_response = call_chat_gpt(reflection, args)
                input_token, input_cost = num_tokens_from_messages(reflection, args.model, True)
                output_token, output_cost = num_tokens_from_messages(reflection_response, args.model, False)
                input_token_init += input_token
                input_cost_init += input_cost
                output_cost_init += output_cost
                output_token_init += output_token
                print(f"\nReflection {i+1}:\n", reflection_response)
                
                # Assume the reflection contains suggestions or improvements
                refined_code = [{'role': 'system', 'content': MBPP_system_message}, {'role': 'user', 'content': MBPP_Refinement_prompt.format(initial_code='\n'.join(code), reflection=reflection_response, function_name=function_name)}]
                refined_response = call_chat_gpt(refined_code, args)
                input_token, input_cost = num_tokens_from_messages(refined_code, args.model, True)
                output_token, output_cost = num_tokens_from_messages(refined_response, args.model, False)
                input_token_init += input_token
                input_cost_init += input_cost
                output_cost_init += output_cost
                output_token_init += output_token
    
    
                code = process_generation_to_code(refined_response)  # Update initial code with the refined version
                
                print(f"\nRefined Code {i+1}:\n", code)

        elif args.dataset == 'DS1000':
            for i in range(3):  # Number of iterations for self-reflection
                reflection = [{'role': 'system', 'content': Reflection_system_message}, {'role': 'user', 'content': Reflection_prompt.format(code='\n'.join(code))}]
                reflection_response = call_chat_gpt(reflection, args)
                input_token, input_cost = num_tokens_from_messages(reflection, args.model, True)
                output_token, output_cost = num_tokens_from_messages(reflection_response, args.model, False)
                input_token_init += input_token
                input_cost_init += input_cost
                output_cost_init += output_cost
                output_token_init += output_token
                print(f"\nReflection {i+1}:\n", reflection_response)
                
                # Assume the reflection contains suggestions or improvements
                refined_code = [{'role': 'system', 'content': DS1000_system_message}, {'role': 'user', 'content': DS1000_Refinement_prompt.format(initial_code='\n'.join(code), reflection=reflection_response)}]
                refined_response = call_chat_gpt(refined_code, args)
                input_token, input_cost = num_tokens_from_messages(refined_code, args.model, True)
                output_token, output_cost = num_tokens_from_messages(refined_response, args.model, False)
                input_token_init += input_token
                input_cost_init += input_cost
                output_cost_init += output_cost
                output_token_init += output_token


                code = process_generation_to_code(refined_response)  # Update initial code with the refined version
                
                print(f"\nRefined Code {i+1}:\n", code)
        
        result['response_code'] = '\n'.join(code)
        result['input_token'] = input_token_init
        result['input_cost'] = input_cost_init
        result['output_token'] = output_token_init
        result['output_cost'] = output_cost_init

        write_to_file(result, output_path)



