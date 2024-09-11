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

HumanEval_plan_prompt = '''
Intent: Write a function to find the similar elements from the given two tuple lists.

Plan: Let's think step by step.
1. Define a function that takes two lists of tuples as arguments.
2. Create an empty list to store the similar elements found in both tuple lists.
3. Use a loop to iterate through each element (tuple) in the first list.
4. For each tuple in the first list, check if it also exists in the second list.
5. If the tuple from the first list is found in the second list, append it to the result list.
6. After iterating through the first list, return the list containing the similar elements.

Intent: Write a python function to identify non-prime numbers.

Plan: Let's think step by step.
1. Start by defining a function that accepts a single integer as an argument. This integer will be the number we want to check.
2. Before proceeding to the prime check, consider edge cases.
3. A prime number is a number greater than 1 that has no divisors other than 1 and itself.
4. Loop through all numbers starting from 2 up to the square root of the number (inclusive).
5. If any of these numbers divides the given number evenly (i.e., the remainder is 0), the number is non-prime.
6. If a divisor is found in the loop, the function should return True (indicating the number is non-prime).
7. If no divisors are found, the function should return False (indicating the number is prime).

Intent: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.

Plan: Let's think step by step.
1. Import the Required Module.
2. Define a function that accepts two arguments.
3. Check if the list is empty or if n is greater than the length of the list. If either case is true, handle it appropriately.
4. Use the heapq.nlargest Function.
5. The heapq.nlargest function returns a list of the n largest integers. Return this list as the result of the function.

How about this intent: {prompt}.

Plan: Let's think step by step.
'''

HumanEval_implementation_prompt = '''
{prompt}
Please complete the task with the following steps in Python.

{plan}
'''

APPS_system_message = ''

APPS_few_shot_prompt = ''

MBPP_system_message = ''

MBPP_system_message_2 = 'Only generate the code.'

MBPP_plan_prompt = '''
Intent: Write a function to find the similar elements from the given two tuple lists.

Plan: Let's think step by step.
1. Define a function that takes two lists of tuples as arguments.
2. Create an empty list to store the similar elements found in both tuple lists.
3. Use a loop to iterate through each element (tuple) in the first list.
4. For each tuple in the first list, check if it also exists in the second list.
5. If the tuple from the first list is found in the second list, append it to the result list.
6. After iterating through the first list, return the list containing the similar elements.

Intent: Write a python function to identify non-prime numbers.

Plan: Let's think step by step.
1. Start by defining a function that accepts a single integer as an argument. This integer will be the number we want to check.
2. Before proceeding to the prime check, consider edge cases.
3. A prime number is a number greater than 1 that has no divisors other than 1 and itself.
4. Loop through all numbers starting from 2 up to the square root of the number (inclusive).
5. If any of these numbers divides the given number evenly (i.e., the remainder is 0), the number is non-prime.
6. If a divisor is found in the loop, the function should return True (indicating the number is non-prime).
7. If no divisors are found, the function should return False (indicating the number is prime).

Intent: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.

Plan: Let's think step by step.
1. Import the Required Module.
2. Define a function that accepts two arguments.
3. Check if the list is empty or if n is greater than the length of the list. If either case is true, handle it appropriately.
4. Use the heapq.nlargest Function.
5. The heapq.nlargest function returns a list of the n largest integers. Return this list as the result of the function.

How about this intent: {problem}.

Plan: Let's think step by step.
'''

MBPP_implementation_prompt = '''
{problem}
Please complete the task with the following steps in Python.
The function name and input variables should follow this template: {function_name}.

{plan}
'''


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
            message =[{'role': 'system', 'content': HumanEval_system_message}, {'role': 'user', 'content': HumanEval_plan_prompt.format(prompt=prompt)}]
        elif args.dataset == 'MBPP':
            prompt = per_data['text']
            message =[{'role': 'system', 'content': MBPP_system_message}, {'role': 'user', 'content': MBPP_plan_prompt.format(problem=prompt)}]
        elif args.dataset == 'APPS':
            prompt = per_data['question']
            message =[{'role': 'system', 'content': APPS_system_message}, {'role': 'user', 'content': APPS_few_shot_prompt.format(prompt=prompt)}]
        messages.append(message)

    return messages, data_selected

def SelfPlan_generate_result(args):
    output_path = f'result/{args.dataset}_gpt4o_result/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}.jsonl'
    messages, data = generate_prompt(args)
    for idx, per_data in enumerate(tqdm.tqdm(data)):
        result = copy.copy(per_data)

        if args.dataset == 'HumanEval':
            prompt = per_data['prompt']
            response = call_chat_gpt(messages[idx], args)
            print(response)

            input_token, input_cost = num_tokens_from_messages(messages[idx], args.model, True)
            output_token, output_cost = num_tokens_from_messages(response, args.model, False)

            imple_message = [{'role': 'system', 'content': HumanEval_system_message_2}, {'role': 'user', 'content': HumanEval_implementation_prompt.format(prompt=prompt,plan=response)}]
            response2 = call_chat_gpt(imple_message, args)

            input_token2, input_cost2 = num_tokens_from_messages(imple_message, args.model, True)
            output_token2, output_cost2 = num_tokens_from_messages(response2, args.model, False)

            input_token += input_token2
            input_cost += input_cost2
            output_token += output_token2
            output_cost += output_cost2

            code = process_generation_to_code(response2)

        if args.dataset == 'MBPP':
            function_name = get_function_info(per_data['code'])
            prompt = per_data['text']
            response = call_chat_gpt(messages[idx], args)
            print(response)

            input_token, input_cost = num_tokens_from_messages(messages[idx], args.model, True)
            output_token, output_cost = num_tokens_from_messages(response, args.model, False)

            imple_message = [{'role': 'system', 'content': MBPP_system_message_2}, {'role': 'user', 'content': MBPP_implementation_prompt.format(problem=prompt, function_name=function_name, plan=response)}]
            response2 = call_chat_gpt(imple_message, args)

            input_token2, input_cost2 = num_tokens_from_messages(imple_message, args.model, True)
            output_token2, output_cost2 = num_tokens_from_messages(response2, args.model, False)

            input_token += input_token2
            input_cost += input_cost2
            output_token += output_token2
            output_cost += output_cost2

            code = process_generation_to_code(response2)
        
        result['response_code'] = '\n'.join(code)
        result['input_token'] = input_token
        result['input_cost'] = input_cost
        result['output_token'] = output_token
        result['output_cost'] = output_cost

        write_to_file(result, output_path)

