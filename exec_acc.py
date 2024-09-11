import json
import copy
from args import get_args
from src.evaluation import check_code, MBPP_check_code, APPS_check_code, DS1000_check_code
from src.utils import write_to_file


def evaluate(args, file_path, output_path):
    # File path
    data = list(map(json.loads, open(file_path)))
    print(data[0])

    success = 0

    for per_data in data:
        exec_acc = 0
        result = copy.copy(per_data)
        
        if args.dataset == 'HumanEval':
            prompt = per_data['prompt']
            final = per_data['response_code']
            test = per_data['test']
            entry_point = per_data['entry_point']
            if check_code(prompt, final, test, entry_point):
                exec_acc = 1
                success += 1
        elif args.dataset == 'MBPP':
            final = per_data['response_code']
            checked = True
            for test in per_data['test_list']:
                if MBPP_check_code(final, test) == False:
                    checked = False
                    break
            if checked == True:
                exec_acc = 1
                success += 1
        elif args.dataset == 'APPS':
            final = per_data['response_code']
            checked = True
            for idx, input in enumerate(per_data['inputs']):
                output = per_data['outputs'][idx]
                if APPS_check_code(final, input, output) == False:
                    checked = False
                    break
            if checked == True:
                exec_acc = 1
                success += 1
        elif args.dataset == 'DS1000':
            final = per_data['response_code']
            test_code = per_data['code_context']
            checked = True
            if DS1000_check_code(final, test_code):
                exec_acc = 1
                success += 1
        result['exec_acc'] = exec_acc
        write_to_file(result, output_path)


    print('ACC: '+str(success/len(data)))

def calulate_cost_token(file_path):
    # Initialize variables to store sums and counts
    input_cost_sum = 0
    output_cost_sum = 0
    input_token_count = 0
    output_token_count = 0
    total_entries = 0

    # Read the JSONL file line by line
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            # Sum the costs
            input_cost_sum += data['input_cost']
            output_cost_sum += data['output_cost']
            
            # Count the tokens
            input_token_count += data['input_token']
            output_token_count += data['output_token']
            
            # Increment the total number of entries
            total_entries += 1

    # Calculate the averages
    average_input_tokens = input_token_count / total_entries
    average_output_tokens = output_token_count / total_entries

    total_tokens = average_input_tokens + average_output_tokens

    print(input_cost_sum, output_cost_sum, average_input_tokens, average_output_tokens, total_tokens)

def main(args):
    file_path = f'result/{args.dataset}_result/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}.jsonl'
    output_path = f'result/{args.dataset}_result_acc/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}.jsonl'
    if args.evaluation == True:
        evaluate(args, file_path, output_path)
    calulate_cost_token(file_path)

if __name__ == '__main__':
    args = get_args()
    main(args)