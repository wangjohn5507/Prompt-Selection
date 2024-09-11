import json
import copy
import ast
from args import get_args
from src.utils import write_to_file
from sklearn.model_selection import train_test_split, KFold
from src.utils import calculate_cyclomatic_complexity, count_physical_loc, calculate_halstead_complexity, calculate_mi, calculate_cognitive_complexity
from src.utils import extract_exec_code


def get_success_list(rank_list):
    success_list = []
    for per_data in rank_list:
        if per_data['exec_acc'] == 1:
            success_list.append(per_data)
    return success_list

def get_failure_list(rank_list):
    failure_list = []
    for per_data in rank_list:
        if per_data['exec_acc'] == 0:
            failure_list.append(per_data)
    return failure_list

def get_label_success_list(success_list, strategy_dict):
    label_success_list = []
    for per_data in success_list:
        x = copy.copy(per_data)
        best_strategy = per_data['exec_record'][0]['strategy']
        x['label'] = strategy_dict[best_strategy]
        label_success_list.append(x)
    return label_success_list

def get_rank_label_success_list(label_success_list, strategy_dict):
    rank_label_success_list = []
    for per_data in label_success_list:
        rank = {}
        x = copy.copy(per_data)
        exec_record = per_data['exec_record']
        initial_rank = 9
        for record in exec_record:
            strategy = record['strategy']
            strategy_exec_acc = record['exec_acc']
            strategy_label = str(strategy_dict[strategy])
            
            if strategy_exec_acc == 0:
                rank[strategy_label] = 0
            else:
                rank[strategy_label] = initial_rank
                initial_rank -= 1
        x['rank'] = rank
        rank_label_success_list.append(x)
    return rank_label_success_list

def generate_exec_array(file_path):
    data = list(map(json.loads, open(file_path)))
    exec_array = []
    for entry in data:
        exec_array.append(entry['exec_acc'])
    return exec_array

def get_difficulty_list(rank_list, args):
    difficuly_list = []

    file_path_1 = f'result/{args.dataset}_selfcon_acc/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}_1.jsonl'
    file_path_2 = f'result/{args.dataset}_selfcon_acc/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}_2.jsonl'
    file_path_3 = f'result/{args.dataset}_selfcon_acc/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}_3.jsonl'
    file_path_4 = f'result/{args.dataset}_selfcon_acc/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}_4.jsonl'
    file_path_5 = f'result/{args.dataset}_selfcon_acc/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}_5.jsonl'

    exec_1 = generate_exec_array(file_path_1)
    exec_2 = generate_exec_array(file_path_2)
    exec_3 = generate_exec_array(file_path_3)
    exec_4 = generate_exec_array(file_path_4)
    exec_5 = generate_exec_array(file_path_5)

    diff0  = []
    diff1 = []
    diff2 = []
    diff3 = []
    diff4 = []
    diff5 = []

    for idx, (exec1, exec2, exec3, exec4, exec5) in enumerate(zip(exec_1, exec_2, exec_3, exec_4, exec_5)):
        combined = [exec1, exec2, exec3, exec4, exec5]
    
        if sum(combined) == 5:
            diff0.append(idx)
        elif sum(combined) == 4:
            diff1.append(idx)
        elif sum(combined) == 3:
            diff2.append(idx)
        elif sum(combined) == 2:
            diff3.append(idx)
        elif sum(combined) == 1:
            diff4.append(idx)
        elif sum(combined) == 0:
            diff5.append(idx)

    for idx, per_data in enumerate(rank_list):
        x = copy.copy(per_data)
        if idx in diff0:
            x['difficulty'] = 0
        elif idx in diff1:
            x['difficulty'] = 1
        elif idx in diff2:
            x['difficulty'] = 2
        elif idx in diff3:
            x['difficulty'] = 3
        elif idx in diff4:
            x['difficulty'] = 4
        elif idx in diff5:
            x['difficulty'] = 5
        difficuly_list.append(x)
    
    return difficuly_list

def get_complexity_list(difficulty_list, args):
    complexity_list = []
    for per_data in difficulty_list:
        x = copy.copy(per_data)
        if args.dataset == 'MBPP':
            code = per_data['code']
            question = per_data['text']
        elif args.dataset == 'HumanEval':
            code = 'def function():\n' + per_data['canonical_solution']
            question = per_data['prompt']
        elif args.dataset == 'DS1000':
            code = per_data['reference_code']
            exec_context = extract_exec_code(per_data['code_context'])
            code = exec_context.replace("[insert]", code)
            question = per_data['prompt']
        try:
            line_complexity = count_physical_loc(code)
            cyclo_complexity = calculate_cyclomatic_complexity(code)
            halstead_complexity = calculate_halstead_complexity(code)
            mi_complexity = calculate_mi(code)
            cognitive_complexity = calculate_cognitive_complexity(code)
            total_complexity = line_complexity + cyclo_complexity + halstead_complexity + mi_complexity + cognitive_complexity
            x['complexity'] = total_complexity 
            x['question'] = question
            complexity_list.append(x)
        except Exception as e:
            print('error')
            # for node in ast.parse(code).body:
            #     print(node)
            print(e)
            continue
    return complexity_list


def main(args):
    strategy_dict = {
        f'{args.dataset}_Zeroshot_{args.temperature}_{args.model}.jsonl': 0,
        f'{args.dataset}_Zeroshot_CoT_{args.temperature}_{args.model}.jsonl': 1,
        f'{args.dataset}_Fewshot_{args.temperature}_{args.model}.jsonl': 2,
        f'{args.dataset}_Fewshot_CoT_{args.temperature}_{args.model}.jsonl': 3,
        f'{args.dataset}_SelfDebug_{args.temperature}_{args.model}.jsonl': 4,
        f'{args.dataset}_Reflection_{args.temperature}_{args.model}.jsonl': 5,
        f'{args.dataset}_SelfPlan_{args.temperature}_{args.model}.jsonl': 6,
        f'{args.dataset}_ProgressiveHint_{args.temperature}_{args.model}.jsonl': 7,
        f'{args.dataset}_Persona_{args.temperature}_{args.model}.jsonl': 8
    }

    output_file_path = f'result/Final_result/{args.dataset}_nine_final_result.jsonl'
    train_output_file_path = f'result/Final_result/{args.dataset}_nine_train.jsonl'
    test_output_file_path = f'result/Final_result/{args.dataset}_nine_test.jsonl'

    if args.dataset == 'MBPP':
        rank_file_path = f'result/Final_result/{args.dataset}_nine_rank.jsonl'
        rank_list = list(map(json.loads, open(rank_file_path)))
        # difficult_list = get_difficulty_list(rank_list, args)
        complexity_list = get_complexity_list(rank_list, args)
    elif args.dataset == 'HumanEval':
        rank_file_path = f'result/Final_result/{args.dataset}_nine_rank.jsonl'
        rank_list = list(map(json.loads, open(rank_file_path)))
        complexity_list = get_complexity_list(rank_list, args)
    elif args.dataset == 'DS1000':
        rank_list = list(map(json.loads, open('dataset/DS1000.jsonl')))
        complexity_list = get_complexity_list(rank_list, args)

    if args.dataset == 'MBPP':
        success_list = get_success_list(complexity_list)
        fail_list = get_failure_list(complexity_list)

        label_success_list = get_label_success_list(success_list, strategy_dict)
        rank_label_success_list = get_rank_label_success_list(label_success_list, strategy_dict)

        label_failure_list = get_label_success_list(fail_list, strategy_dict)
        rank_label_failure_list = get_rank_label_success_list(label_failure_list, strategy_dict)

        for per_data in rank_label_success_list:
            write_to_file(per_data, output_file_path)
        for per_data in rank_label_failure_list:
            write_to_file(per_data, output_file_path)

        # train, test = train_test_split(rank_label_success_list, test_size=0.2, random_state=42)
        # for per_data in train:
        #     write_to_file(per_data, train_output_file_path)
        # for per_data in test:
        #     write_to_file(per_data, test_output_file_path)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        idx = 0
        idx1 = 0

        for train_index, test_index in kf.split(rank_label_success_list):
            for train_per_index in train_index:
                train = rank_label_success_list[train_per_index]
                write_to_file(train, f'result/Final_result_5fold/{args.dataset}_nine_train_{idx}.jsonl')
            for test_per_index in test_index:
                test = rank_label_success_list[test_per_index]
                write_to_file(test, f'result/Final_result_5fold/{args.dataset}_nine_test_{idx}.jsonl')
            idx += 1

        for train_index, test_index in kf.split(rank_label_failure_list):
            for test_per_index in test_index:
                test = rank_label_failure_list[test_per_index]
                write_to_file(test, f'result/Final_result_5fold/{args.dataset}_nine_test_{idx1}.jsonl')
            idx1 += 1
    
    elif args.dataset == 'HumanEval':
        success_list = get_success_list(complexity_list)
        fail_list = get_failure_list(complexity_list)

        label_success_list = get_label_success_list(success_list, strategy_dict)
        rank_label_success_list = get_rank_label_success_list(label_success_list, strategy_dict)

        label_failure_list = get_label_success_list(fail_list, strategy_dict)
        rank_label_failure_list = get_rank_label_success_list(label_failure_list, strategy_dict)

        for per_data in rank_label_success_list:
            write_to_file(per_data, output_file_path)
        for per_data in rank_label_failure_list:
            write_to_file(per_data, output_file_path)

        # train, test = train_test_split(rank_label_success_list, test_size=0.2, random_state=42)
        # for per_data in train:
        #     write_to_file(per_data, train_output_file_path)
        # for per_data in test:
        #     write_to_file(per_data, test_output_file_path)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        idx = 0
        idx1 = 0

        for train_index, test_index in kf.split(rank_label_success_list):
            for train_per_index in train_index:
                train = rank_label_success_list[train_per_index]
                write_to_file(train, f'result/Final_result_5fold/{args.dataset}_nine_train_{idx}.jsonl')
            for test_per_index in test_index:
                test = rank_label_success_list[test_per_index]
                write_to_file(test, f'result/Final_result_5fold/{args.dataset}_nine_test_{idx}.jsonl')
            idx += 1

        for train_index, test_index in kf.split(rank_label_failure_list):
            for test_per_index in test_index:
                test = rank_label_failure_list[test_per_index]
                write_to_file(test, f'result/Final_result_5fold/{args.dataset}_nine_test_{idx1}.jsonl')
            idx1 += 1

    else:
        train_output_file_path = 'result/Final_result/MBPP_train_extend.jsonl'
        for per_data in complexity_list:
            write_to_file(per_data, train_output_file_path)



if __name__ == '__main__':
    args = get_args()
    main(args)