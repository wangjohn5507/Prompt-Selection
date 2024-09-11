import json
import os
import copy
import random
import math
from collections import defaultdict
from args import get_args
from src.evaluation import check_code, MBPP_check_code,APPS_check_code
from src.utils import write_to_file

random.seed(42)


def process_strategy_files(file_list):
    results = defaultdict(list)
    max_tokens = defaultdict(int)
    
   
    for file_path in file_list:
        with open(file_path, 'r') as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                task_id = idx
                exec_acc = entry['exec_acc']
                input_token = entry['input_token']
                output_token = entry['output_token']
                response_code = entry['response_code']
                total_tokens = input_token + output_token
                results[task_id].append({
                    "strategy": file_path.split('/')[-1],
                    "exec_acc": exec_acc,
                    "response_code": response_code,
                    "total_tokens": total_tokens
                })
                if total_tokens > max_tokens[task_id]:
                    max_tokens[task_id] = total_tokens
   
    sorted_results = {}
    for task_id, strategies in results.items():
        # print(results[task_id])
        max_total_tokens = max_tokens[task_id]
        # print(max_total_tokens)
        sorted_results[task_id] = sorted(
            strategies, 
            key=lambda x: (
                -(math.log(max_total_tokens) * x['exec_acc'] - math.log(x['total_tokens']))
            )
        )
    
    return sorted_results


def record_rank(origin_path, ranked_results, output_file, args):
    data = list(map(json.loads, open(origin_path)))
    for per_data, strategy_list in zip(data, ranked_results.values()):
        x = copy.copy(per_data)
        best_strategy = strategy_list[0]['strategy']
        
        if best_strategy == f"{args.dataset}_Zeroshot_{args.temperature}_{args.model}.jsonl":
            label = 'Zeroshot'
        elif best_strategy == f"{args.dataset}_Zeroshot_CoT_{args.temperature}_{args.model}.jsonl":
            label = 'Zeroshot_CoT'
        elif best_strategy == f"{args.dataset}_Fewshot_{args.temperature}_{args.model}.jsonl":
            label = 'Fewshot'
        elif best_strategy == f"{args.dataset}_Fewshot_CoT_{args.temperature}_{args.model}.jsonl":
            label = 'Fewshot_CoT'
        elif best_strategy == f"{args.dataset}_SelfDebug_{args.temperature}_{args.model}.jsonl":
            label = 'SelfDebug'
        elif best_strategy == f"{args.dataset}_Reflection_{args.temperature}_{args.model}.jsonl":
            label = 'Reflection'
        elif best_strategy == f"{args.dataset}_SelfPlan_{args.temperature}_{args.model}.jsonl":
            label = 'SelfPlan'
        elif best_strategy == f"{args.dataset}_ProgressiveHint_{args.temperature}_{args.model}.jsonl":
            label = 'ProgressiveHint'
        elif best_strategy == f"{args.dataset}_Persona_{args.temperature}_{args.model}.jsonl":
            label = 'Persona'

        x['best_strategy'] = label
        x['exec_acc'] = strategy_list[0]['exec_acc']
        x['exec_record'] = strategy_list
        write_to_file(x, output_file)

# def count_strategy_occurrences(filepath):
#     rank_counts = {f"Rank {i+1}": {} for i in range(5)}
    
#     with open(filepath, 'r', encoding='utf-8') as file:
#         data = json.load(file)
    
#     for task_id, strategies in data.items():
#         for index, strategy_info in enumerate(strategies):
#             strategy = strategy_info['strategy']
#             rank_key = f"Rank {index+1}"
#             if strategy in rank_counts[rank_key]:
#                 rank_counts[rank_key][strategy] += 1
#             else:
#                 rank_counts[rank_key][strategy] = 1
    
#     return rank_counts

# def count_strategy_rank_one(filepath):
#     strategy_rank_one = {}
    
#     with open(filepath, 'r', encoding='utf-8') as file:
#         data = json.load(file)
    
#     for task_id, strategies in data.items():
#         if strategies and 'strategy' in strategies[0]:
#             rank_one_strategy = strategies[0]['strategy'] 
#             if rank_one_strategy in strategy_rank_one:
#                 strategy_rank_one[rank_one_strategy].append(task_id)
#             else:
#                 strategy_rank_one[rank_one_strategy] = [task_id]
    
#     return strategy_rank_one

# def generate_random_strategy(strategy1_result, strategy1_rank_one, strategy2_result, strategy2_rank_one, strategy3_result, strategy3_rank_one):
#     strategies = {'Zeroshot':0, 'Zeroshot_CoT':0, 'Fewshot':0, 'Fewshot_CoT':0, 'SelfDebug':0}
#     strategies_rank_one = {'Zeroshot':[], 'Zeroshot_CoT':[], 'Fewshot':[], 'Fewshot_CoT':[], 'SelfDebug':[]}

#     def count_strategy(strategy_dict, strategies):
#         for strategy, num in strategy_dict['Rank 1'].items():
#             if 'Zeroshot_CoT' in strategy:
#                 strategies['Zeroshot_CoT'] += num
#             elif 'Zeroshot' in strategy:
#                 strategies['Zeroshot'] += num
#             elif 'Fewshot_CoT' in strategy:
#                 strategies['Fewshot_CoT'] += num
#             elif 'Fewshot' in strategy:
#                 strategies['Fewshot'] += num
#             elif 'SelfDebug' in strategy:
#                 strategies['SelfDebug'] += num
#         return strategies
    
#     def combine_strategies(strategy_rank_one, strategies_rank_one):
#         for strategy, list in strategy_rank_one.items():
#             if 'Zeroshot_CoT' in strategy:
#                 strategies_rank_one['Zeroshot_CoT'] += list
#             elif 'Zeroshot' in strategy:
#                 strategies_rank_one['Zeroshot'] += list
#             elif 'Fewshot_CoT' in strategy:
#                 strategies_rank_one['Fewshot_CoT'] += list
#             elif 'Fewshot' in strategy:
#                 strategies_rank_one['Fewshot'] += list
#             elif 'SelfDebug' in strategy:
#                 strategies_rank_one['SelfDebug'] += list
#         return strategies_rank_one
    
#     def random_choice(sorted_data, strategies_rank_one):
#         random_strategy = {'Zeroshot':[], 'Zeroshot_CoT':[], 'Fewshot':[], 'Fewshot_CoT':[], 'SelfDebug':[]}
#         num = sorted_data[0][1]
#         for strategy in sorted_data:
#             selected_task = random.sample(strategies_rank_one[strategy[0]], num)
#             random_strategy[strategy[0]] += selected_task
#         print(random_strategy)
#         return random_strategy
    
#     strategies = count_strategy(strategy1_result, strategies)
#     strategies = count_strategy(strategy2_result, strategies)
#     strategies = count_strategy(strategy3_result, strategies)
#     strategies_rank_one = combine_strategies(strategy1_rank_one, strategies_rank_one)
#     strategies_rank_one = combine_strategies(strategy2_rank_one, strategies_rank_one)
#     strategies_rank_one = combine_strategies(strategy3_rank_one, strategies_rank_one)

#     sorted_data = sorted(strategies.items(), key=lambda item: item[1], reverse=False)
#     print(sorted_data)
#     print('--------------------------------')
#     print(strategies_rank_one)
#     print('--------------------------------')

#     random_strategy = random_choice(sorted_data, strategies_rank_one)
#     return random_strategy

# def generate_balanced_dataset(file1, file2, file3, random_strategy):
#     data1 = list(map(json.loads, open(file1)))
#     data2 = list(map(json.loads, open(file2)))
#     data3 = list(map(json.loads, open(file3)))
#     # print(data2[0])
#     for strategy, data_list in random_strategy.items():
#         with open('result/balanced_dataset.jsonl','a')as f:
#             for data in data_list:
#                 for line in data1:
#                     if line['task_id'] == data:
#                         f.write(json.dumps(line) + '\n')
#                         break
#                 for line1 in data2:
#                     # print(line1['task_id'])
#                     if str(line1['task_id']) == data:
#                         f.write(json.dumps(line1) + '\n')
#                         break
#                 for line2 in data3:
#                     if str(line2['question_id']) == data:
#                         f.write(json.dumps(line2) + '\n')
#                         break
            
def main(args):
    directory_path = f"result/{args.dataset}_gpt4o_result_acc/"
    output_path = f'result/Final_result/{args.dataset}_gpt4o_nine_rank.jsonl'
    origin_path = f'dataset/{args.dataset}.jsonl'
    # List of file paths
    file_paths = os.listdir(directory_path)
    file_paths = [directory_path+file for file in file_paths]


    # Process all files
    final_results = process_strategy_files(file_paths)
    print(final_results)
    if args.finalize == True:
        record_rank(origin_path, final_results, output_path, args)
        print('--------------------------------')

        
        # result_fixed = count_strategy_occurrences(output_path)
        # print(result_fixed)
        # print('--------------------------------')
        # rank_one_result = count_strategy_rank_one(output_path)
        # print(rank_one_result)
        # print('--------------------------------')

    # if args.balance == True:
    #     HumanEval_result_fixed = count_strategy_occurrences('result/HumanEval_rank.json')
    #     HumanEval_rank_one_result = count_strategy_rank_one('result/HumanEval_rank.json')
    #     MBPP_result_fixed = count_strategy_occurrences('result/MBPP_rank.json')
    #     MBPP_rank_one_result = count_strategy_rank_one('result/MBPP_rank.json')
    #     APPS_result_fixed = count_strategy_occurrences('result/APPS_rank.json')
    #     APPS_rank_one_result = count_strategy_rank_one('result/APPS_rank.json')

    #     random_strategy = generate_random_strategy(HumanEval_result_fixed, HumanEval_rank_one_result, MBPP_result_fixed, MBPP_rank_one_result, APPS_result_fixed, APPS_rank_one_result)
    #     generate_balanced_dataset('result/HumanEval_rank.jsonl', 'result/MBPP_rank.jsonl', 'result/APPS_rank.jsonl', random_strategy)

if __name__ == '__main__':
    args = get_args()
    main(args)
