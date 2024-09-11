import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
import tqdm
import copy
from args import get_args

random.seed(42)
    
def find_samples(pos_data, neg_data, idx):
    pos_samples = []
    neg_samples = []
    for pos_idx, pos_per_data in enumerate(pos_data):
        if idx == pos_idx:
            continue
        else:
            pos_samples.append(pos_per_data)
    for neg_idx, neg_per_data in enumerate(neg_data):
        neg_samples.append(neg_per_data)
    
    pos_sample = random.choice(pos_samples)
    neg_sample = random.choice(neg_samples)
    
    return pos_sample, neg_sample

def get_contrastive_data(easy_data, hard_data):
    contrastive_data = []

    for idx, per_data in enumerate(tqdm.tqdm(hard_data)):
        pos_sample, neg_sample = find_samples(hard_data, easy_data, idx)
        contrastive_data.append({
            'anchor': per_data['question'],
            'positive': pos_sample['question'],
            'negative': neg_sample['question']
        })

    # for idx, per_data in enumerate(tqdm.tqdm(hard_data)):
    #     pos_sample, neg_sample = find_samples(hard_data, easy_data, idx)
    #     contrastive_data.append({
    #         'anchor': per_data['text'],
    #         'positive': pos_sample['text'],
    #         'negative': neg_sample['text']
    #     })
    return contrastive_data

def get_difficulty_data(data, args):
    easy_difficulty_data = []
    hard_difficulty_data = []
    if args.difficulty == 1:
        easy_difficulty = [0, 1, 2]
        hard_difficulty = [3, 4, 5]
    elif args.difficulty == 2:
        easy_difficulty = [0, 1]
        hard_difficulty = [4, 5]
    elif args.difficulty == 3:
        easy_difficulty = [0]
        hard_difficulty = [5]
    for per_data in data:
        difficulty = per_data['difficulty']
        if difficulty in easy_difficulty:
            easy_difficulty_data.append(per_data)
        elif difficulty in hard_difficulty:
            hard_difficulty_data.append(per_data)
    return easy_difficulty_data, hard_difficulty_data

def get_difficulty_classification_data(train_easy_data, train_hard_data, args):
    train_difficulty_classification_data = []
    type = args.type
    if type == 1:
        easy_label = [0, 1, 2, 3, 4, 5]
        hard_label = [4, 5] 
    elif type == 2:
        easy_label = [0, 1, 2, 3]
        hard_label = [4, 5]
    elif type == 3:
        easy_label = [0, 1, 2, 3, 4, 5]
        hard_label = [1, 2, 3, 4, 5]
    elif type == 4:
        easy_label = [0]
        hard_label = [1, 2, 3, 4, 5]
    elif type == 5:
        easy_label = [0, 1, 2, 3, 4, 5]
        hard_label = [0, 1, 2, 3, 4, 5]

    for per_data in train_easy_data:
        label = per_data['label']
        if label in easy_label:
            train_difficulty_classification_data.append(per_data)
    for per_data in train_hard_data:
        label = per_data['label']
        if label in hard_label:
            train_difficulty_classification_data.append(per_data)
    return train_difficulty_classification_data

def get_complexity_classification_data(train_easy_data, train_hard_data, args):
    # strategy_dict = {
    #     f'{args.dataset}_Zeroshot_{args.temperature}_{args.model}.jsonl': 0,
    #     f'{args.dataset}_Zeroshot_CoT_{args.temperature}_{args.model}.jsonl': 1,
    #     f'{args.dataset}_Fewshot_{args.temperature}_{args.model}.jsonl': 2,
    #     f'{args.dataset}_Fewshot_CoT_{args.temperature}_{args.model}.jsonl': 3,
    #     f'{args.dataset}_SelfDebug_{args.temperature}_{args.model}.jsonl': 4,
    #     f'{args.dataset}_Reflection_{args.temperature}_{args.model}.jsonl': 5,
    #     f'{args.dataset}_SelfPlan_{args.temperature}_{args.model}.jsonl': 6,
    #     f'{args.dataset}_ProgressiveHint_{args.temperature}_{args.model}.jsonl': 7,
    #     f'{args.dataset}_Persona_{args.temperature}_{args.model}.jsonl': 8
    # }
    train_complexity_classification_data = []
    type = args.type
    if type == 1:
        easy_label = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        hard_label = [5, 6, 7, 8] 
    elif type == 2:
        easy_label = [0, 1, 2, 3]
        hard_label = [4, 5, 6, 7, 8]
    elif type == 3:
        easy_label = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        hard_label = [1, 2, 3, 4, 5, 6, 7, 8]
    elif type == 4:
        easy_label = [0]
        hard_label = [1, 2, 3, 4, 5, 6, 7, 8]
    elif type == 5:
        easy_label = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        hard_label = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    for per_data in train_easy_data:
        # print(per_data.keys())
        if 'label' not in per_data.keys():
            continue
        label = per_data['label']
        if label in easy_label:
            train_complexity_classification_data.append(per_data)
    for per_data in train_hard_data:
        if 'label' not in per_data.keys():
            continue
        label = per_data['label']
        if label in hard_label:
            train_complexity_classification_data.append(per_data)

    return train_complexity_classification_data

def get_complexity_multi_classification_data(train_easy_data, train_hard_data, test_data, args):
    train_complexity_multi_classification_data = []
    test_complexity_multi_classification_data = []
    type = args.type
    easy_label = 0
    for per_data in train_easy_data:
        x = copy.copy(per_data)
        labels = []
        for idx, exec_record in enumerate(per_data['exec_record']):
            if exec_record['exec_acc'] == 1:
                labels.append(idx)
        x['labels'] = labels
        if easy_label in labels:
            train_complexity_multi_classification_data.append(x)
    for per_data in train_hard_data:
        x = copy.copy(per_data)
        labels = []
        for idx, exec_record in enumerate(per_data['exec_record']):
            if exec_record['exec_acc'] == 1:
                labels.append(idx)
        x['labels'] = labels
        # print(labels)
        if easy_label not in labels:
            print('hard' + str(labels))
            train_complexity_multi_classification_data.append(x)
    for per_data in test_data:
        x = copy.copy(per_data)
        labels = []
        for idx, exec_record in enumerate(per_data['exec_record']):
            if exec_record['exec_acc'] == 1:
                labels.append(idx)
        x['labels'] = labels
        test_complexity_multi_classification_data.append(x)
    return train_complexity_multi_classification_data, test_complexity_multi_classification_data


def get_complexity_data(data, args):
    easy_complexity_data = []
    hard_complexity_data = []
    complexity_threshold = args.complexity
    for per_data in data:
        complexity = per_data['complexity']
        if complexity < complexity_threshold:
            easy_complexity_data.append(per_data)
        elif complexity >= complexity_threshold:
            hard_complexity_data.append(per_data)
    return easy_complexity_data, hard_complexity_data


def write_difficulty_data(train_data, test_data, args):
    train_easy_difficulty_data, train_hard_difficulty_data = get_difficulty_data(train_data, args)
    test_easy_difficulty_data, test_hard_difficulty_data = get_difficulty_data(test_data, args)

    train_classification_data = get_difficulty_classification_data(train_easy_difficulty_data, train_hard_difficulty_data, args)

    train_contrastive_data = get_contrastive_data(train_easy_difficulty_data, train_hard_difficulty_data)
    test_contrastive_data = get_contrastive_data(test_easy_difficulty_data, test_hard_difficulty_data)

    train_contrastive_file_path = 'result/self_con_dataset/self_con_contrastive_hard_dataset_train_all.jsonl'
    test_contrastive_file_path = 'result/self_con_dataset/self_con_contrastive_hard_dataset_test_all.jsonl'
    train_classification_file_path = 'result/self_con_dataset/self_con_classification_hard_dataset_train_all.jsonl'
    test_classification_file_path = 'result/self_con_dataset/self_con_classification_hard_dataset_test_all.jsonl'

    with open(train_contrastive_file_path, 'w') as file:
        for item in train_contrastive_data:
            file.write(json.dumps(item) + '\n')

    with open(test_contrastive_file_path, 'w') as file:
        for item in test_contrastive_data:
            file.write(json.dumps(item) + '\n')
    
    with open(train_classification_file_path, 'w') as file:
        for item in train_classification_data:
            file.write(json.dumps(item) + '\n')
    
    with open(test_classification_file_path, 'w') as file:
        for item in test_data:
            file.write(json.dumps(item) + '\n')



def write_complexity_data(train_data, test_data, args):
    train_easy_complexity_data, train_hard_complexity_data = get_complexity_data(train_data, args)
    test_easy_complexity_data, test_hard_complexity_data = get_complexity_data(test_data, args)

    train_classification_data = get_complexity_classification_data(train_easy_complexity_data, train_hard_complexity_data, args)
    # train_complexity_multi_classification_data, test_complexity_multi_classification_data = get_complexity_multi_classification_data(train_easy_complexity_data, train_hard_complexity_data, test_data, args)

    train_contrastive_data = get_contrastive_data(train_easy_complexity_data, train_hard_complexity_data)
    test_contrastive_data = get_contrastive_data(test_easy_complexity_data, test_hard_complexity_data)

    train_contrastive_file_path = 'result/code_complex_dataset/code_complex_contrastive_hard_dataset_train_all.jsonl'
    test_contrastive_file_path = 'result/code_complex_dataset/code_complex_contrastive_hard_dataset_test_all.jsonl'
    train_classification_file_path = 'result/code_complex_dataset/code_complex_classification_hard_dataset_train_all.jsonl'
    test_classification_file_path = 'result/code_complex_dataset/code_complex_classification_hard_dataset_test_all.jsonl'
    # train_classification_file_path = 'result/code_complex_dataset/code_complex_multi_classification_hard_dataset_train_all.jsonl'
    # test_classification_file_path = 'result/code_complex_dataset/code_complex_multi_classification_hard_dataset_test_all.jsonl'

    with open(train_contrastive_file_path, 'w') as file:
        for item in train_contrastive_data:
            file.write(json.dumps(item) + '\n')

    with open(test_contrastive_file_path, 'w') as file:
        for item in test_contrastive_data:
            file.write(json.dumps(item) + '\n')

    with open(train_classification_file_path, 'w') as file:
        for item in train_classification_data:
            file.write(json.dumps(item) + '\n')
        # for item in train_complexity_multi_classification_data:
        #     file.write(json.dumps(item) + '\n')
    
    with open(test_classification_file_path, 'w') as file:
        for item in test_data:
            file.write(json.dumps(item) + '\n')
        # for item in test_complexity_multi_classification_data:
        #     file.write(json.dumps(item) + '\n')


def main(args):
    # train_file_path = f'result/Final_result/{args.dataset}_gpt4o_nine_train.jsonl'
    train_file_path = f'result/Final_result_5fold/{args.dataset}_nine_train_0.jsonl'
    train_data = list(map(json.loads, open(train_file_path)))
    # test_file_path = f'result/Final_result/{args.dataset}_gpt4o_nine_test.jsonl'
    test_file_path = f'result/Final_result_5fold/{args.dataset}_nine_test_0.jsonl'
    test_data = list(map(json.loads, open(test_file_path)))

    if args.dataset == 'MBPP':
        # write_difficulty_data(train_data, test_data, args)
        write_complexity_data(train_data, test_data, args)
    elif args.dataset == 'HumanEval':
        write_complexity_data(train_data, test_data, args)
    elif args.dataset == 'Combined':
        train_file_path = f'result/Final_result/{args.dataset}_train.jsonl'
        train_data = list(map(json.loads, open(train_file_path)))
        test_file_path = f'result/Final_result/HumanEval_test.jsonl'
        test_data = list(map(json.loads, open(test_file_path)))
        write_complexity_data(train_data, test_data)

    



if __name__ == '__main__':
    args = get_args()
    main(args)
