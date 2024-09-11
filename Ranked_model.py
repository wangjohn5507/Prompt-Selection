import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import random
import tqdm
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2)

# Customize Dataset
class CustomDataset(Dataset):
    def __init__(self, embeddings, labels, ranks):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ranks = ranks

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.ranks[idx]
    
from sentence_transformers import SentenceTransformer
def get_embedding(questions, model_path):
    model = SentenceTransformer(model_path)
    embeddings = []
    print('Generating embeddings...')
    for question in tqdm.tqdm(questions):
        embedding = model.encode(question)
        embeddings.append(embedding)
    return embeddings

class ClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

# Check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = 'code_complex'
hard = '_hard'
data_type = '_all'
file_path = f'result/{dataset}_dataset/{dataset}_classification{hard}_dataset_train{data_type}.jsonl'
test_file_path = f'result/{dataset}_dataset/{dataset}_classification{hard}_dataset_test{data_type}.jsonl'

model_path = f'result/{dataset}_contrastive{hard}{data_type}_model'
data = pd.read_json(file_path, lines=True)
test_data = pd.read_json(test_file_path, lines=True)

questions = data['text'].tolist()
labels = data['label'].tolist()
ranks = data['rank'].tolist()

test_questions = test_data['text'].tolist()
test_labels = test_data['label'].tolist()
test_ranks = test_data['rank'].tolist()

# embeddings = data['embedding'].tolist()
# test_embeddings = test_data['embedding'].tolist()

embeddings = get_embedding(questions, model_path)
test_embeddings = get_embedding(test_questions, model_path)

def dcg(relevances, k):
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def ndcg(relevances, k):
    dcg_value = dcg(relevances, k)
    idcg_value = dcg(sorted(relevances, reverse=True), k)
    if idcg_value == 0:
        return 0.0
    return dcg_value / idcg_value

def custom_collate_fn(batch):
    embeddings = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    relevance_scores = [item[2] for item in batch]
    return embeddings, labels, relevance_scores

# hyperparameter setting
input_size = len(embeddings[0])
print(input_size)
num_classes = 9
print(num_classes)
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Split train, eval, and test data
# train_embeddings, temp_embeddings, train_labels, temp_labels, train_ranks, temp_ranks = train_test_split(embeddings, labels, ranks, test_size=0.3, random_state=42)
# eval_embeddings, test_embeddings, eval_labels, test_labels, eval_ranks, test_ranks = train_test_split(temp_embeddings, temp_labels, temp_ranks, test_size=0.5, random_state=42)

train_embeddings, eval_embeddings, train_labels, eval_labels, train_ranks, eval_ranks = train_test_split(embeddings, labels, ranks, test_size=0.2, random_state=42)
test_embeddings, test_labels, test_ranks = test_embeddings, test_labels, test_ranks


# create CustomDataset
train_dataset = CustomDataset(train_embeddings, train_labels, train_ranks)
eval_dataset = CustomDataset(eval_embeddings, eval_labels, eval_ranks)
test_dataset = CustomDataset(test_embeddings, test_labels, test_ranks)

# create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = [0.5,  0.5,  0.5,  0.5,  1, 0.5]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
# print(class_weights)

# create model, criterion, and optimizer
model = ClassificationModel(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# calcuate MRR
def calculate_mrr(outputs, labels):
    mrr = 0.0
    for i in range(outputs.size(0)):
        scores = outputs[i]
        target = labels[i].item()
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1
        mrr += 1.0 / rank
    return mrr / outputs.size(0)

#calculate ndcg
def calculate_ndcg(outputs, ranks, k=6):
    ndcg_value = 0.0
    for i in range(outputs.size(0)):
        rank = ranks[i]
        scores = outputs[i]
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        relevances = np.array([rank[str(idx.item())] for idx in sorted_indices])
        ndcg_value += ndcg(relevances, k)
    return ndcg_value / outputs.size(0)

# train_model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for embeddings_batch, labels_batch, ranks_batch in train_dataloader:
        # print(ranks_batch)
        embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)

        outputs = model(embeddings_batch)
        loss = criterion(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    epoch_acc = 100 * correct / total


    # evaluate model
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    val_mrr = 0.0
    val_ndcg = 0.0
    val_ndcg_total = 0
    with torch.no_grad():
        for val_embeddings_batch, val_labels_batch, val_ranks_batch in eval_dataloader:
            val_embeddings_batch, val_labels_batch = val_embeddings_batch.to(device), val_labels_batch.to(device)

            val_outputs = model(val_embeddings_batch)
            val_loss = criterion(val_outputs, val_labels_batch)
            val_running_loss += val_loss.item()

            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels_batch.size(0)
            val_ndcg_total += val_outputs.size(0)
            val_correct += (val_predicted == val_labels_batch).sum().item()

            val_mrr += calculate_mrr(val_outputs, val_labels_batch) * val_labels_batch.size(0)
            val_ndcg += calculate_ndcg(val_outputs, val_ranks_batch, num_classes) * val_outputs.size(0)

    val_epoch_loss = val_running_loss / len(eval_dataloader)
    val_epoch_acc = 100 * val_correct / val_total
    val_epoch_mrr = val_mrr / val_total
    val_epoch_ndcg = val_ndcg / val_ndcg_total

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}%, Val MRR: {val_epoch_mrr:.4f}, Val nDCG: {val_epoch_ndcg:.4f}')


def evaluate_nDCG(model, dataloader, device, k=6):
    model.eval()
    ndcg_total = 0.0
    total = 0
    with torch.no_grad():
        for embeddings_batch, labels_batch, rank_batch in dataloader:
            embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)

            outputs = model(embeddings_batch)

            for i in range(outputs.size(0)):
                rank = rank_batch[i]
                # print(rank)
                scores = outputs[i]
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)

                # first_two_choices = [0, 4]
                # remaining_choices = [1, 2, 3, 5]
                # # Generate the list
                # random_labels = random.sample(first_two_choices, 2) + random.sample(remaining_choices, 4)
                # # print(target)
                # sorted_indices = torch.tensor(random_labels)
                
                relevances = np.array([rank[str(idx.item())] for idx in sorted_indices])
                # print(relevances)
                ndcg_total += ndcg(relevances, k)
                total += 1

    ndcg_avg = ndcg_total / total
    print(f'nDCG@{k}: {ndcg_avg:.4f}')

def evaluate_mrr(model, dataloader, device):
    model.eval()
    mrr = 0.0
    total = 0
    with torch.no_grad():
        for embeddings_batch, labels_batch, _ in dataloader:
            embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)

            outputs = model(embeddings_batch)
            for i in range(outputs.size(0)):
                scores = outputs[i]
                # print(scores)
                target = labels_batch[i].item()
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                print(target, sorted_indices)

                # # Define the lists for random choices
                # first_two_choices = [0, 4]
                # remaining_choices = [1, 2, 3, 5]

                # # Generate the list
                # random_labels = random.sample(first_two_choices, 2) + random.sample(remaining_choices, 4)
                # # print(target)
                # sorted_indices = torch.tensor(random_labels)

                rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1
                mrr += 1.0 / rank
                total += 1

    mrr = mrr / total
    print(f'MRR: {mrr:.4f}')

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings_batch, labels_batch, _ in dataloader:
            embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)

            outputs = model(embeddings_batch)
            _, predicted = torch.max(outputs.data, 1)

            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')


evaluate_nDCG(model, test_dataloader, device, num_classes)
evaluate_mrr(model, test_dataloader, device)
evaluate_model(model, test_dataloader, device)

output_model_path = f'result/classification_model/{dataset}_classification{hard}{data_type}_model_parameters.pth'
torch.save(model.state_dict(), output_model_path)

def generate_data_list(data, test_data):
    data_list = []
    test_data_list = []
    for per_test in test_data:
        test_data_list.append(per_test['text'])
    for per_data in data:
        if per_data['text'] in test_data_list:
            data_list.append(per_data)
    return data_list

def calculate_actual_acc(data_list, model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)


    questions = [per_data['text'] for per_data in data_list]
    embeddings = get_embedding(questions, model_path)


    if isinstance(embeddings, list):
        embeddings = torch.tensor(embeddings).to(device)


    with torch.no_grad():
        outputs = model(embeddings)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()


    correct = 0
    zero_shot_correct = 0
    zero_cot_correct = 0
    few_shot_correct = 0
    few_cot_correct = 0
    self_correct = 0
    reflection_correct = 0
    self_plan_correct = 0
    progressive_hint_correct = 0
    persona_correct = 0
    total = len(data_list)


    strategy_dict = {
        0: 'MBPP_Zeroshot_0.0_gpt-3.5-turbo-0125.jsonl',
        1: 'MBPP_Zeroshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl',
        2: 'MBPP_Fewshot_0.0_gpt-3.5-turbo-0125.jsonl',
        3: 'MBPP_Fewshot_CoT_0.0_gpt-3.5-turbo-0125jsonl',
        4: 'MBPP_SelfDebug_0.0_gpt-3.5-turbo-0125.jsonl',
        5: 'MBPP_Reflection_0.0_gpt-3.5-turbo-0125.jsonl',
        6: 'MBPP_SelfPlan_0.0_gpt-3.5-turbo-0125.jsonl',
        7: 'MBPP_ProgressiveHint_0.0_gpt-3.5-turbo-0125.jsonl',
        8: 'MBPP_Persona_0.0_gpt-3.5-turbo-0125.jsonl'
    }


    for idx, per_data in enumerate(data_list):
        # y_pred = predictions[idx]
        # y_pred = random.choice([0, 4])
        y_pred = random.randint(0, 8)
        strategy = strategy_dict[y_pred]
        exec_record = per_data['exec_record']

        # print(y_pred, per_data['label'])

        for exec in exec_record:
            exec_strategy = exec['strategy']
            exec_acc = exec['exec_acc']

            if exec_strategy == strategy and exec_acc == 1:
                correct += 1
            if exec_strategy == 'MBPP_Zeroshot_0.0_gpt-3.5-turbo-0125.jsonl' and exec_acc == 1:
                zero_shot_correct += 1
            if exec_strategy == 'MBPP_Zeroshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl' and exec_acc == 1:
                zero_cot_correct += 1
            if exec_strategy == 'MBPP_Fewshot_0.0_gpt-3.5-turbo-0125.jsonl' and exec_acc == 1:
                few_shot_correct += 1
            if exec_strategy == 'MBPP_Fewshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl' and exec_acc == 1:
                few_cot_correct += 1
            if exec_strategy == 'MBPP_SelfDebug_0.0_gpt-3.5-turbo-0125.jsonl' and exec_acc == 1:
                self_correct += 1
            if exec_strategy == 'MBPP_Reflection_0.0_gpt-3.5-turbo-0125.jsonl' and exec_acc == 1:
                reflection_correct += 1
            if exec_strategy == 'MBPP_SelfPlan_0.0_gpt-3.5-turbo-0125.jsonl' and exec_acc == 1:
                self_plan_correct += 1
            if exec_strategy == 'MBPP_ProgressiveHint_0.0_gpt-3.5-turbo-0125.jsonl' and exec_acc == 1:
                progressive_hint_correct += 1
            if exec_strategy == 'MBPP_Persona_0.0_gpt-3.5-turbo-0125.jsonl' and exec_acc == 1:
                persona_correct += 1

    actual_acc = correct / total
    zero_shot_acc = zero_shot_correct / total
    zero_cot_acc = zero_cot_correct / total
    few_shot_acc = few_shot_correct / total
    few_cot_acc = few_cot_correct / total
    self_acc = self_correct / total
    reflection_acc = reflection_correct / total
    self_plan_acc = self_plan_correct / total
    progressive_hint_acc = progressive_hint_correct / total
    persona_acc = persona_correct / total

    return actual_acc, zero_shot_acc, zero_cot_acc, few_shot_acc, few_cot_acc, self_acc, reflection_acc, self_plan_acc, progressive_hint_acc, persona_acc, zero_shot_correct

def calculate_token_saved(data_list, model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)


    questions = [per_data['text'] for per_data in data_list]
    embeddings = get_embedding(questions, model_path)


    if isinstance(embeddings, list):
        embeddings = torch.tensor(embeddings).to(device)


    with torch.no_grad():
        outputs = model(embeddings)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()


    pred_token = 0
    zero_shot_token = 0
    zero_cot_token = 0
    few_shot_token = 0
    few_cot_token = 0
    self_token = 0
    reflection_token = 0
    self_plan_token = 0
    progressive_hint_token = 0
    persona_token = 0
    total = len(data_list)


    strategy_dict = {
        0: 'MBPP_Zeroshot_0.0_gpt-3.5-turbo-0125.jsonl',
        1: 'MBPP_Zeroshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl',
        2: 'MBPP_Fewshot_0.0_gpt-3.5-turbo-0125.jsonl',
        3: 'MBPP_Fewshot_CoT_0.0_gpt-3.5-turbo-0125jsonl',
        4: 'MBPP_SelfDebug_0.0_gpt-3.5-turbo-0125.jsonl',
        5: 'MBPP_Reflection_0.0_gpt-3.5-turbo-0125.jsonl',
        6: 'MBPP_SelfPlan_0.0_gpt-3.5-turbo-0125.jsonl',
        7: 'MBPP_ProgressiveHint_0.0_gpt-3.5-turbo-0125.jsonl',
        8: 'MBPP_Persona_0.0_gpt-3.5-turbo-0125.jsonl'
    }


    for idx, per_data in enumerate(data_list):
        # y_pred = predictions[idx]
        # y_pred = random.choice([0, 4])
        y_pred = random.randint(0, 8)
        strategy = strategy_dict[y_pred]
        exec_record = per_data['exec_record']

        for exec in exec_record:
            exec_strategy = exec['strategy']
            token = exec['total_tokens']

            if exec_strategy == strategy:
                pred_token += token
            if exec_strategy == 'MBPP_Zeroshot_0.0_gpt-3.5-turbo-0125.jsonl':
                zero_shot_token += token
            if exec_strategy == 'MBPP_Zeroshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl':
                zero_cot_token += token
            if exec_strategy == 'MBPP_Fewshot_0.0_gpt-3.5-turbo-0125.jsonl':
                few_shot_token += token
            if exec_strategy == 'MBPP_Fewshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl':
                few_cot_token += token
            if exec_strategy == 'MBPP_SelfDebug_0.0_gpt-3.5-turbo-0125.jsonl':
                self_token += token
            if exec_strategy == 'MBPP_Reflection_0.0_gpt-3.5-turbo-0125.jsonl':
                reflection_token += token
            if exec_strategy == 'MBPP_SelfPlan_0.0_gpt-3.5-turbo-0125.jsonl':
                self_plan_token += token
            if exec_strategy == 'MBPP_ProgressiveHint_0.0_gpt-3.5-turbo-0125.jsonl':
                progressive_hint_token += token
            if exec_strategy == 'MBPP_Persona_0.0_gpt-3.5-turbo-0125.jsonl':
                persona_token += token

    pred_avg_token = pred_token / total
    zero_shot_avg_token = zero_shot_token / total
    zero_cot_avg_token = zero_cot_token / total
    few_shot_avg_token = few_shot_token / total
    few_cot_avg_token = few_cot_token / total
    self_avg_token = self_token / total
    reflection_avg_token = reflection_token / total
    self_plan_avg_token = self_plan_token / total
    progressive_hint_avg_token = progressive_hint_token / total
    persona_avg_token = persona_token / total

    return pred_avg_token, zero_shot_avg_token, zero_cot_avg_token, few_shot_avg_token, few_cot_avg_token, self_avg_token, reflection_avg_token, self_plan_avg_token, progressive_hint_avg_token, persona_avg_token


def calculate_avg_rank(data_list, model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)


    questions = [per_data['text'] for per_data in data_list]
    embeddings = get_embedding(questions, model_path)


    if isinstance(embeddings, list):
        embeddings = torch.tensor(embeddings).to(device)


    with torch.no_grad():
        outputs = model(embeddings)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()


    pred_rank = 0
    zero_shot_rank = 0
    zero_cot_rank = 0
    few_shot_rank = 0
    few_cot_rank = 0
    self_rank = 0
    reflection_rank = 0
    self_plan_rank = 0
    progressive_hint_rank = 0
    persona_rank = 0
    total = len(data_list)


    strategy_dict = {
        0: 'MBPP_Zeroshot_0.0_gpt-3.5-turbo-0125.jsonl',
        1: 'MBPP_Zeroshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl',
        2: 'MBPP_Fewshot_0.0_gpt-3.5-turbo-0125.jsonl',
        3: 'MBPP_Fewshot_CoT_0.0_gpt-3.5-turbo-0125jsonl',
        4: 'MBPP_SelfDebug_0.0_gpt-3.5-turbo-0125.jsonl',
        5: 'MBPP_Reflection_0.0_gpt-3.5-turbo-0125.jsonl',
        6: 'MBPP_SelfPlan_0.0_gpt-3.5-turbo-0125.jsonl',
        7: 'MBPP_ProgressiveHint_0.0_gpt-3.5-turbo-0125.jsonl',
        8: 'MBPP_Persona_0.0_gpt-3.5-turbo-0125.jsonl'
    }

    for idx, per_data in enumerate(data_list):
        # y_pred = predictions[idx]
        # y_pred = random.choice([0, 4])
        y_pred = random.randint(0, 8)
        strategy = strategy_dict[y_pred]
        exec_record = per_data['exec_record']

        for idx1, exec in enumerate(exec_record):
            exec_strategy = exec['strategy']
            exec_acc = exec['exec_acc']

            if exec_acc == 0:
                idx1 = 9
            else:
                idx1 += 1

            if exec_strategy == strategy:
                pred_rank += idx1
            if exec_strategy == 'MBPP_Zeroshot_0.0_gpt-3.5-turbo-0125.jsonl':
                zero_shot_rank += idx1
            if exec_strategy == 'MBPP_Zeroshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl':
                zero_cot_rank += idx1
            if exec_strategy == 'MBPP_Fewshot_0.0_gpt-3.5-turbo-0125.jsonl':
                few_shot_rank += idx1
            if exec_strategy == 'MBPP_Fewshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl':
                few_cot_rank += idx1
            if exec_strategy == 'MBPP_SelfDebug_0.0_gpt-3.5-turbo-0125.jsonl':
                self_rank += idx1
            if exec_strategy == 'MBPP_Reflection_0.0_gpt-3.5-turbo-0125.jsonl':
                reflection_rank += idx1
            if exec_strategy == 'MBPP_SelfPlan_0.0_gpt-3.5-turbo-0125.jsonl':
                self_plan_rank += idx1
            if exec_strategy == 'MBPP_ProgressiveHint_0.0_gpt-3.5-turbo-0125.jsonl':
                progressive_hint_rank += idx1
            if exec_strategy == 'MBPP_Persona_0.0_gpt-3.5-turbo-0125.jsonl':
                persona_rank += idx1

    pred_avg_rank = pred_rank / total
    zero_shot_avg_rank = zero_shot_rank / total
    zero_cot_avg_rank = zero_cot_rank / total
    few_shot_avg_rank = few_shot_rank / total
    few_cot_avg_rank = few_cot_rank / total
    self_avg_rank = self_rank / total
    reflection_avg_rank = reflection_rank / total
    self_plan_avg_rank = self_plan_rank / total
    progressive_hint_avg_rank = progressive_hint_rank / total
    persona_avg_rank = persona_rank / total

    return pred_avg_rank, zero_shot_avg_rank, zero_cot_avg_rank, few_shot_avg_rank, few_cot_avg_rank, self_avg_rank, reflection_avg_rank, self_plan_avg_rank, progressive_hint_avg_rank, persona_avg_rank


# truth_file_path = 'result/Final_result/HumanEval_gpt4o_nine_rank.jsonl'
test_embedding_model_path = f'result/{dataset}_contrastive{hard}{data_type}_model'
test_classification_model_path = f'result/classification_model/{dataset}_classification{hard}{data_type}_model_parameters.pth'
test_classification_model = ClassificationModel(input_size, num_classes)
test_classification_model.load_state_dict(torch.load(test_classification_model_path))
# data = list(map(json.loads, open(truth_file_path)))
test_data = list(map(json.loads, open(test_file_path)))
# data_list = generate_data_list(data, test_data)

print('\n----Actual ACC----\n')

pred_acc, zero_shot_acc, zero_cot_acc, few_shot_acc, few_cot_acc, self_acc, reflection_acc, self_plan_acc, progressive_hint_acc, persona_acc, zero_shot_correct = calculate_actual_acc(test_data, test_classification_model, test_embedding_model_path)
print(f'Pred accuracy: {pred_acc * 100:.2f}%; \nZeroshot accuracy: {zero_shot_acc * 100:.2f}%; \nZeroshot CoT accuracy: {zero_cot_acc * 100:.2f}%; \nFewshot accuracy: {few_shot_acc * 100:.2f}%; \nFewshot CoT accuracy: {few_cot_acc * 100:.2f}%; \nSelfDebug accuracy: {self_acc * 100:.2f}%; \nReflection accuracy: {reflection_acc * 100:.2f}%; \nSelfPlan accuracy: {self_plan_acc * 100:.2f}%; \nProgressive accuracy: {progressive_hint_acc * 100:.2f}%; \nPersona accuracy: {persona_acc * 100:.2f}%;')
print('zeroshot correct', zero_shot_correct)

print('\n----Token saved----\n')

pred_avg_token, zero_shot_avg_token, zero_cot_avg_token, few_shot_avg_token, few_cot_avg_token, self_avg_token, reflection_avg_token, self_plan_avg_token, progressive_hint_avg_token, persona_avg_token = calculate_token_saved(test_data, test_classification_model, test_embedding_model_path)
print(f'Pred avg token: {pred_avg_token}; \nZeroshot avg token: {zero_shot_avg_token}; \nZeroshot CoT avg token: {zero_cot_avg_token}; \nFewshot avg token: {few_shot_avg_token}; \nFewshot CoT avg token: {few_cot_avg_token}; \nSelfDebug avg token: {self_avg_token}; \nReflection avg token: {reflection_avg_token}; \nSelfPlan avg token: {self_plan_avg_token}; \nProgressive avg token: {progressive_hint_avg_token}; \nPersona avg token: {persona_avg_token}')

print('\n----Avg Rank----\n')

pred_avg_rank, zero_shot_avg_rank, zero_cot_avg_rank, few_shot_avg_rank, few_cot_avg_rank, self_avg_rank, reflection_avg_rank, self_plan_avg_rank, progressive_hint_avg_rank, persona_avg_rank = calculate_avg_rank(test_data, test_classification_model, test_embedding_model_path)
print(f'Pred avg rank: {pred_avg_rank}; \nZeroshot avg rank: {zero_shot_avg_rank}; \nZeroshot CoT avg rank: {zero_cot_avg_rank}; \nFewshot avg rank: {few_shot_avg_rank}; \nFewshot CoT avg rank: {few_cot_avg_rank}; \nSelfDebug avg rank: {self_avg_rank}; \nReflection avg rank: {reflection_avg_rank}; \nSelfPlan avg rank: {self_plan_avg_rank}; \nProgressive avg rank: {progressive_hint_avg_rank}; \nPersona avg rank: {persona_avg_rank}')
