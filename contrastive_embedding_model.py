import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from enum import Enum
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader, random_split

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

def create_triplet_dataset(anchor_sentences, positive_sentences, negative_sentences):
    """
    Create triplet dataset

    :param anchor_sentences: List of anchor sentences
    :param positive_sentences: List of positive sentences
    :param negative_sentences: List of negative sentences
    :return: DataLoader
    """
    if not (len(anchor_sentences) == len(positive_sentences) == len(negative_sentences)):
        raise ValueError("Anchor, positive, and negative sentences lists must have the same length")

    triplet_examples = []
    for anchor, positive, negative in zip(anchor_sentences, positive_sentences, negative_sentences):
        triplet_examples.append(InputExample(texts=[anchor, positive, negative]))

    return triplet_examples

dataset = 'code_complex'
hard = '_hard'
data_type = '_all'
file_path = f'result/{dataset}_dataset/{dataset}_contrastive{hard}_dataset_train{data_type}.jsonl'
test_file_path = f'result/{dataset}_dataset/{dataset}_contrastive{hard}_dataset_test{data_type}.jsonl'

anchor_sentences = []
positive_sentences = []
negative_sentences = []
test_anchor_sentences = []
test_positive_sentences = []
test_negative_sentences = []

# Read jsonl file and store the data
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        anchor_sentences.append(data['anchor'])
        positive_sentences.append(data['positive'])
        negative_sentences.append(data['negative'])

with open(test_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        test_anchor_sentences.append(data['anchor'])
        test_positive_sentences.append(data['positive'])
        test_negative_sentences.append(data['negative'])

# Create dataset
triplet_dataset = create_triplet_dataset(anchor_sentences, positive_sentences, negative_sentences)
test_examples = create_triplet_dataset(test_anchor_sentences, test_positive_sentences, test_negative_sentences)

class TripletDistanceMetric(Enum):
    """The metric for the triplet loss"""

    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

# Initialize model
# model_name = 'distilroberta-base'
model_name = 'microsoft/codebert-base'
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)


# Split dataset (80% training, 10% evaluation, 10% test)
train_size = int(0.8 * len(triplet_dataset))
# eval_size = int(0.1 * len(triplet_dataset))
# test_size = len(triplet_dataset) - train_size - eval_size
eval_size = len(triplet_dataset) - train_size

# train_examples, eval_examples, test_examples = random_split(triplet_dataset, [train_size, eval_size, test_size])
train_examples, eval_examples = random_split(triplet_dataset, [train_size, eval_size])

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
# train_loss = losses.TripletLoss(model, TripletDistanceMetric.COSINE)
train_loss = losses.TripletLoss(model)


# Create evaluator
evaluator = evaluation.TripletEvaluator.from_input_examples(eval_examples, name='eval')

num_epochs = 10
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

print(warmup_steps)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluator=evaluator
)


test_evaluator = evaluation.TripletEvaluator.from_input_examples(test_examples, name='test')
output = model.evaluate(test_evaluator)
print(output)

model.save(f'result/{dataset}_contrastive{hard}{data_type}_model')