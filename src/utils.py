import tiktoken
import json
import ast
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from radon.complexity import cc_visit
from radon.visitors import ComplexityVisitor
from radon.metrics import h_visit
from radon.metrics import mi_visit
from cognitive_complexity.api import get_cognitive_complexity


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613", is_input=True):
    input_pricing = 0.5/1000000
    output_pricing = 1.5/1000000
    """Return the number of tokens used by a list of messages."""
    try:
        encoding_name = tiktoken.encoding_for_model(model).name
        # encoding = tiktoken.encoding_for_model(model)
        encoding = tiktoken.get_encoding(encoding_name)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    if is_input:
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        cost = num_tokens * input_pricing
    else:
        num_tokens += len(encoding.encode(messages))
        cost = num_tokens * output_pricing

    return num_tokens, cost

def process_generation_to_code(gens):
    if '```python' in gens:
        gens = gens.split('```python')[1].split('```')[0]
    elif '```' in gens:
        gens = gens.split('```')[1].split('```')[0]
        
    return gens.split('\n')[1:-1]


def write_to_file(result, output_file):
    # print(output_file)
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + '\n')

def extract_one_assert(code):
    # Parse the code into an AST
    tree = ast.parse(code)

    # Define a function to find all assert statements in the code
    def find_asserts(node):
        results = []
        for n in ast.walk(node):
            if isinstance(n, ast.Assert):
                results.append(n)
        return results

    # Get all assert statements
    asserts = find_asserts(tree)

    # Print the first assert statement
    if asserts:
        first_assert = asserts[0]
        return ast.unparse(first_assert)
    else:
        print("No assert statements found.")

def extract_function_name_from_assert(test_cases):
    function_names = set()
    
    for test_case in test_cases:
        tree = ast.parse(test_case)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                test = node.test
                if isinstance(test, ast.Compare) and isinstance(test.left, ast.Call):
                    function_name = test.left.func.id
                    function_names.add(function_name)
    
    return list(function_names)[0]

def get_function_info(code):
    tree = ast.parse(code)
    
    function_name = None
    input_variables = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            input_variables = [arg.arg for arg in node.args.args]
            break 

    function_signature = f"def {function_name}({', '.join(input_variables)}):"
    
    return function_signature

def plot_multiclass_precision_recall(
    y_score, y_true_untransformed, class_list, classifier_name
):
    """
    Precision-Recall plotting for a multiclass problem. It plots average precision-recall, per class precision recall and reference f1 contours.

    Code slightly modified, but heavily based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    n_classes = len(class_list)
    y_true = pd.concat(
        [(y_true_untransformed == class_list[i]) for i in range(n_classes)], axis=1
    ).values

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true.ravel(), y_score.ravel()
    )
    average_precision_micro = average_precision_score(y_true, y_score, average="micro")
    print(
        str(classifier_name)
        + " - Average precision score over all classes: {0:0.2f}".format(
            average_precision_micro
        )
    )

    # setup plot details
    plt.figure(figsize=(9, 10))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")
    (l,) = plt.plot(recall_micro, precision_micro, color="gold", lw=2)
    lines.append(l)
    labels.append(
        "average Precision-recall (auprc = {0:0.2f})" "".format(average_precision_micro)
    )

    for i in range(n_classes):
        (l,) = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append(
            "Precision-recall for class `{0}` (auprc = {1:0.2f})"
            "".format(class_list[i], average_precision[i])
        )

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{classifier_name}: Precision-Recall curve for each class")
    plt.legend(lines, labels)
    # plt.show()

def count_physical_loc(code_string):
    # Split the input string into lines
    lines = code_string.split('\n')
    
    # Filter out empty lines and count the remaining lines
    non_empty_lines = [line for line in lines if line.strip() != '']
    
    return len(non_empty_lines)

def calculate_cyclomatic_complexity(code):
    # Analyze the code
    blocks = cc_visit(code)
    # for block in blocks:
    #     print(f'{block.name}: {block.complexity}')

    # Calculate the average Cyclomatic Complexity
    total_complexity = sum(block.complexity for block in blocks)
    average_complexity = total_complexity / len(blocks) if blocks else 0
    # print(f'Average Cyclomatic Complexity: {average_complexity}')
    return average_complexity

def calculate_halstead_complexity(code):
    results = h_visit(code)
    return results[0].vocabulary

def calculate_mi(code_string):
    mi_score = mi_visit(code_string, True)
    return 100-mi_score

def calculate_cognitive_complexity(code):
    parsed_code = ast.parse(code)
    try:
        new_body = [node for node in parsed_code.body if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr, ast.For, ast.AugAssign))]
        if not new_body:
            funcdef = ast.parse('')

        else:
            funcdef = new_body[0]
            
    except Exception as e:
        print(e)
        print('Using original code.')
        if not parsed_code.body:
            raise ValueError("The code provided is empty or invalid.")
        funcdef = parsed_code.body[0]
    
    cc_score = get_cognitive_complexity(funcdef)
    
    return cc_score

def extract_exec_code(code_str):
    """
    Extracts the value of the exec_context variable from the given code string.
    
    Parameters:
    code_str (str): A string containing the Python code.
    
    Returns:
    str: The value of the exec_context variable, or None if not found.
    """
    # Parse the code string into an abstract syntax tree (AST)
    tree = ast.parse(code_str)

    # Define a visitor class to extract the exec_context variable
    class ExecContextExtractor(ast.NodeVisitor):
        def __init__(self):
            self.exec_context = None

        def visit_Assign(self, node):
            # Check if the variable being assigned is exec_context
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'exec_context':
                # Extract the value of exec_context
                self.exec_context = ast.literal_eval(node.value)
    
    # Create an instance of the visitor class and visit the tree
    extractor = ExecContextExtractor()
    extractor.visit(tree)
    
    # Return the extracted exec_context value as a string
    if extractor.exec_context:
        return extractor.exec_context
    else:
        return None