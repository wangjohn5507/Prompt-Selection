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

HumanEval_few_shot_prompt = '''
There are some examples of how to generate the code.

Example 1:

```python
from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
```

Example 2:

```python
from typing import List
def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result
```

Example 3:

```python
def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    return number % 1.0
```

How about this function?
{prompt}
'''


MBPP_system_message = 'Only generate the code.'

MBPP_few_shot_prompt = '''
Here are some examples of how to generate the code.

Example 1:
Here is your task: Write a function to find the similar elements from the given two tuple lists.
Your code should pass these tests: ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]

```python
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return (res) 
```

Example 2:
Here is your task: Write a python function to identify non-prime numbers.
Your code should pass these tests: ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"]

```python
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
```

Example 3:
Here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
Your code should pass these tests: ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"]

```python
import heapq as hq
def heap_queue_largest(nums,n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```

How about this task?
Here is your task: {prompt}
The function name and input variables should follow this template: {function_name}.
'''

DS1000_system_message = 'Only generate the code.'

DS1000_few_shot_prompt = '''
You are an expert Python programmer, and you should write code to solve the problem.

Example 1:
Problem:
I have the following DataFrame:
    Col1  Col2  Col3  Type
0      1     2     3     1
1      4     5     6     1
2      7     8     9     2
3    10    11    12     2
4    13    14    15     3
5    16    17    18     3

The DataFrame is read from a CSV file. All rows which have Type 1 are on top, followed by the rows with Type 2, followed by the rows with Type 3, etc.
I would like to shuffle the order of the DataFrame's rows according to a list. 
For example, give a list [2, 4, 0, 3, 1, 5] and desired result should be:
    Col1  Col2  Col3  Type
2      7     8     9     2
4     13    14    15     3
0     1     2     3     1
3    10    11    12     2
1     4     5     6     1
5    16    17    18     3
...

How can I achieve this?

A:
<code>
import pandas as pd
import numpy as np

df = pd.DataFrame({'Col1': [1, 4, 7, 10, 13, 16],
                   'Col2': [2, 5, 8, 11, 14, 17],
                   'Col3': [3, 6, 9, 12, 15, 18],
                   'Type': [1, 1, 2, 2, 3, 3]})
List = np.random.permutation(len(df))
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

The answer will be:
```python
def g(df, List):
    return df.iloc[List]

result = g(df.copy(), List)
```

Example 2:
Problem:
I have following pandas dataframe :

import pandas as pd
from pandas import Series, DataFrame
data = DataFrame({'Qu1': ['apple', 'potato', 'cheese', 'banana', 'cheese', 'banana', 'cheese', 'potato', 'egg'],
              'Qu2': ['sausage', 'banana', 'apple', 'apple', 'apple', 'sausage', 'banana', 'banana', 'banana'],
              'Qu3': ['apple', 'potato', 'sausage', 'cheese', 'cheese', 'potato', 'cheese', 'potato', 'egg']})


I'd like to change values in columns Qu1,Qu2,Qu3 according to value_counts() when value count great or equal 2
For example for Qu1 column
>>> pd.value_counts(data.Qu1) >= 2
cheese     True
potato     True
banana     True
apple     False
egg       False

I'd like to keep values cheese,potato,banana, because each value has at least two appearances.
From values apple and egg I'd like to create value others
For column Qu2 no changes :
>>> pd.value_counts(data.Qu2) >= 2
banana     True
apple      True
sausage    True


The final result as in attached test_data
test_data = DataFrame({'Qu1': ['other', 'potato', 'cheese', 'banana', 'cheese', 'banana', 'cheese', 'potato', 'other'],
                  'Qu2': ['sausage', 'banana', 'apple', 'apple', 'apple', 'sausage', 'banana', 'banana', 'banana'],
                  'Qu3': ['other', 'potato', 'other', 'cheese', 'cheese', 'potato', 'cheese', 'potato', 'other']})

Thanks !


A:
<code>
import pandas as pd

df = pd.DataFrame({'Qu1': ['apple', 'potato', 'cheese', 'banana', 'cheese', 'banana', 'cheese', 'potato', 'egg'],
                   'Qu2': ['sausage', 'banana', 'apple', 'apple', 'apple', 'sausage', 'banana', 'banana', 'banana'],
                   'Qu3': ['apple', 'potato', 'sausage', 'cheese', 'cheese', 'potato', 'cheese', 'potato', 'egg']})
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

The answer will be:
```python
def g(df):
    return df.where(df.apply(lambda x: x.map(x.value_counts())) >= 2, "other")

result = g(df.copy())
```


Example 3:
Problem:
I have a dataset :
id    url     keep_if_dup
1     A.com   Yes
2     A.com   Yes
3     B.com   No
4     B.com   No
5     C.com   No

I want to remove duplicates, i.e. keep first occurence of "url" field, BUT  keep duplicates if the field "keep_if_dup" is YES.
Expected output :
id    url     keep_if_dup
1     A.com   Yes
2     A.com   Yes
3     B.com   No
5     C.com   No

What I tried :
Dataframe=Dataframe.drop_duplicates(subset='url', keep='first')

which of course does not take into account "keep_if_dup" field. Output is :
id    url     keep_if_dup
1     A.com   Yes
3     B.com   No
5     C.com   No

A:
<code>
import pandas as pd

df = pd.DataFrame({'url': ['A.com', 'A.com', 'A.com', 'B.com', 'B.com', 'C.com', 'B.com'],
                   'keep_if_dup': ['Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes']})
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

The answer will be:
```python
def g(df):
    return df.loc[(df['keep_if_dup'] =='Yes') | ~df['url'].duplicated()]

result = g(df.copy())
```
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
            message =[{'role': 'system', 'content': HumanEval_system_message}, {'role': 'user', 'content': HumanEval_few_shot_prompt.format(prompt=prompt)}]
        elif args.dataset == 'MBPP':
            function_name = get_function_info(per_data['code'])
            prompt = per_data['text']
            message =[{'role': 'system', 'content': MBPP_system_message}, {'role': 'user', 'content': MBPP_few_shot_prompt.format(prompt=prompt, function_name=function_name)}]
        elif args.dataset == 'DS1000':
            prompt = per_data['prompt']
            message =[{'role': 'system', 'content': DS1000_system_message}, {'role': 'user', 'content': DS1000_few_shot_prompt+f'\n\nHow about this problem?\n{prompt}'}]
        messages.append(message)

    return messages, data_selected

def Fewshot_generate_result(args):
    output_path = f'result/{args.dataset}_gpt4o_result/{args.dataset}_{args.strategy}_{args.temperature}_{args.model}.jsonl'
    messages, data = generate_prompt(args)
    for idx, per_data in enumerate(tqdm.tqdm(data)):
        result = copy.copy(per_data)
        response = call_chat_gpt(messages[idx], args)

        input_token, input_cost = num_tokens_from_messages(messages[idx], args.model, True)
        output_token, output_cost = num_tokens_from_messages(response, args.model, False)

        code = process_generation_to_code(response)
        
        result['response_code'] = '\n'.join(code)
        result['input_token'] = input_token
        result['input_cost'] = input_cost
        result['output_token'] = output_token
        result['output_cost'] = output_cost

        write_to_file(result, output_path)

