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

HumanEval_Fewshot_CoT_prompt = '''
Here are some examples of how to generate the code step by step.

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
    Let's complete the following code step by step.
    """
    # Step 1: Create a variable to store the result
    result = False
    # Step 2: Loop through the list of numbers
    for i in range(len(numbers)):
        # Step 3: Check if the current number is within the threshold of any other number in the list
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) <= threshold:
                # Step 4: If the condition is met, set the result to True and break out of the loop
                result = True
                break
        # Step 5: If the result is already True, break out of the loop
        if result:
            break

    # Step 6: Return the result
    return result
```

Example 2:

```python
from typing import List
def rescale_to_unit(numbers: List[float]) -> List[float]:
    """ Given list of numbers (of at least two elements), apply a linear transform to that list,
    such that the smallest number will become 0 and the largest will become 1
    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
    [0.0, 0.25, 0.5, 0.75, 1.0]
    Let's complete the following code step by step.
    """
    # Step 1: Find the smallest and largest numbers in the list
    smallest = min(numbers)
    largest = max(numbers)
    # Step 2: Calculate the difference between the largest and smallest numbers
    difference = largest - smallest
    # Step 3: Create a new list to store the rescaled numbers
    rescaled_numbers = []
    # Step 4: Loop through each number in the original list
    for number in numbers:
        # Step 5: Apply the linear transform to each number
        rescaled_number = (number - smallest) / difference
        # Step 6: Add the rescaled number to the new list
        rescaled_numbers.append(rescaled_number)
    # Step 7: Return the new list
    return rescaled_numbers
```

Example 3:

```python
def strlen(string: str) -> int:
    """ Return length of given string
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    Let's complete the following code step by step.
    """
    # 1. Initialize a variable to store the length of the string
    length = 0
    # 2. Use a for loop to iterate through each character in the string
    for char in string:
        # 3. Increment the length variable by 1 for each character
        length += 1
    # 4. Return the length variable
    return length
```

How about this function?
{prompt}
'''

MBPP_system_message = 'Only generate the code.'

MBPP_Fewshot_CoT_prompt = '''
You are an expert Python programmer, and you should write code step by step to complete the task.

Example 1:
Here is your task: Write a function to find the similar elements from the given two tuple lists.
Your code should pass these tests: ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]

```python
def similar_elements(test_tup1, test_tup2):
    # Convert both tuples to sets to remove duplicates and allow for set operations
    res = tuple(set(test_tup1) & set(test_tup2))
    # The '&' operator finds the intersection of the two sets, i.e., elements common to both sets
    return (res)  # Return the common elements as a tuple
```

Example 2:
Here is your task: Write a python function to identify non-prime numbers.
Your code should pass these tests: ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"]

```python
import math  # Import the math module to use mathematical functions

def is_not_prime(n):
    result = False  # Initialize a variable 'result' to False. It indicates whether 'n' is not a prime number.

    # Loop from 2 to the square root of 'n', rounded down to the nearest whole number, then add 1 to include that number in the loop.
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:  # Check if 'n' is divisible by 'i' (i.e., no remainder)
            result = True  # If 'n' is divisible by any number other than 1 and itself, set 'result' to True.

    return result  # Return the value of 'result'. True if 'n' is not a prime number, False if 'n' is a prime number.
```

Example 3:
Here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
Your code should pass these tests: ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"]

```python
import heapq as hq  # Import the heapq module and rename it as hq for convenience

def heap_queue_largest(nums, n):
    # Use the heapq.nlargest function to get the 'n' largest elements from the list 'nums'
    largest_nums = hq.nlargest(n, nums)
    return largest_nums  # Return the list of the 'n' largest elements
```

How about this task?
Here is your task: {prompt}
The function name and input variables should follow this template: {function_name}.
'''

DS1000_system_message = 'Only generate the code.'

DS1000_Fewshot_CoT_prompt = '''
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
I would like to shuffle the order of the DataFrame's rows according to a list. \
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
# Define a function 'g' that takes a DataFrame 'df' and a list 'List'
def g(df, List):
    # Return a new DataFrame containing rows indexed by 'List'
    return df.iloc[List]

# Example usage:
# Assuming 'df' is a pandas DataFrame and 'List' is a list of indices or labels
# Make sure to pass a copy of 'df' to avoid modifying the original DataFrame outside of this function
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
# Define a function 'g' that takes a DataFrame 'df' as input
def g(df):
    # Use the 'apply' method to apply a function to each column (Series) in the DataFrame 'df'
    # The lambda function 'lambda x: x.map(x.value_counts())' maps each element in the Series 'x'
    # to its count within that Series
    # This effectively replaces each element with its count within its column
    # The result is a DataFrame of boolean values indicating whether each element's count is >= 2
    # Where the condition is not met (count < 2), replace the value with "other"
    return df.where(df.apply(lambda x: x.map(x.value_counts())) >= 2, "other")

# Example usage:
# Assuming 'df' is a pandas DataFrame
# Make sure to pass a copy of 'df' to avoid modifying the original DataFrame outside of this function
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
# Define a function 'g' that takes a DataFrame 'df' as input
def g(df):
    # Select rows from 'df' where either:
    # - 'keep_if_dup' column equals 'Yes'
    # - 'url' column values are not duplicated (retain the first occurrence)
    return df.loc[(df['keep_if_dup'] == 'Yes') | ~df['url'].duplicated()]

# Example usage:
# Assuming 'df' is a pandas DataFrame
# Make sure to pass a copy of 'df' to avoid modifying the original DataFrame outside of this function
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
            message =[{'role': 'system', 'content': HumanEval_system_message}, {'role': 'user', 'content': HumanEval_Fewshot_CoT_prompt.format(prompt=prompt)}]
        elif args.dataset == 'MBPP':
            function_name = get_function_info(per_data['code'])
            prompt = per_data['text']
            message =[{'role': 'system', 'content': MBPP_system_message}, {'role': 'user', 'content': MBPP_Fewshot_CoT_prompt.format(prompt=prompt, function_name=function_name)}]
        elif args.dataset == 'DS1000':
            prompt = per_data['prompt']
            message =[{'role': 'system', 'content': DS1000_system_message}, {'role': 'user', 'content': DS1000_Fewshot_CoT_prompt+f'\n\nHow about this problem?\n{prompt}'}]

        messages.append(message)

    return messages, data_selected

def Fewshot_CoT_generate_result(args):
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

