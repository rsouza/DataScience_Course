# Databricks notebook source
# MAGIC %md
# MAGIC # Transforming the data

# COMMAND ----------

# importing essential packages
import pandas as pd
import math

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. `.map()` method
# MAGIC 
# MAGIC - `.map()` method is a great tool to have when it comes to processing & transforming iterables of numeric values
# MAGIC - it is a convenient way to perform _element-wise_ transformations and other data cleaning-related operations
# MAGIC - this method on Series takes a function object and a sequence of iterables (list, tuple, dictionary, set, or Series) as arguments
# MAGIC - any built-in functions that take an argument and returns a value can be used with `.map()`
# MAGIC - it returns an iterator (don't worry about this concept for now)
# MAGIC - the resulting values (an iterator) can be passed to the `list()` function or `set()` function to create a list or a set
# MAGIC 
# MAGIC Example code:
# MAGIC 
# MAGIC `map(function, iterable)`
# MAGIC 
# MAGIC To extract the result we can use for example: <break> 
# MAGIC 
# MAGIC `list(map(function, iterable))`
# MAGIC 
# MAGIC or 
# MAGIC 
# MAGIC `set(map(function, iterable))`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1.1 - Example using For Loops

# COMMAND ----------

# Run this code
our_list = ['This', 'is', 'the', 'first', 'example']

# COMMAND ----------

#Step 1: intialize an empty list "result_loop" that will store our results later
#Step 2: get the length of each variable in the list "our_list"
#Step 3: append the result to the list "result_loop"
#Step 4: print the result

result_loop = []

for word in our_list:
    result_loop.append(len(word))

print(result_loop)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1.2 - Example using .map() function

# COMMAND ----------

# Run this code
our_list = ['This', 'is', 'the', 'first', 'example']

# COMMAND ----------

# Step 1: Use the .map() method to get the length of the words in our_list
# Step 2: Pass the list() function to create a list of resulting values
# Step 3: Assign the result to the variable name "result" to print it to the screen

result_map = list(map(len, our_list))
print(result_map)

# COMMAND ----------

# MAGIC %md
# MAGIC In the above example the `.map()` method iterates over `our_list`, applies the function on each element and returns the length of the strings as a new list.

# COMMAND ----------

# MAGIC %md
# MAGIC Which one do you think is neater and shorter?
# MAGIC 
# MAGIC ```python
# MAGIC result_loop = []
# MAGIC 
# MAGIC for word in our_list:
# MAGIC   result_loops.append(len(word))
# MAGIC 
# MAGIC print(result_loop)
# MAGIC ```
# MAGIC vs. 
# MAGIC 
# MAGIC ```python
# MAGIC result_map = list(map(len, our_list))
# MAGIC print(result_map)
# MAGIC ```
# MAGIC 
# MAGIC In the programming world, it is cleaner and much more concise and sophisticated to use ``map()`` instead of for-loops. On top of that, with `map()` you can guarantee that the original sequence won't be acccidentally mutated or changed, since `map()` always returns a sequence of the results and leads to fewer errors in code. 
# MAGIC 
# MAGIC Feel free to check out [this](https://stackoverflow.com/questions/1975250/when-should-i-use-a-map-instead-of-a-for-loop#:~:text=4%20Answers&text=map%20is%20useful%20when%20you,loop%20and%20constructing%20a%20list.) on stackoverflow, where the advantages of using map over for-loops are discussed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1
# MAGIC Now, let's create a function `triple` and a Pandas Series `numbers` which will be our iterable.

# COMMAND ----------

# Run this code 
def triple(x):
    return x * 3

# COMMAND ----------

# Run this code
numbers = pd.Series([15, 4, 8, 45, 36, 7])

# COMMAND ----------

# TASK 1 >>>> Apply the .map() method with the function triple on our Pandas Series 'numbers' and store it in the variable result_2 
#             Print result_2 (the result should be the numbers multiply by 3)
#             Think about the three different steps performed in Example 1

### Start your code below ###

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. .filter() method
# MAGIC 
# MAGIC - similar to `.map()`, but instead of any function, `.filter()` takes a Boolean-valued function (a function that returns True or False based on the input data) instead of any built-in functions and a sequence of iterables (list, tuple, dictionary, set, or Series) as arugments
# MAGIC - returns the items of the intput data which the Boolean-valued function returns `True`
# MAGIC - the Boolean-valued function can be used-defined function

# COMMAND ----------

# MAGIC %md
# MAGIC Imagine there is a list with positive and negative numbers

# COMMAND ----------

# Run this code
list_mixed = [-1,0,2,24,-42,-5,30,99]

# COMMAND ----------

# Run this code
def criteria(x): 
    return x >= 0

# COMMAND ----------

# MAGIC %md
# MAGIC With the help of filter and our own user-defined function we can filter out the negative values and be left with only positive values.

# COMMAND ----------

list_positive = list(filter(criteria, list_mixed))
print(list_positive)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. `.apply()` method
# MAGIC 
# MAGIC - this method applies a function along an axis of the DataFrame \\(^{1}\\) 
# MAGIC - it also works elementwise but is suited to more complex functions and operations
# MAGIC - it accepts user-defined functions which apply a transformation/aggregation on a DataFrame (or Series) as well
# MAGIC 
# MAGIC You can find a nice comparison of `.map()` and `.apply()` methods and when to use them in [this article on stackoverflow](https://stackoverflow.com/questions/19798153/difference-between-map-applymap-and-apply-methods-in-pandas).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 3.1

# COMMAND ----------

# Run this code
students = [(1, 'Robert', 30, 'Slovakia', 26),
           (2, 'Jana', 29, 'Sweden' , 27),
           (3, 'Martin', 31, 'Sweden', 26),
           (4, 'Kristina', 26,'Germany' , 30),
           (5, 'Peter', 33, 'Austria' , 22),
           (6, 'Nikola', 25, 'USA', 23),
           (7, 'Renato', 35, 'Brazil', 26)]

students_1 = pd.DataFrame(students, columns= ['student_id', 'first_name', 'age', 'country', 'score'])
print(students_1)

# COMMAND ----------

# Run this code to create a regular function

def score_func(x): 
    if x < 25: 
        return "Retake" 
    else: 
        return "Pass"

# COMMAND ----------

# Use .apply() along with score_func that 
students_1['result'] = students_1.score.apply(score_func)
print(students_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 4.0
# MAGIC 
# MAGIC As we already know, regular functions are created using the `def` keyword. These type of functions can have any number of arguments and expressions.

# COMMAND ----------

# Example of regular function
def multi_add(x):
    return x * 2 + 5

# COMMAND ----------

result_1 = multi_add(5)
print(result_1)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Lambda Function
# MAGIC - an anonymous function (it means it can be defined without a name)
# MAGIC - the `def` keyword is not necessary with a lambda function
# MAGIC - lambda functions can have any number of parameters, but the function body can only **contain one expression** (that means multiple statements are not allowed in the body of a lambda function) = it is used for *_one-line expressions_*
# MAGIC - it returns a function object which can be assigned to variable
# MAGIC 
# MAGIC General syntax: `lambda x: x`
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Example 4.1

# COMMAND ----------

our_lambda = lambda x: x * 2 + 5
print(our_lambda(5))

# COMMAND ----------

# MAGIC %md
# MAGIC This simple lambda function takes an input `x` (in our case number 5), multiplies it by `2` and adds `5`. <br>
# MAGIC 
# MAGIC Lambda functions are commonly used along `.apply()` method and can be really useful. <br>
# MAGIC 
# MAGIC ### Example 4.2
# MAGIC 
# MAGIC Imagine that the scores of students above have not been correctly recorded and we need to multiply them by 10. 
# MAGIC 
# MAGIC Use a lambda function along with `apply()` and assign it to the specific column of the dataset ('score'). 

# COMMAND ----------

students_1.score = students_1.score.apply(lambda x: x * 10)
print(students_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2

# COMMAND ----------

# TASK 2 >>>> Use the .apply() method on column 'country' along with lambda to make words uppercase 
#             Do not forget assign it to this column

### Start your code below ###

# COMMAND ----------

# MAGIC %md
# MAGIC We can use lambda functions to simplify Example 3.1 like this:

# COMMAND ----------

# Run this code
students = [(1, 'Robert', 30, 'Slovakia', 26),
           (2, 'Jana', 29, 'Sweden' , 27),
           (3, 'Martin', 31, 'Sweden', 26),
           (4, 'Kristina', 26,'Germany' , 30),
           (5, 'Peter', 33, 'Austria' , 22),
           (6, 'Nikola', 25, 'USA', 23),
           (7, 'Renato', 35, 'Brazil', 26)]

students_1 = pd.DataFrame(students, columns= ['student_id', 'first_name', 'age', 'country', 'score'])

# COMMAND ----------

# A Lambda function is used instead of the custom defined function "score_func"

students_1['result'] = students_1.score.apply(lambda x: "Pass" if (x > 25) else "Retake")
print(students_1)

# COMMAND ----------

# MAGIC %md
# MAGIC Did you know we can combine the `.map()` and `.filter()` methods? Since `.filter()` returns a selected iterable based on certain criteria, the output of `.filter()` can be our input for the `.map()` method.
# MAGIC 
# MAGIC In order to avoid a negative number as an argument for `math.sqrt()` which will result in a `ValueError`, we want to filter out the negative numbers before we apply the `math.sqrt()` method.

# COMMAND ----------

# Run this code
list_mixed = [-1,0,2,24,-42,-5,30,99]

# COMMAND ----------

# Run this code
def criteria(x): 
    return x >= 0

# COMMAND ----------

list_sqrt = list(map(math.sqrt, filter(criteria, list_mixed)))
print(list_sqrt)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: Task 3

# COMMAND ----------

# TASK 3 >>>> With the help of .map() and .filter(),
#             round up any number that is bigger than 5 from the list "list_sqrt" to the next whole digit.
#             To round up the number, you can use round().
#             Don't forget to write your user-defined function as your criteria to filter out the "not desirable" numbers

### Start your code below ###

# COMMAND ----------

# MAGIC %md
# MAGIC # References
# MAGIC 
# MAGIC \\(^{1}\\) pandas. pandas.DataFrame.apply. [ONLINE] Available at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html#pandas-dataframe-apply. [Accessed 14 September 2020].
# MAGIC 
# MAGIC Stackoverflow. Difference between map, applymap and apply methods in Pandas. [ONLINE] Available at: https://stackoverflow.com/questions/19798153/difference-between-map-applymap-and-apply-methods-in-pandas. [Accessed 14 September 2020].
# MAGIC 
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. Source: https://github.com/zatkopatrik/authentic-data-science
