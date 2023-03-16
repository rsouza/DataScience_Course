# Databricks notebook source
# MAGIC %md
# MAGIC # 1. String Manipulation Methods (READ-AND-PLAY)

# COMMAND ----------

# Run this code
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image

# COMMAND ----------

# Run this code
string = '    pen,pineapple,apple, pen   '
print(string)

# COMMAND ----------

# MAGIC %md
# MAGIC `split()` method 
# MAGIC 
# MAGIC - This method splits a string into a list where each word is a list item.
# MAGIC - We need to specify the `separator` to use when splitting the string.
# MAGIC - We can specifiy how many splits to do by setting the `maxsplit` parameter.

# COMMAND ----------

# Split the string using separator ','
string.split(',')

# COMMAND ----------

# Run this code
string_2 = 'summer#autumn   #spring  # winter'
print(string_2)

# COMMAND ----------

# Split the string using separator '#'
string_2.split('#')

# COMMAND ----------

# Split string_2 and set the maxsplit parameter to 2 (this should return a list with 3 elements)
x = string_2.split('#', 2)
print(x)

# COMMAND ----------

# MAGIC %md
# MAGIC `strip()` method
# MAGIC 
# MAGIC - It removes whitespaces at the beginning and at the end of the string.

# COMMAND ----------

# Remove whitespaces in the variable our_string
our_string = '     There is a lot of space at the beginning and at the end of this sentence, let`s remove it.       '
our_result = our_string.strip()
print(our_result)

# COMMAND ----------

# MAGIC %md
# MAGIC `join()` method
# MAGIC 
# MAGIC - This method takes all items in an iterable and joins them into one string.

# COMMAND ----------

# Run this code
my_list = ['Please', 'join', 'these', 'items.']
'_'.join(my_list)

# COMMAND ----------

# Run this code
my_tuple = ('We','are', 'joining', 'again.')
'-'.join(my_tuple)

# COMMAND ----------

# MAGIC %md
# MAGIC In the case of a dictionary, `join()` tries to join keys of the dictionary, not values.

# COMMAND ----------

# Run this code
my_dictionary = {'Key_1':'1',
                 'Key_2':'2'}
'#'.join(my_dictionary)

# COMMAND ----------

# MAGIC %md
# MAGIC `index()` method
# MAGIC 
# MAGIC - It returns the position of the first character in a substring if the substring is found in the string.
# MAGIC - It raises a `ValueError` if nothing is found.

# COMMAND ----------

# Run this code
string_3 = 'That is my string'

# COMMAND ----------

# Find the position of 'm' using `index()`
string_3.index('m')

# COMMAND ----------

# MAGIC %md
# MAGIC `replace()` method
# MAGIC 
# MAGIC - This method replaces occurences of a substring with another string.
# MAGIC - It is commonly used to remove characters by passing an empty string.

# COMMAND ----------

# Replacing string in string_3
string_3.replace('is','was')

# COMMAND ----------

# Run this code
string_4 = 'Why is here a semicolon; ?'

# COMMAND ----------

# Replacing character
string_4.replace(';','')

# COMMAND ----------

# Run this code
string_5 = 'Banana, avocado, pineapple, artichoke'

# COMMAND ----------

# TASK 1 >>>> Use .replace() method to replace 'a' with 'A' in string_5 and store it in variable result_1

result_1 = string_5.replace('a', 'A')
print(result_1)

# COMMAND ----------

# MAGIC %md
# MAGIC `upper()` method
# MAGIC 
# MAGIC - This method converts all lowercase characters in a string into uppercase characters and returns it.
# MAGIC 
# MAGIC `lower()` method
# MAGIC - This method converts all upercase characters in a string into lowercase characters and returns it.

# COMMAND ----------

# Run this code
string_to_upper = "Make this uppercase"
print(string_to_upper.upper())

# COMMAND ----------

# Run this code
string_to_lower = 'THIS SHOULD BE ALL LOWERCASE'
print(string_to_lower.lower())

# COMMAND ----------

# MAGIC %md
# MAGIC ``find()`` method
# MAGIC - This method is similar to `index()`.
# MAGIC - If the substring is found, this method returns the index of first occurrence of the substring.
# MAGIC - If the substring is not found, -1 is returned.
# MAGIC - This function is **case sensitive**.

# COMMAND ----------

# Run this code
quote = "Data Science is cool"

print("The quote is: " + quote)

# first occurance of 'Data Science'
result = quote.find('Data Science')
print("Substring 'Data Science':", result)

# what happens when we neglect the case sensitivity 
result = quote.find('data science')
print("Substring 'data science':", result)

# COMMAND ----------

# find returns -1 if substring not found
result = quote.find('RBI')
print("Substring 'RBI':", result)

# How to use find()
if (quote.find('is') != -1):
    print("Substring is found")
else:
    print("Substring is not found")

# COMMAND ----------

# MAGIC %md
# MAGIC If you go to the [Documentation of `find()`](https://python-reference.readthedocs.io/en/latest/docs/str/find.html), you will see that ``find()`` can accept three parameters. One is compulsory, and the others are optional. 
# MAGIC 
# MAGIC The general syntax looks like this:
# MAGIC ````
# MAGIC string.find(value, start, end)
# MAGIC ````
# MAGIC 
# MAGIC |Parameter|Characteristics|Description|Default|
# MAGIC |---------|-----|------------- |-----|
# MAGIC |sub| Required|The string that you are searching for| (no default)|
# MAGIC |start|Optional|Specify the start position|Default is 0, corresponds to beginning of the string|
# MAGIC |end|Optional|Specify the end position|Default is the end of the string|

# COMMAND ----------

# Run this code
quote = "Data Science is so cool, I love Data Science!"

print("The new quote is:" + quote)

# Where in the text is the first occurrence of the substring "Data" when you only want to search between position 10 and 40?
result = quote.find("Data",10,40)

print("Substring 'Data' from position 10 to 40: ", result)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Project: Cleaning Column Names

# COMMAND ----------

# Import Pandas library
import pandas as pd
data = pd.read_csv('../Data/avocado.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC If we take a look at the column names, we notice that they need some cleaning, such as removing the whitespaces. Some systems and data pipelines can have issues with these.

# COMMAND ----------

# Run this code
data_2015 = data[data['year'] == 2015]
data_2015.columns

# COMMAND ----------

# MAGIC %md
# MAGIC Let's use a lambda function and three of the methods which we have just learned - strip, lower and replace.

# COMMAND ----------

# Run this code
data_2015.rename(columns = lambda x: x.strip().lower().replace(' ','_'), inplace = True)

# COMMAND ----------

# Run this code
data_2015.head()

# COMMAND ----------

# MAGIC %md
# MAGIC One column is still ugly. It would not be worth it to attempt and write a specific function for it. We address it manually via a dictionary.

# COMMAND ----------

# BONUS TASK - Hints: use .rename() method and specify columns through dictionary, i.e. 'column_name_to_clean':'new_column_name'
#                   specify inplace = True

data_2015.rename(columns={'averageprice':'average_price'}, inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Cleaning Text Column (READ-ONLY)
# MAGIC 
# MAGIC Imagine we have 2 possible categories of avocado (A and B) in the same row for the same day separated with '/'. 
# MAGIC It would be an issue for us if we'd like to explore and visualize data based on the avocado's category. 
# MAGIC 
# MAGIC We can use `str.split()` method to resolve this issue in few steps.

# COMMAND ----------

# Run this code - don't bother what it does for now

data_avo = {'day':'Monday', 'category':'A/B', 'type':'organic'}
monday_data = pd.DataFrame(data_avo, range(10))           

# COMMAND ----------

# MAGIC %md
# MAGIC Let's now examine the special altered dataset which we created. You will notice that in the 'category' column. we have A and B symbols. These represent avocado types, which means that in **every row we have stored 2 observations**. That is not good and we need to split each row into 2 separate rows.

# COMMAND ----------

# Run this code
monday_data

# COMMAND ----------

# MAGIC %md
# MAGIC At first, we use `split` the method to create a list of two objects from the original element in the column.

# COMMAND ----------

# Firstly, split the 'category' column with separator '/'

monday_data['category'] = monday_data['category'].str.split('/')
monday_data

# COMMAND ----------

# MAGIC %md
# MAGIC As the next steps:
# MAGIC 
# MAGIC - next we use `apply()` function on `monday_data` that return Series: use lambda function `lambda x:` to create new Series - we also need to specify `axis = 1` which returns a new column for avocado's type
# MAGIC - after the `apply()` part add `stack()` - to stack avocado's category 

# COMMAND ----------

# Run this code

series_2 = monday_data.apply(lambda x: pd.Series(x['category']), axis = 1).stack()

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see below, **categories are now separated into new rows**: 10 observation for Monday. However there is also new level (another index) for A and B that we don't need anymore. 

# COMMAND ----------

# Run this code
series_2

# COMMAND ----------

# MAGIC %md
# MAGIC We can remove this index using `reset_index()`: 
# MAGIC - use `drop = True`
# MAGIC - set `level = 1`

# COMMAND ----------

# Run this code
series_2 = monday_data.apply(lambda x: pd.Series(x['category']), axis = 1).stack().reset_index(level = 1, drop = True)

# COMMAND ----------

# MAGIC %md
# MAGIC - give the Series (it will be a new column) a name 'avocado_category'

# COMMAND ----------

# Run this code
series_2.name = 'avocado_category' 

# COMMAND ----------

# MAGIC %md
# MAGIC - drop the column 'category' from `new_data` (this is the column that contain A/B), set axis = 1
# MAGIC - join `series_2` where we have separated categories

# COMMAND ----------

# Run this code
new_data = monday_data.drop('category', axis = 1).join(series_2)

# COMMAND ----------

# Run this code
new_data

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Project: Cleaning Text Column

# COMMAND ----------

# Run the code
import numpy as np
data_1 = pd.read_csv('Data/movie_metadata.csv')
movie_data = data_1.iloc[:,np.r_[1:3, 8:13]]

# COMMAND ----------

# Display first 5 rows of movie_data and look at the genres column
movie_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we use the same way to split genres of movies, the only difference is the separator '|'.

# COMMAND ----------

# Split the 'genres' column with separator '/'

movie_data.genres = movie_data.genres.str.split('|')
movie_data.head()

# COMMAND ----------

# Create a new Series for genres using a lambda function and apply it to movie_data

series_genres = movie_data.apply(lambda x: pd.Series(x['genres']), axis = 1).stack().reset_index(level = 1,drop = True)

# COMMAND ----------

# Print the new Series

print(series_genres)

# COMMAND ----------

# Give the Series (new column) the name 'genre'
series_genres.name = 'genre'

# COMMAND ----------

# TASK 2 >>>> Drop the old column 'genres' from movie_data on axis = 1
#             Join to the new Series 'series_genres'. 
#             Assign it to our_movie_data.

our_movie_data = movie_data.drop('genres', axis = 1).join(series_genres)

# COMMAND ----------

# Run this code

print(our_movie_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Regular expressions
# MAGIC 
# MAGIC - provide a flexible way to serach or match string patterns in text
# MAGIC - a single expression, commonly called a **regex**, is a string formed according to the regular expression language
# MAGIC - using built-in module `re` we can apply regular expressions to strings
# MAGIC 
# MAGIC Run the following cell showing example of regular expression for validating an email \\(^{1}\\). 

# COMMAND ----------

# Run this code
Image('Images/regex.PNG')

# COMMAND ----------

# Import re module
import re

# COMMAND ----------

# MAGIC %md
# MAGIC Regex Methods
# MAGIC 
# MAGIC There is a set of methods that allows us to search a string for a match such as:
# MAGIC 
# MAGIC `findall`
# MAGIC - returns a list that contain all matches
# MAGIC 
# MAGIC `match`
# MAGIC - if zero or more characters at the beginning of string match this regular expression, return a corresponding match object
# MAGIC 
# MAGIC `search`
# MAGIC - scan through string looking for the first location where regular expression produces a match and return a corresponding match object
# MAGIC 
# MAGIC `split`
# MAGIC - breaks string into pieces at each occurence of pattern

# COMMAND ----------

# Split string called 'sentence' by whitespaces 
sentence = 'This  sentence contains     whitespace'

# COMMAND ----------

# MAGIC %md
# MAGIC To split this string we need to call `re.split()`. 
# MAGIC 
# MAGIC Within this method we specify regex `'\s+'` describing one or more whitespace character and string to split (in our case 'sentence').
# MAGIC 
# MAGIC Firstly, the regex is compiled and then the `split` function is called on the passed string.

# COMMAND ----------

# Run this code
re.split('\s+', sentence)

# COMMAND ----------

# MAGIC %md
# MAGIC With `re.compile()` we can combine a regular expression pattern into pattern objects which can be used for pattern matching
# MAGIC - this approach is recommended if you intend to apply the same expression to many strings 

# COMMAND ----------

# Run this code
our_regex = re.compile('\s+')

# COMMAND ----------

# Split string 'sentence' using regex object 'our_regex'
our_regex.split(sentence)

# COMMAND ----------

# Get the list of all patterns that match regex using findall() method
our_regex.findall(sentence)

# COMMAND ----------

# Create regex object that match pattern contain 'e'
another_regex = re.compile('e')

# COMMAND ----------

# Run the code
sentence_2 = 'Learning RegEx is fun'

# COMMAND ----------

# Return the list that contain all matches in string 'sentence_2'
another_regex.findall(sentence_2)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, the regex object performed case-sensitive matching and matched lowercase letters only. 
# MAGIC 
# MAGIC We can also define a case insensitive regex object during the pattern compile using `flags = re.IGNORECASE`

# COMMAND ----------

# Create regex object that is not case sensitive using re.IGNORECASE
regex_sensitive = re.compile('e', flags = re.IGNORECASE)

# COMMAND ----------

# Run this code
regex_sensitive.findall(sentence_2)

# COMMAND ----------

text = 'Regex, Regex pattern, Expressions'

# Create a regex object with the matche pattern 's'
pattern = re.compile('s')

# COMMAND ----------

# Check for a match anywhere in the string using .search()

pattern.search(text)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see `search` returns only the start and end position of the pattern.

# COMMAND ----------

# Check for a match only at the beginning of the string using .match()

pattern.match(text)

# COMMAND ----------

# Run this line of code

email = 'Email addresses of our two new employees are first.example@gmail.com and second_example@gmail.com'

# COMMAND ----------

# Write a regex to match email addresses

email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'

# COMMAND ----------

# Create a regex object that matches email addresses and make it case-insensitive

rege = re.compile(email_pattern, flags = re.IGNORECASE)

# COMMAND ----------

# Get list of email addresses from 'email' string

rege.findall(email)

# COMMAND ----------

# Search for the position of the first email address in the string 'email'

rege.search(email)

# COMMAND ----------

text = 'The average price of the avocados was $1.35 last year, hopefully, this year the price don't exceed $1.50 for a piece!'

# COMMAND ----------

# TASK 3 >>>> Google for Regex patterns to match decimal numbers and assign it to the variable decimal_number

decimal_number = "[0-9]*[.][0-9]*"

# COMMAND ----------

# Regex object that match decimal number - won't work if TASK 3 is not completed

pattern_dec = re.compile(decimal_number)

# COMMAND ----------

# Run this code - won't work if TASK 3 is not completed

pattern_dec.findall(text)

# COMMAND ----------

# MAGIC %md
# MAGIC You can find many Regular Expressions Cheat Sheets on the web, like [this one](https://cheatography.com/mutanclan/cheat-sheets/python-regular-expression-regex/).

# COMMAND ----------

# MAGIC %md
# MAGIC **Hint**
# MAGIC 
# MAGIC If we want to find some pattern (decimal numbers for example) within the string of a Series, we can also use the pandas function `str.contains`. For more information check the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html).

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix
# MAGIC 
# MAGIC Data Source 1: https://www.kaggle.com/neuromusic/avocado-prices
# MAGIC 
# MAGIC License: Database: Open Database, Contents: © Original Authors
# MAGIC 
# MAGIC 
# MAGIC Data source 2: https://www.kaggle.com/orgesleka/imdbmovies
# MAGIC 
# MAGIC License: CC0: Public Domain
# MAGIC 
# MAGIC # References
# MAGIC 
# MAGIC \\(^{1}\\) BreatheCode. 2017. Regex Tutorial. [ONLINE] Available at: https://content.breatheco.de/en/lesson/regex-tutorial-regular-expression-examples. [Accessed 14 September 2020].
# MAGIC 
# MAGIC pandas. pandas.Series.str.contains. [ONLINE] Available at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html. [Accessed 14 September 2020].
# MAGIC 
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. Source: https://github.com/zatkopatrik/authentic-data-science
