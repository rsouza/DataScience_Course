# Databricks notebook source
# MAGIC %md
# MAGIC # Univariate Analysis with Seaborn
# MAGIC 
# MAGIC You are presumably wondering: "Yeah, Matplotlib is great, it allows me to customize everything I want. But is there any other visualization library which is able to conjure up a good-looking graph with less code? 
# MAGIC 
# MAGIC In fact, there are many visualization libraries for Python. We will explore **Seaborn**, which is built on top of the Matplotlib.
# MAGIC 
# MAGIC The strength of Seaborn is the ability to create attractive, aesthetically pleasing plots integrating **Pandas DataFrame**s functionalities. So far, in order to create plots we always needed to 'extract' a Series of the DataFrame and then we were able to apply some plotting function. Seaborn, on the other hand, operates on the whole dataset, intelligently using labels of the `DataFrame` and internally performing the necessary steps. Seaborn makes creating visualizations very easy and intuitive by using high-level functions. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### [Figure-level vs. axes-level functions](https://seaborn.pydata.org/tutorial/function_overview.html)  
# MAGIC 
# MAGIC There is a cross-cutting classification of seaborn functions as “axes-level” or “figure-level”. 
# MAGIC + **Axes-level** functions plot data onto a single matplotlib.pyplot.Axes object, which is the return value of the function.
# MAGIC + **Figure-level** functions interface with matplotlib through a seaborn object, usually a FacetGrid, that manages the figure. Each module has a single figure-level function, which offers a unitary interface to its various axes-level functions.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Importing Seaborn library and loading the data
# MAGIC 
# MAGIC Firstly, we import the Seaborn library and give it conventional alias `sns`. The abbreviation is derived from Samuel Norman "Sam" Seaborn, a fictional character portrayed by Rob Lowe in the television serial drama _The West Wing_. 

# COMMAND ----------

# Importing Seaborn library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC There are 18 example datasets provided by Seaborn. After completing this notebook, you can choose a few of them that seem interesting to you and and try to apply your gained knowledge about visualization using Seaborn. 
# MAGIC 
# MAGIC To get a list of available datasets you can use `get_dataset_names()` function.

# COMMAND ----------

# Print available example datasets
# This is blocked by security.
# sns.get_dataset_names()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Setting the theme
# MAGIC 
# MAGIC Firstly, we discuss how you can control the **aesthetic of a figure** in other words, choose the **theme** based on your needs and preferences. It always depends on whether you are exploring the data for yourself or you want to communicate your insights to an audience. During your exploratory part, your visualizations do not need to be perfect and polished as long as they serve the purpose of revealing necessary and useful insight. 
# MAGIC 
# MAGIC But if your visualization will be presented to others, it is appropriate to take care of the plot's appearance in order to make it appealing and catching the attention. This is true also in the case of a theme. 
# MAGIC 
# MAGIC There are 5 predefined themes ('darkgrid' is by default):
# MAGIC 
# MAGIC - darkgrid
# MAGIC - whitegrid
# MAGIC - dark
# MAGIC - white
# MAGIC - ticks
# MAGIC 
# MAGIC To set a specific theme use `set_style('ticks')` with the chosen theme as the argument. Take a look at the documentation [here](https://seaborn.pydata.org/tutorial/aesthetics.html) for more information.

# COMMAND ----------

# Setting style
sns.set_style('dark')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Loading dataset
# MAGIC 
# MAGIC We will go with the 'penguins' dataset that can be loaded using `load_dataset()` function which returns a Pandas DataFrame.
# MAGIC 
# MAGIC This dataset consists of 7 attributes and 344 observations about penguins from islands in the Palmer Archipelago in Antarctica.
# MAGIC 
# MAGIC **Attributes explanation**
# MAGIC - species: species of a penguin (Adelie, Gentoo and Chinstrap)
# MAGIC - island: the name of an island (Biscoe, Dream, Torgersen)
# MAGIC - bill_length_mm: the length of the bill (in mm)
# MAGIC - bill_depth_mm: the depth of the bill (in mm)
# MAGIC - flipper_length_mm: the length of the flipper (in mm)
# MAGIC - body_mass_g: body mass (in grams)
# MAGIC - sex: the sex of a penguin

# COMMAND ----------

# Load the data
penguins = pd.read_csv('../Data/penguins.csv')

# COMMAND ----------

# Take a look at the first 5 rows
penguins.head()

# COMMAND ----------

# Explore statistics information about the data
penguins.describe()

# COMMAND ----------

# Explore whether there are some missing values
penguins.isnull().sum()

# COMMAND ----------

# Dropping missing values
penguins.dropna(inplace = True)

# COMMAND ----------

# Checking for duplicated data
penguins.duplicated().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Numerical variables

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Histogram
# MAGIC 
# MAGIC We'll look at the distribution of our data using the `displot()` function where we specify parameters such as `data` and `x` that define a position on the x-axis. `displot()` is a Figure-level method and the size of the output can be changed using the parameters `height` and `aspect`. In case of Axes-level functions, the size can be controlled with `plt.figure(figsize = (Width, height in inches))`. You can find all necessary information in Seaborn's documentation.
# MAGIC 
# MAGIC This function uses the same underlying code as `histplot()` function. Moreover it provides different approaches for visualizing the distribution. The histogram will be drawn by default. But we can choose a particular approach with the `kind` parameter:  
# MAGIC `kind = 'hist'`   
# MAGIC `kind = 'kde'`  
# MAGIC `kind = 'ecdf'`  
# MAGIC 
# MAGIC All of these approaches to visualize distributions have their very own function in the _distribution module_ and belong to the _distribution plots category_. We'll discuss all approaches later on. 
# MAGIC 
# MAGIC Now let's see how we can display the distribution of the length of penguins' bills. Seaborn's function `displot()` returns a Matplotlib's FacetGrid object. Assign the resulting object to the `ax` variable to be able add things such as title or axes labels.
# MAGIC 
# MAGIC Another way is to use `plt.title()`, `plt.xlabel()` and `plt.ylabel()`.

# COMMAND ----------

# Create a histogram of 'bill_length_mm'
ax = sns.displot(data = penguins, 
                 x = 'bill_length_mm',
                 height = 6.5,
                 aspect = 1.3);

# Setting a title
ax.set(title = 'The distribution of bill length');

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, Seaborn automatically annotates labels according to defined parameters. We can see from the plot that the most common bill length is about 45 mm. There are very few penguins that have bill length less than 10 mm. 
# MAGIC 
# MAGIC As we have already learned, it is always appropriate to experiment and try different numbers of bins as well as change the size of bins. By default, `displot()` and `histplot()` plotting functions determine the size of the bins based on the number of observations and the variance. 
# MAGIC 
# MAGIC If you want to zoom in into the particular area of a histogram, you can do so by limiting the axis using Matplotlib's `xlim` (alteratively `ylim`). The options are:
# MAGIC 
# MAGIC - `plt.xlim(left, right)` - setting values for the left and the right limit
# MAGIC - `plt.xlim(left)` - setting a value only for left limit
# MAGIC - `plt.xlim(right)` - setting a value only for right limit

# COMMAND ----------

# A displot of 'bill_length_mm' with specified axis limits from value 53 to 60
sns.displot(data = penguins, 
            x = 'bill_length_mm')

# Setting the right and left limit
plt.xlim(53,60);

# COMMAND ----------

# MAGIC %md
# MAGIC Changing the size of the bins can be acomplished with the `binwidth` parameter. If we set `binwidth = 2`, each bin will coumpound observations in the range of 2 millimeters: 

# COMMAND ----------

# Histogram with specified binwidth
sns.displot(data = penguins, 
            x = 'bill_length_mm', 
            binwidth = 2,
            height = 6.5,
            aspect = 1.3);
# Change the size of the bins yourself and observe the output

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, we can control the number of bins. There is no right answer to the question of what number of bins you should set. 
# MAGIC 
# MAGIC It always depends on the data and our aim is to choose the most appropriate one that describes the data the best. If we didn't alter the number of bins during the exploration, we might miss important patterns. 
# MAGIC 
# MAGIC You can assign the number of bins to the `bins` parameter.

# COMMAND ----------

# Histogram with specified number of bins
sns.displot(data = penguins, 
            x = 'bill_length_mm', 
            bins = 30,
            height = 6.5,
            aspect = 1.3);
# Again, replace the number of bins and observe the output

# COMMAND ----------

# MAGIC %md
# MAGIC In the above histogram, we can see that the number of bins is too big, since the gap appeared after value 55. Try change this number yourself and observe the output.

# COMMAND ----------

# TASK 1 >>> Create a histogram showing the distribution of penguins's body mass ('body_mass_g' variable)
#        >>> Just define data and a variable and let Seaborn create the default plot
#        >>> Set parameters: height = 6.5, aspect = 1.3

sns.displot(data = penguins, 
            x = 'body_mass_g',
            height = 6.5,
            aspect = 1.3);

# COMMAND ----------

# MAGIC %md
# MAGIC >What can you say about the distribution of body mass? Does the default bins correctly captured pattern in data? What is the most common weight of penguins? 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Plotting the distribution using KDE plot
# MAGIC 
# MAGIC KDE abbreviation stands for Kernel Density Estimate and using this approach we can create distribution using continuous probability density curve. KDE is calculated using a specific formula which you do not need to worry about. There are however mathematical and statistical reasons why sometimes it might be more appropriate to show a KDE.

# COMMAND ----------

# The distribution using kernel density estimation

sns.displot(data = penguins, 
            x = 'bill_depth_mm', 
            bins = 20,
            kde = True,
            height = 6.5,
            aspect = 1.3);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Boxplot
# MAGIC 
# MAGIC Now we'll look at the distribution of penguins' body mass using boxplot. 
# MAGIC 
# MAGIC Seaborn's `boxplot()` function takes several parameters. Refer to the documentation [here](https://seaborn.pydata.org/generated/seaborn.boxplot.html) to learn more. 
# MAGIC 
# MAGIC Below you can see what the default boxplot looks like. We passed our dataset to the `data` parameter and the 'body_mass_g' feature as the input of the x parameter. 

# COMMAND ----------

# Boxplot of 'body_mass_g' variable

plt.figure(figsize=(13,9))
sns.boxplot(data = penguins, 
            x = 'body_mass_g');

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4.1 Customizing boxplots
# MAGIC 
# MAGIC If you want to further customize a boxplot, you can do so with Matplotlib's help.
# MAGIC 
# MAGIC Since Seaborn's boxplot is fancier version of Matplotlib's boxplot, you can again use the same parameters to control boxplot's appearance. For example, changing the style of box (`boxprops`), whiskers (`whiskerprops`), emphasizing median value (`medianprops`) or outliers (`flierprops`) if present. 
# MAGIC 
# MAGIC You can specify and pass these properties within a dictionary and then insert it into the boxplot plotting function. These `props` dictionaries refer to the class [`Line2D`](https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D) and you can further customize those parameters  present in this class.

# COMMAND ----------

# Small customizations of boxplot

plt.figure(figsize=(13,9))
sns.boxplot(data = penguins, 
            y = 'body_mass_g',
            color = 'cadetblue',
            width = 0.2,
            linewidth = 3,
            medianprops={'color':'white'});

# COMMAND ----------

# MAGIC %md
# MAGIC In our example we customize the color, the width of the boxplot and the width of the lines. Changing the color of the line presenting the median value can be useful in order to emphasize it. 

# COMMAND ----------

# Some advanced customizations
plt.figure(figsize=(13,9))

box_cust = dict(color = '#834177',                # customizing properties of the box 
                alpha = 0.5, 
                linestyle = 'dashed', 
                linewidth = 6)

whisker_cust = dict(color = '#000184',            # customizing properties of whiskers
                    alpha = 0.9,                  # the transparency
                    linestyle = 'dotted',         # the style of the line
                    linewidth = 3,                # the width of the line
                    dash_capstyle = 'projecting') # setting the cap style for dashed line

median_cust = dict(color = '#ff7f0e',
                  alpha = 0.9,
                  linestyle = 'dashdot',
                  linewidth = 5)

ax = sns.boxplot(data = penguins, 
                y = 'flipper_length_mm',
                width = 0.2,
                linewidth = 3,
                boxprops = box_cust,
                whiskerprops = whisker_cust,
                medianprops = median_cust,
                notch = True);                    # noteched boxplot (notch represent the confidence interval around median) 

# COMMAND ----------

# TASK 2 >>> Create a boxplot of the 'bill_length_mm' feature
#        >>> Create a figure and set its size to (13,9)
#        >>> Set color as hexadecimal code: '#98D8D8',
#        >>> Set width of a box to 0.3
#        >>> Set width of a line to 1
#        >>> Change a color of median line to be yellow and line width: 3 
#            (you can specify medianprops properties within boxplot() function or outside this function)

plt.figure(figsize=(13,9))
sns.boxplot(data = penguins, 
            y = 'bill_length_mm', 
            color = '#98D8D8', 
            width = 0.3, 
            linewidth = 1, 
            medianprops={'color':'yellow','linewidth':3});

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Swarmplot
# MAGIC 
# MAGIC The `swarmplot()` plotting function is useful if we want to better see the distribution of the values. In this case, each data point will be drawn and adjusted to avoid overlapping of values. You can, of course, create a swarmplot on its own, but it's nice to see drawn values on the top of distribution created with boxplot. 
# MAGIC 
# MAGIC There is one disadvantage of the swarmplot: if you have dataset with large number of observations, let's say in thousands, there will be a huge overlap of data points. We can, however, take a random sample of only a few percent of points to be able to utilize the swarmplot. In such case, do not forget to mention next to your visual that the swarmplot displays only a subsample of data. In the case of the penguins dataset, swarmplot is a good choice of plotting method and we can nicely see the drawn observations. 

# COMMAND ----------

# Distribution of data displayed with boxplot and swarmplot

plt.figure(figsize=(13,9))
sns.boxplot(data = penguins,
            y = 'body_mass_g',
            color = 'skyblue',
            width = 0.2,
            linewidth = 1)
sns.swarmplot(data = penguins, 
              y = 'body_mass_g',
              color = 'white',
              edgecolor = 'violet',    # the color of line around data point
              linewidth = 1);          # The width of line that frame data point

# COMMAND ----------

# TASK 3 >>> Create a swarmplot of the 'bill_depth_mm' feature
#        >>> Create a figure and set its size to (13,9)
#        >>> Set the color of the data points to 'coral'

plt.figure(figsize=(13,9))
sns.swarmplot(data = penguins, 
              y = 'bill_depth_mm', 
              color = 'coral');

# COMMAND ----------

# MAGIC %md
# MAGIC > Could you exactly say where the majority of data points lie based on the drawn data points? Sometimes, judging the distribution's shape from swarmplot can be tricky. To be sure, it's better to create a swarmplot in conjunction with a boxplot. 

# COMMAND ----------

# TASK 4 >>> Create a boxplot of the 'bill_depth_mm' feature
#        >>> Create a figure and set its size to (13,9)
#        >>> Set the width of boxplot to 0.2, the width of the line to 3 and the color to 'white'
#        >>> Plot a swarmplot right after the boxplot (just copy and paste the line of code you created in TASK 3

plt.figure(figsize=(13,9))
sns.boxplot(data = penguins, 
            y = 'bill_depth_mm', 
            width = 0.2, 
            color = 'white', 
            linewidth = 3)
sns.swarmplot(data = penguins, 
              y = 'bill_depth_mm', 
              color = 'coral');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6 Stripplot
# MAGIC 
# MAGIC Stripplots are very similar to swarmplots since they also show all of the data. There is the same disadvantage of the data overlapping, but you can add some random noise (jitter) among values. Sometimes it's hard to tell what the distribution of the data is without additional representation of the underlying distribution (for instance, created with a boxplot). 
# MAGIC 
# MAGIC You can change the amount of jitter using the `jitter` parameter where you have 2 options:
# MAGIC - set `jitter = True` for a good default amount of jitter
# MAGIC - specify amount of jitter 
# MAGIC 
# MAGIC Stripplots can be useful for indicating outliers in the data, too. 

# COMMAND ----------

# Stripplot of the 'body_mass_g' feature

plt.figure(figsize=(13,9))
sns.stripplot(data = penguins, 
              x = 'body_mass_g', 
              color = 'green', 
              jitter = 0.2, 
              size = 6);

# COMMAND ----------

# MAGIC %md
# MAGIC Since we do not have so many observations, we can better estimate the distribution of the data because the data poins do not overlap. Here, we can take into consideration the density of the data points which can gives us a good approximation of the shape. There are more data points in the range of 3200-3800 grams compared to the rest of data points. After that, the data points become more sparse. Based on that, we would say that the distribution seems to be right-skewed. In most cases, it would not be appropriate to assume the distribution only from the stripplot because it can be misleading. Therefore, always visualize the data using several plotting approaches. 

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Categorical variables

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Catplot
# MAGIC 
# MAGIC To create a visualization of a categorical variable you can use the `catplot()` plotting function. This is a Figure-level interface which allows you to specify a certain plot type such as boxplot using the `kind` parameter. 
# MAGIC 
# MAGIC For example, if you want to visualize the number of occurences of observations based on the specific category the code would be:

# COMMAND ----------

# The count of penguins based on the island

sns.catplot(data = penguins,
           x = 'island',
           kind = 'count',
           height = 6.5,
           aspect = 1.3);

# COMMAND ----------

# MAGIC %md
# MAGIC This is the default output of a countplot. The visual appearance of a countplot is similar to a histogram, the values are placed within the respective bars.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Countplot
# MAGIC 
# MAGIC Alternatively, we can use the Axis-level ploting function `sns.countplot()`.

# COMMAND ----------

# Create a countplot of the 'island' feature

plt.figure(figsize = (13,9))
ax = sns.countplot(data = penguins, 
                   x = 'island',
                   order = penguins.island.value_counts().index, # Order categories by their value counts
                   palette = 'Paired')
ax.set(title = 'Count of penguins based on the island', 
       xlabel = 'The island',
       ylabel = 'Count');

# COMMAND ----------

# MAGIC %md
# MAGIC Here we order the categories based on their values in descending order. Another way to change the order based on your preference is to define categories in the list:
# MAGIC ```
# MAGIC order = ['Biscoe','Dream','Torgersen']
# MAGIC ```
# MAGIC 
# MAGIC All of customizations we did are applicable also with `catplot()`.

# COMMAND ----------

# TASK 5 >>> Display the count of penguins with respect to the species
#        >>> Use the catplot() plotting function
#        >>> Order the species in the following way: Chinstrap, Adelie, Gentoo
#        >>> Set the saturation of colors to 0.3
#        >>> Change the size of the figure:
#            parameter: height = 6.5
#            parameter: aspect = 1.3
#        >>> Set the title to 'Count of penguins based on species'

sns.catplot(data = penguins,
           x = 'species',
           kind = 'count',
           order = ['Chinstrap','Adelie','Gentoo'],
           palette = 'Set2',
           saturation = 0.3,
           height = 6.5,
           aspect = 1.3)
plt.title('Count of penguins based on species');

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Tasks
# MAGIC 
# MAGIC Now it's your turn to create some visualization of the Bank Marketing dataset. The data is related to a marketing campaign of a Portuguese banking institution that was performed via phone communication with clients of the bank. Let's look at attributes and do some preprocessing. 
# MAGIC 
# MAGIC **Attributes explanation:**
# MAGIC 
# MAGIC Bank client data:
# MAGIC - age
# MAGIC - job : type of job
# MAGIC - marital : marital status
# MAGIC - education
# MAGIC - default: has credit in default?
# MAGIC - housing: has housing loan?
# MAGIC - loan: has personal loan?
# MAGIC 
# MAGIC Related with the last contact of the current campaign:
# MAGIC - contact: contact communication type
# MAGIC - month: last contact month of year
# MAGIC - day_of_week: last contact day of the week
# MAGIC - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# MAGIC 
# MAGIC Other attributes:
# MAGIC - campaign: number of contacts performed during this campaign and for this client
# MAGIC - pdays: number of days that passed by after the client was last contacted from a previous campaign (999 means client was not previously contacted)
# MAGIC - previous: number of contacts performed before this campaign and for this client
# MAGIC - poutcome: outcome of the previous marketing campaign
# MAGIC 
# MAGIC Social and economic context attributes
# MAGIC - emp.var.rate: employment variation rate - quarterly indicator
# MAGIC - cons.price.idx: consumer price index - monthly indicator
# MAGIC - cons.conf.idx: consumer confidence index - monthly indicator
# MAGIC - euribor3m: euribor 3 month rate - daily indicator
# MAGIC it is calculated by eliminating the highest 15% and the lowest 15% of the interest rates submitted and calculating the arithmetic mean of the remaining values
# MAGIC - nr.employed: number of employees - quarterly indicator
# MAGIC 
# MAGIC Target variable:
# MAGIC - y - has the client subscribed to a term deposit?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 Loading data, preprocessing

# COMMAND ----------

# Load the data 
data = pd.read_csv('../Data/bank_data.csv',sep = ';')
data.head()

# COMMAND ----------

# Let's rename some of variables
data.rename(columns = {'marital':'marital_status','default':'default_credit','housing':'house_loan',
                      'contact':'contact_type','duration':'contact_duration','campaign':'number_of_contacts',
                      'pdays':'days_passed','previous':'number_previous_contact','poutcome':'previous_campaign_outcome',
                      'emp.var.rate':'emp_variation_rate','cons.price.idx':'cpi','cons.conf.idx':
                      'cci','euribor3m':'euribor_rate','nr.employed':'no_employees','y':'target'},
           inplace = True)

# COMMAND ----------

# Examine summary statistics
data.describe()

# COMMAND ----------

# Check for missing values
data.isnull().sum()

# COMMAND ----------

# Check for duplicated data
duplicated_rows = data[data.duplicated()]
duplicated_rows

# COMMAND ----------

# MAGIC %md
# MAGIC There are 12 duplicated rows in the dataset that need to be removed. 

# COMMAND ----------

# Remove duplicated rows

data.drop_duplicates(inplace = True)

# COMMAND ----------

# Examine data types of variables

data.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC It seems that the data type of the numerical features has been correctly recognized by Python. All non-numerical features have the _object_ data type, so let's  convert them to categories.

# COMMAND ----------

# Select all variables that need to be converted

to_category = ['job','marital_status','education','default_credit','house_loan', 
               'loan','contact_type','month','day_of_week', 'previous_campaign_outcome']

# COMMAND ----------

# Convert them to the category data type

for col in to_category:
    data[col] = data[col].astype('category')

# COMMAND ----------

# MAGIC %md
# MAGIC Days and months are recorded as their abbreviations. Let's map them to their full name.

# COMMAND ----------

# Create a dictionary with original values and corresponding new values for days

mapping_days = {'mon':'Monday','tue':'Tuesday','wed':'Wednesday','thu':'Thursday','fri':'Friday'}

# COMMAND ----------

# Map the new values to the column 'day_of_week'

data.day_of_week = data.day_of_week.map(mapping_days)

# COMMAND ----------

# Create a dictionary with the original values and the corresponding new values for the months

mapping_months = {'mar':'March', 'apr':'April','may':'May','jun':'Jun','jul':'Jul','aug':'August',
                  'sep':'September','oct':'October','nov':'November','dec':'December'}

# COMMAND ----------

# Map the new values to the column 'month'
data.month = data.month.map(mapping_months)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 Numerical variables
# MAGIC 
# MAGIC Let's start with the numerical features and explore the distribution of the data points. 

# COMMAND ----------

# TASK 6 >>> Create a histogram of 'age' variable
#        >>> Set parameters: height = 6.5, aspect = 1.3
#        >>> Set the number of bins to 25
#        >>> Set the title: 'Age distribution'

sns.displot(data = data,
            x = 'age', 
            bins = 25,
            height = 6.5,
            aspect = 1.3)
plt.title('Age distribution');

# COMMAND ----------

# TASK 7 >>> Create a boxplot of the 'age' feature
#        >>> Set a the figure size to (13,9)
#        >>> Assign variable 'age' to the y parameter
#        >>> Set the width of the boxplot to 0.3

plt.figure(figsize = (13,9))
sns.boxplot(data = data, 
            y = 'age', 
            width = 0.3);

# COMMAND ----------

# MAGIC %md
# MAGIC The values of the feature 'contact_duration' are recorded in seconds. Run the line below to convert them to minutes.

# COMMAND ----------

# Converting seconds to minutes
data.contact_duration = data.contact_duration.apply(lambda x: x / 60)

# COMMAND ----------

# TASK 8 >>> Create a histogram of 'contact_duration'
#        >>> Set the width of the bins to 0.5, so every bin contains a call duration of 50 seconds

sns.displot(data = data, 
            x = 'contact_duration', 
            binwidth = 0.5);

# COMMAND ----------

# MAGIC %md
# MAGIC There are some records where no call was performed, so the corresponding values are of value 0. 
# MAGIC Let's take a look at the rows with no recorded duration. We will drop them since they do not provide us with any useful information. 

# COMMAND ----------

# Print only those rows where 'contact_duration' is 0
data[data['contact_duration'] == 0]

# COMMAND ----------

# Get the index of rows that should be dropped
index_rows_to_drop = data[data['contact_duration'] == 0].index

# COMMAND ----------

# Drop these rows from the dataframe
data.drop(index_rows_to_drop, inplace = True)

# COMMAND ----------

# TASK 9 >>> Recreate a histogram of 'contact_duration'
#        >>> Set parameters: height = 6.5, aspect = 1.3
#        >>> Zoom in and set the x-axis limit from 0.5 to 25 minutes
#        >>> Set the title: 'Call duration up to 25 min'

sns.displot(data = data, 
            x = 'contact_duration',
            height = 6.5, 
            aspect = 1.3)
plt.xlim(0.5,25)
plt.title('Call duration up to 25 min');

# COMMAND ----------

# MAGIC %md
# MAGIC We already saw that the distribution of data points can be visualized using a stripplot. When we plotted some feature of the penguins dataset, we could nicely observe each data point in the figure and there was minimal overlap of values. Now we create a stripplot of the 'contact_duration' feature from the Bank marketing dataset to see how a stripplot looks like when you have thousands of records available. 

# COMMAND ----------

# TASK 10 >>> Create a stripplot of the 'contact_duration' feature
#         >>> Set a Figure size: (13,9)

plt.figure(figsize = (13,9))
sns.stripplot(data = data, 
              x = 'contact_duration');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 Categorical variables

# COMMAND ----------

# TASK 11 >>> Create a countplot of the 'job' feature to see the client's job and corresponding counts
#         >>> Set a the figure size to (13,9)
#         >>> Order the counts by the most common job
#         >>> Set the rotation of x-axis tick labels to 45 degrees using Matplotlib's xticks()
#         >>> Set the color palette to 'Set3'

plt.figure(figsize = (13,9))
sns.countplot(data = data, 
              x = 'job', 
              order = data.job.value_counts(ascending = False).index, 
              palette = 'Set3')
plt.xticks(rotation = 45);

# COMMAND ----------

# MAGIC %md
# MAGIC Now we'll look at how many calls have been performed during the respective months. We are missing January in the dataset for some reason, but don't worry about it. To correctly visualize the months of the year we need to specify their order.

# COMMAND ----------

# Unique categories
month_order = ['February','March','April','May','Jun','Jul','August','September','October','November','December']

# COMMAND ----------

# Creating CategoricalDtype
order_cat = pd.api.types.CategoricalDtype(categories = month_order, ordered = True)

# COMMAND ----------

# Change data type of month variable as order_cat data type
data.month = data.month.astype(order_cat)

# COMMAND ----------

# TASK 12 >>> Create a countplot of 'month' variable to see how many calls have been performed through the months
#         >>> Set the figure size to (13,9)
#         >>> Set the color palette to 'Pastel1'
#         >>> Set the rotation of x-axis tick labels to 45 degrees using Matplotlib's xticks()

plt.figure(figsize = (13,9))
sns.countplot(data = data, 
              x = 'month', 
              # order = data.month.value_counts(ascending = False).index, 
              palette = 'Pastel1')
plt.xticks(rotation = 45);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Citation request:
# MAGIC 
# MAGIC [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
# MAGIC 
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science) 
