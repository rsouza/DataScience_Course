# Databricks notebook source
# MAGIC %md
# MAGIC # Multivariate Analysis with Seaborn
# MAGIC 
# MAGIC Multivariate visualizations are an expansion of bivariate analysis, where we add another variable (or variables). Often, adding the third variable helps us to find some important pattern or information that we couldn't have observed before.

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# Load the data
penguins = pd.read_csv('Data/penguins.csv')
penguins.dropna(inplace = True)

# COMMAND ----------

# Take a look at first 5 rows
penguins.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Mapping the third variable to encoding
# MAGIC 
# MAGIC There are 3 ways how to map the third variable to create a visual semantic:
# MAGIC 
# MAGIC - **encoding with color**
# MAGIC - **encoding with the size**
# MAGIC - **encoding with the shape**
# MAGIC 
# MAGIC Again, choosing appropriate encoding depends on the question we ask, input data or purpose of visualizations. Let's look at some examples.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Visualizing the distribution

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Boxplot
# MAGIC 
# MAGIC As we mentioned in the Bivariate analysis notebook, boxplots are great when becomes to comparing several groups. Let's say we want to see the distribution of penguins body mass based on the island. We are also interested whether there are some differences in the ranges of the values between females and males. As before, we plot the first categorical variable 'island', then numerical variable 'body_mass_g' and pass the third groupiny variable 'sex' to `hue` parameter.
# MAGIC 
# MAGIC Here, the third variable is mapped with **color encoding** that produces different colors and visually help determines levels of a subset.

# COMMAND ----------

# Boxplots of body mass based on the island and the gender

plt.subplots(figsize = (13,9))
sns.boxplot(data = penguins, 
            x = 'island', 
            y = 'body_mass_g', 
            hue = 'sex', 
            palette = 'Set3',
            linewidth = 0.6)
plt.xlabel('The island', fontsize = 14, labelpad = 20)     # Setting the title, fontsize and adjusting the spacing
plt.ylabel('Body mass (g)', fontsize = 14)
plt.title('The distribution of body mass', fontsize = 20);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Stripplots
# MAGIC 
# MAGIC A stripplot is another kind of categorical scatterplot that can be useful when comparing different groups. Again, the categories of the third variable are distinguished using **different colors**.

# COMMAND ----------

# Stripplots

plt.subplots(figsize = (13,9))
sns.stripplot(data = penguins, 
              x = 'species', 
              y = 'flipper_length_mm', 
              hue ='sex')
plt.xlabel('Species', fontsize = 14, labelpad = 20)     
plt.ylabel('The flipper length (mm)', fontsize = 14)
plt.title('The distribution of the flipper length', 
          fontsize = 20)
plt.legend(loc = 4, prop = {'size': 13});           # Adjusting the legend's position and the size

# COMMAND ----------

# MAGIC %md
# MAGIC In the above plot we can observe the flipper length distribution based on species and the gender of penguins. We can immediately see some differences and similarities between species thanks to adding a third variable.
# MAGIC 
# MAGIC ---
# MAGIC ## 1.3 Relplot
# MAGIC 
# MAGIC When we want to see a possible relationship between variables we can choose between three encoding approaches and decide which kind is the most suitable. In the below example we can see how body mass and the flipper length relate based on penguins's species.

# COMMAND ----------

# A scatterplot of body mass and the flipper length based on species

sns.relplot(data = penguins, 
            x = 'body_mass_g', 
            y = 'flipper_length_mm', 
            hue = 'species',
            palette = 'Dark2',
            height = 7,
            aspect = 1.5,)
plt.xlabel('Body mass (g)', fontsize = 14, labelpad = 20)     
plt.ylabel('The flipper length (mm)', fontsize = 14)
plt.title('The relationship of body mass and the flipper length', fontsize = 20);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 Scatterplot
# MAGIC 
# MAGIC In some cases, encoding with the third variable with **the size** can emphasize important aspects we found during exploratory data analysis. The graph below shows that the Gentoo species' penguin has the highest body mass and the longest flippers.

# COMMAND ----------

# A scatterplot 

fig, ax = plt.subplots(figsize = (13,9))
sns.scatterplot(data = penguins, 
                x = 'body_mass_g', 
                y = 'flipper_length_mm', 
                size = 'species',
                color = 'green')
plt.xlabel('Body mass (g)', fontsize = 14, labelpad = 20)     
plt.ylabel('The flipper length (mm)', fontsize = 14)
plt.title('The relationship of body mass and the flipper length', fontsize = 20);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5 Lmplot
# MAGIC 
# MAGIC Sometimes, we want to emphasize different categories of subset more sophisticatedly. In that case, we can choose specific  **markers** for each category.

# COMMAND ----------

# A lmplot

sns.lmplot(data = penguins, 
           x = 'bill_length_mm', 
           y = 'body_mass_g', 
           hue = 'species', 
           markers = ['+','8','*'],
           palette = 'Dark2',
           height = 7,
           aspect = 1.3)
plt.xlabel('The bill length (mm)', fontsize = 14, labelpad = 20)     
plt.ylabel('The body mass (g)', fontsize = 14)
plt.title('The relationship of body mass and the bill length', fontsize = 20);

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.  FacetGrid
# MAGIC 
# MAGIC Sometimes we want to display a relationship or a distribution not in a single Axes, but create a separate subplots. This can be done using a FacetGrid object, where we specify 3 dimensions:
# MAGIC - row 
# MAGIC - col
# MAGIC - hue - plotting different subset
# MAGIC 
# MAGIC Let's say we want to look at the distribution of penguins species, so we assign 'species' to `col` parameter.

# COMMAND ----------

# Initializing a FacetGrid object
g = sns.FacetGrid(penguins, col = 'species')

# COMMAND ----------

# MAGIC %md
# MAGIC When we initialized FacetGrid object, a Figure and Axes will be returned. To create some plot we apply `.map()` on a FacetGrid, where we specify plotting function and variables we want to plot.

# COMMAND ----------

# Initializing a FacetGrid object and col parameter

g = sns.FacetGrid(penguins, 
                  col = 'species',
                  height = 4,
                  aspect = 1)
# Mapping plotting function and defining a variable
g.map(sns.histplot, 'body_mass_g');

# COMMAND ----------

# MAGIC %md
# MAGIC Let's add the third variable using `row` parameter. We want to see the distribution of body mass also based on species gender.

# COMMAND ----------

# Initializing a FacetGrid object and col and row parameters

g = sns.FacetGrid(penguins, 
                  col = 'species', 
                  row = 'sex')
# Mapping plotting function and defining variable
g.map(sns.histplot, 'body_mass_g', color = 'paleturquoise');

# COMMAND ----------

# MAGIC %md
# MAGIC To visualize a relationship between 2 numerical variables we just add the names of the particular features. Let's visualize a relationship between body mass and the flipper length based on species. We also add the 'sex' variable encoded using color.

# COMMAND ----------

# Initializing a FacetGrid object and col parameter

g = sns.FacetGrid(data = penguins, 
                  col = 'species', 
                  hue = 'sex',
                  height = 4,
                  aspect = 1, 
                  palette = 'Accent')
# Mapping plotting function and defining variable
g.map(sns.scatterplot, 'body_mass_g', 'flipper_length_mm')
# Setting x.axis and y-axis labels
g.set_axis_labels('Body mass (g)', 'The flipper length (mm)')
# Displaying the legend
g.add_legend();

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. PairGrid
# MAGIC 
# MAGIC Pairwise relationships of variables can be visualized using PairGrid. The initialization of a PairGrid results in a subplot grid with multiple Axes. Then we can call Axes-level plotting functions to draw plots in the upper and lower triangles and the marginal distribution of variables can be drawn along the diagonal. Creation of a PairGrid and a FacetGrid is similar, but the main difference is that using a FacetGrid you are allowed to use only one specific plotting function that is applied on each subplot. 
# MAGIC 
# MAGIC ## 3.1 Customizations of a PairGrid
# MAGIC 
# MAGIC You can customize a PairGrid output in several ways, all of which are described in the [documentation](https://seaborn.pydata.org/generated/seaborn.PairGrid.html#seaborn.PairGrid).
# MAGIC 
# MAGIC Since the upper and lower triangles have mirrored plots you can specify different plotting functions using `map_upper()` or `map.lower()`. 
# MAGIC There are also possibilities to encode the third variable in plots other than through color.  
# MAGIC 
# MAGIC Similar result can be accomplished using a high-level interface `pairplot()`. However, if you want to have more control of subplot grid, use a PairGrid.

# COMMAND ----------

# Defining colors for categories
palette = ['cornflowerblue','lightgreen','gold']

# Setting a palette
sns.set_palette(sns.color_palette(palette))

# COMMAND ----------

# Initializing a PairGrid object
g_grid = sns.PairGrid(penguins, hue = 'species')

# Plotting univariate plot on diagonal subplots
g_grid.map_diag(sns.kdeplot, fill = True)

# Plotting relational plot on the off-diagonal subplots
g_grid.map_offdiag(sns.scatterplot)
g_grid.add_legend();

# COMMAND ----------

# MAGIC %md
# MAGIC # Task for you

# COMMAND ----------

# TASK >>> How do the bill length and bill depth relate to each other based on penguin species ? 

# COMMAND ----------

# MAGIC %md
# MAGIC Some material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)
