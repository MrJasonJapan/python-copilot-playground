from prep_data import prep_data
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
    
df = prep_data()

# 2) --------- Exporatory Data Analysis ---------

# print a stastics summary of the dataframe, including for object columns
print(df.describe(include='all'))

# Use value counts on drive-wheels to get a count of each type of drive wheel.
print(df['drive-wheels'].value_counts())

# Using Seaborn, create a box-plot of price vs drive-wheels, with drive-wheels on the x-axis and price on the y-axis. Exclude any NaN values.
# df.boxplot(column='price', by='drive-wheels', showfliers=False) # using standard matplotlib
sns.boxplot(x='drive-wheels', y='price', data=df) # for some reason this doesn't work on my machine
plt.show()

# Use a scatter plot to show the relationship between engine size and price
plt.scatter(df['engine-size'], df['price'])
plt.xlabel('engine-size')
plt.ylabel('price')
plt.title('engine-size vs price')
plt.show()

# group the data by drive-wheels and body-style, and then calculate the mean price for each group
df_test = df[['drive-wheels', 'body-style', 'price']]
# as_index=False means don't use the groupby column as the index. this is useful if you want to use the groupby column as a regular column
df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean() 
# pivot the data so that drive-wheels is the index, body-style is the column, and price is the value.
df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')
# print the pivot table
print(df_pivot)

# convert the pivot table into a heatmap, and show the color bar. Lable the x and y axis, and give the plot a title
plt.pcolor(df_pivot, cmap='RdBu') # RdBu stands for Red-Blue
plt.colorbar()
plt.xlabel('body-style')
plt.ylabel('drive-wheels')
plt.title('price vs drive-wheels and body-style')
plt.show()

# Use Analysis of Variance (ANOVA) to determine if there is a significant difference between the average price of cars between Honda and Subaru
# first, create a dataframe with only the price and make columns
df_anova = df[['make', 'price']]
# group the data by make, and then calculate the mean price for each group
grouped_anova = df_anova.groupby(['make'])
# use the f_oneway function from the scipy.stats module to calculate the ANOVA. 
# Notice how the prices between Honda and Subaru are very similar, and we can confim this because the f-test score is less than 1, and the p-value is greater than 0.05.
f_val, p_val = stats.f_oneway(grouped_anova.get_group('honda')['price'], grouped_anova.get_group('subaru')['price'])
print('ANOVA results: F=', f_val, ', P=', p_val)

# Do the same for Honda and Juaguar
# Notice how the prices between Honda and Jaguar are very different, and we can confim this because the f-test score is greater than 1 (around 400), and the p-value is relatively small.
f_val, p_val = stats.f_oneway(grouped_anova.get_group('honda')['price'], grouped_anova.get_group('jaguar')['price'])
print('ANOVA results: F=', f_val, ', P=', p_val)

# print a regression line for engine-size vs price
sns.regplot(x='engine-size', y='price', data=df)
plt.show()

# reset the plot
plt.clf()

# on the other hand print a regression line for highway-mpg vs price
sns.regplot(x='highway-mpg', y='price', data=df)
plt.ylim(0,)
plt.show()

# determine the characteristics of the data that have the highest correlation with price
#   note that we are only interested in the correlation of the numeric columns with price
#   we will use the Pearson correlation method
#   we will only show the correlation of the top 10 columns with price.
#   Include both positive and negative correlations.
# Notice how engine-size and highway-mpg have the highest correlation with price.
df_corr = df.corr()
print(df_corr['price'].sort_values(ascending=False)[1:6])
print(df_corr['price'].sort_values(ascending=False)[-5:])
# Print the p-values as well (for engine-size and highway-mpg)
# Note that the p-values are very small, which means that the correlation is statistically significant.
# For example a p-value less than 0.001 means that there is a 99.9% chance that the correlation is statistically significant.
print(stats.pearsonr(df['engine-size'], df['price']))
print(stats.pearsonr(df['highway-mpg'], df['price']))

# Cretae a heatmap of the correlation matrix
sns.heatmap(df_corr, cmap='RdBu')
plt.show()




