
### Correlation
- Average: A measurement of central tendency (mean, median, or mode)
- Standard deviation: A measurement of spread
- Pearson’s correlation coefficient = correlation coefficient
  - $r = \frac{covariance(X,Y)}{(SD X)(SD Y)}$
  - $covariance(X,Y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{n}$

### Underfitting and Overfitting
- Bias versus Variance tradeoff
  - A model that underfits the sample data is described as having a high bias
  - A model that does not perform well on new data is described as having high variance

### Linear Regression
- Residuals are the difference between the predicted and observed values.
  - Residuals are used to estimate errors when checking the normality and homoscedasticity assumptions of linear regression.
- Errors are the natural noise assumed to be in the model.
- The four main assumptions of simple linear regression
  - Linearity: Each predictor variable (Xi) is linearly related to the outcome variable (Y).
  - Normality: The errors are normally distributed.
  - Independent Observations: Each observation in the dataset is independent.
  - Homoscedasticity: The variance of the errors is constant or similar across the model.
- The quantile-quantile plot (Q-Q plot)
  - A graphical tool used to compare two probability distributions by plotting their quantiles against each other.
  - Data professionals often prefer Q-Q plots to histograms to gauge the **normality of a distribution**
```
import statsmodels.api as sm
import matplotlib.pyplot as plt

residuals = model.resid
fig = sm.qqplot(residuals, line = 's')
plt.show()
```
- What to do if an assumption is violated
  - Linearity
    - Transform one or both of the variables, such as taking the logarithm.
    - For example, if you are measuring the relationship between years of education and income, you can take the logarithm of the income variable and check if that helps the linear relationship.
  - Normality
    - Transform one or both variables. Most commonly, this would involve taking the logarithm of the outcome variable.
    - When the outcome variable is right skewed, such as income, the normality of the residuals can be affected.
      - So, taking the logarithm of the outcome variable can sometimes help with this assumption.
    - If you transform a variable, you will need to reconstruct the model and then recheck the normality assumption to be sure.
      - If the assumption is still not satisfied, you’ll have to continue troubleshooting the issue. 
  - Independent observations
    - Take just a subset of the available data.
    - If, for example, you are conducting a survey and get responses from people in the same household, their responses may be correlated.
      - You can correct for this by just keeping the data of one person in each household.
    - Another example is if you are collecting data over a time period.
      - Let’s say you are researching data on bike rentals.
      - If you collect your data every 15 minutes, the number of bikes rented out at 8:00 a.m. might correlate with the number of bikes rented out at 8:15 a.m.
      - But, perhaps the number of bikes rented out is independent if the data is taken once every 2 hours, instead of once every 15 minutes.
  - Homoscedasticity
    - Define a different outcome variable.
    - If you are interested in understanding how a city’s population correlates with the number of restaurants in a city,
      - you know that some cities are much more populous than others.
      - You can then redefine the outcome variable as the ratio of population to restaurants.
    - Transform the Y variable.
      - As with the above assumptions, sometimes taking the logarithm or transforming the Y variable in another way can potentially fix inconsistencies with the homoscedasticity assumption.

```
# Import packages
import pandas as pd
import seaborn as sns

# Load dataset
penguins = sns.load_dataset("penguins")

# Examine first 5 rows of dataset
penguins.head()

# Subset just Chinstrap penguins from data set
chinstrap_penguins = penguins[penguins["species"] == "Chinstrap"]
 
# Reset index of dataframe
chinstrap_penguins.reset_index(inplace = True, drop = True)

# Subset Data
ols_data = chinstrap_penguins[["bill_depth_mm", "flipper_length_mm"]]

# Write out formula
ols_formula = "flipper_length_mm ~ bill_depth_mm"

# Import ols function
from statsmodels.formula.api import ols

# Build OLS, fit model to data
OLS = ols(formula = ols_formula, data = ols_data)
model = OLS.fit()
model.summary()

predictions = model.predict(chinstrap_penguins[["bill_depth_mm"]])

residuals = model.resid
```
- $R^2$ : The coefficient of determination
  - $1 - \frac{Sum of squared residuals}{Total sum of squares}$ 
  - measures the proportion of variation in the dependent variable, Y, explained by the independent variable(s), X.
- Adjusted $R^2$
  - Penalizes the addition of more independent variables to the multiple regression model
  - Only captures the proportion of variation explained by the independent variables that show a significant relationship with the outcome variable
- MSE : Mean squared error
  - the average of the squared difference between the predicted and actual values.
  - Because of how MSE is calculated, MSE is very sensitive to large errors.
- MAE: Mean absolute error
  - the average of the absolute difference between the predicted and actual values.
  - If your data has outliers that you want to ignore, you can use MAE, as it is not sensitive to large errors.

### Multiple Linear Regression
- Multiple linear regression assumptions
  - Linearity: Each predictor variable ($X_i$) is linearly related to the outcome variable (Y).
    - With multiple linear regression, you need to consider whether each x variable has a linear relationship with the y variable.
    - You can make multiple scatterplots instead of just one, using seaborn’s pairplot() function or the scatterplot() function multiple times
  - (Multivariate) normality: The errors are normally distributed.
    - The independent observations assumption is still primarily focused on data collection.
    - You can check the validity of the assumption in the same way you would with simple linear regression.
  - Independent observations: Each observation in the dataset is independent.
    - Just as with simple linear regression, you can construct the model, and then create a Q-Q plot of the residuals. 
    - If you observe a straight diagonal line on the Q-Q plot, then you can proceed in your analysis.
    - You can also plot a histogram of the residuals and check if you observe a normal distribution that way.
    - It’s a common misunderstanding that the independent and/or dependent variables must be normally distributed when performing linear regression.
      - This is not the case. Only the model’s residuals are assumed to be normal.
  - Homoscedasticity: The variation of the errors is constant or similar across the model.
    - As with simple linear regression, for multiple linear regression, just create a plot of the residuals vs. fitted values.
    - If the data points seem to be scattered randomly across the line where residuals equal 0, then you can proceed.
  - No multicollinearity: No two independent variables ($X_i$ and $X_j$) can be highly correlated with each other

### Chi-squared ($X^2$) tests
- The Chi-squared goodness of fit test 
  - Identify the Null ($H_0$) and Alternative ($H_a$) Hypotheses 
  - Calculate the chi-square test statistic
    - $X^2 = \sum \frac{(observed - expected)^2}{expected}$
    - Quantify the extent of any discrepancies between observed frequencies and expected frequencies for each categorical level
  - Calculate the p-value
    - degrees of freedom = number of categorical levels – 1
  - Make a conclusion : If the p-value is less than 0.05, it is sufficient to suggest the alternative hypothesis

```
import scipy.stats as stats
observations = [650, 570, 420, 480, 510, 380, 490]
expectations = [500, 500, 500, 500, 500, 500, 500]
result = stats.chisquare(f_obs=observations, f_exp=expectations)
result
```

### ANOVA
- One-way ANOVA
  - Compares the means of one continuous dependent variable based on three or more groups of one categorical variable
  - Steps
    - Calculate group means and grand (overall) mean
    - Calculate the sum of squares between groups (SSB) and the sum of squares within groups (SSW)
    - Calculate mean squares for both SSB and SSW
    - Compute the F-statistic
      - the ratio of the mean sum of squares between groups (MSSB) to the mean sum of squares within groups (MSSW)
      - MSSB / MSSW
    - Use the F-distribution and the F-statistic to get a p-value, which you use to decide whether to reject the null hypothesis
- Two-way ANOVA
  - Compares the means of one continuous dependent variable based on three or more groups of two categorical variables

### Logistic Regression
- logit(p) = $log(\frac{p}{1-p})$

