### Feature Transformation

- Log normalization : skewed data -> log-normal distribution
- Scaling
  - Normalization : MinMaxScaler in scikit-learn : $\frac{x_i - x_min}{x_max - x_min}$
  - Standardization : StandardScaler in scikit-learn : $\frac{x_i - x_mean}{x_std}$ 
- Encoding : categorical data -> numerical data

### Imbalanced Datasets

- Downsampling : make the minority class represent a larger share of the whole dataset by removing observations from the majority
- Upsampling : when the dataset doesn't have a very large number of observations in the first place

### Bayes' Theorem
- Find the probability of an event, A, given that another event B is true
- $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
- $P(B|C, A) = P(B|A)$
  - given A, introducing C does not change the probability of B
  - In Naive Bayes, the predictor variables (B and C in the equation above) are assumed to be conditionally independent of each other, given the target variable (A)
  - This is an assumption that very often is not actually true.
  - But, this assumption is made to simplify the model.
    - Naive Bayes is one of the simplest, the training time for a Naive Bayes model can sometimes be drastically lower than for other models 
- Naive Bayes
  - Pros :
  - Cons : 
