# Feature Transformation

1. Log normalization : skewed data -> log-normal distribution
2. Scaling
  - Normalization : MinMaxScaler in scikit-learn : (x_i - x_min) / (x_max - x_min)
  - Standardization : StandardScaler in scikit-learn : (x_i - x_mean) / x_std
3. Encoding : categorical data -> numerical data

# Imbalanced Datasets
1. Downsampling : make the minority class represent a larger share of the whole dataset by removing observations from the majority
2. Upsampling : when the dataset doesn't have a very large number of observations in the first place

# Bayes' Theorem
$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
