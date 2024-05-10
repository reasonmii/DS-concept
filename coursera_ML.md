### Feature Transformation

- Log normalization : skewed data -> log-normal distribution
- Scaling
  - Normalization : MinMaxScaler in scikit-learn : $\frac{x_i - x_{min}}{x_{max} - x_{min}}$
  - Standardization : StandardScaler in scikit-learn : $\frac{x_i - x_{mean}}{x_{std}}$ 
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
  - This is an assumption that very often is not actually true. (cons)
  - But, this assumption is made to simplify the model. (pros)
    - Naive Bayes is one of the simplest
    - the training time for a Naive Bayes model can sometimes be drastically lower than for other models 
- Implementations in scikit-learn : sklearn.naive_bayes module
  - BernoulliNB : Used for binary/Boolean features
  - CategoricalNB : Used for categorical features
  - ComplementNB : Used for imbalanced datasets, often for text classification tasks
  - GaussianNB : Used for continuous features, normally distributed features
  - MultinomialNB : Used for multinomial (discrete) features

### Evaluation Metrics
- Accuracy : $\frac{True Positive + True Negative}{Total}$
- Precision : $\frac{True Positive}{True Positive + False Positive}$
  - use when it’s important to avoid false positives.
- Recall : $\frac{True Positive}{True Positive + False Negative}$
  - use when it’s important that you identify as many true responders as possible
  - ex) identifying poisonous mushrooms, it’s better to identify all of the true occurrences of poisonous mushrooms, even if that means making a few more false positive predictions.
- ROC Curves
  - X : False positive rate = $\frac{False Positive}{False Positive + True Negative}$
  - Y : True positive rate = Recall = $\frac{True Positive}{True Positive + False Negative}$
- AUC : two-dimensional area underneath an ROC curve
  - ranges in value from 0 to 1
- F1 : $2 \cdot \frac{precision \cdot recall}{precision + recall}$
  - give equal importance to both precision and recall
  - prevent one very strong factor (precision/recall) carrying the other
  - range 0 (worst) to 1 (best)
- $F_B$ score : $(1 + B^2) \cdot \frac{precision \cdot recall}{B^2 \cdot precision + recall}$
  - if you still want to capture both precision and recall in a single metric, but you consider one more important than the other
  - $B$ is a factor that represents how many times more important recall is compared to precision
