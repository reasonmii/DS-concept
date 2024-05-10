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

### Clustering
- K-means
  - Process
    - Randomly place centroids in the data space.
    - Assign each point to its nearest centroid.
    - Update the location of each centroid to the mean position of all the points assigned to it.
    - Repeat steps 2 and 3 until the model converges (i.e., all centroid locations remain unchanged with successive iterations).
  - K-means works by minimizing intercluster variance
    - it aims to minimize the distance between points and their centroids
    - it works best when the clusters are round
    - If you aren’t satisfied with the way K-means is clustering your data, you need other clustering methods
  - Inertia : measurement of intracluster distance = $\sum_{i=1}^n (x_i - C_k)^2$
    - n : the number of observations in the data
    - $x_i$ : the location of a particular observation
    - $C_k$ : the location of the centroid of cluster $k$, which is the cluster to which point $x_i$ is assigned
    - The greater the inertia, the greater the distances between points and their centroids
    - Inertia is a useful metric to determine how well your clustering model identifies meaningful patterns in the data.
      - But it’s generally not very useful by itself.
      - If your model has an inertia of 53.25, is that good? It depends.
  - The elbow method
    - Compare different values of k
    - Help decide which clustering gives the most meaningful model of your data
  - Silhouette analysis : comparison of different models’ silhouette scores
    - Silhouette coefficient = $\frac{(b - a)}{max(a, b)}$
      - $a$ : the mean distance between the instance and each other instance in the same cluster
      - $b$ : the mean distance from the instance to each instance in the nearest other cluster (i.e., excluding the cluster that the instance is assigned to)
      - $max(a,b)$ : which value is greater
      - between -1 (bad) to +1 (good)
        - +1 : point is close to other points in its own cluster and well separted from points in other clusters
    - Silhouette score is the mean silhouette coefficient over all the observations in a model
      - The greater the silhouette score, the better defined the model clusters
- K-means++
  - still randomly initializes centroids in the data, but it does so based on a probability calibration
  - it randomly chooses one point within the data to be the first centroid, then it uses other data points as centroids, selecting them pseudo-randomly.
  - The probability that a point will be selected as a centroid increases the farther it is from other centroids.
  - This helps to ensure that centroids aren’t initially placed very close together, which is when convergence in local minima is most likely to occur.
  - K-means++ is the default implementation when you instantiate K-means in scikit-learn.
- DBSCAN : density-based spatial clustering of applications with noise
  - Find clusters based on density, the shape of the cluster isn’t as important as it is for K-means
    - Instead of trying to minimize variance between points in each cluster,
    - it searches your data space for continuous regions of high density
  - Process
    - Start at random point
    - Examine radius and around the point
    - If there are min_samples within radius, $\epsilon$ of this instance (including itself), all samples in this neighborhood become part of the same cluster
    - Repeat
  - Hyperparameters
    - eps: Epsilon $\epsilon$ - The radius of your search area from any given point
    - min_samples: The number of samples in an ε-neighborhood for a point to be considered a core point (including itself)
- Agglomerative clustering
  - First assigning every point to its own cluster, then progressively combining clusters based on intercluster distance.
  - It requires that you specify a desired number of clusters or a distance threshold, which is the linkage distance.
  - If you do not specify a desired number of clusters, then the distance threshold is an important parameter,
    - because without it the model would converge into a single cluster every time.
  - When does it stop?
    - You reach a specified number of clusters.
    - You reach an intercluster distance threshold (clusters that are separated by more than this distance are too far from each other and will not be merged).
  - Hyperparameters
    - n_clusters: The number of clusters you want in your final model
    - linkage: The linkage method to use to determine which clusters to merge (as described above)
    - affinity: The metric used to calculate the distance between clusters. Default = euclidean distance.
    - distance_threshold: The distance above which clusters will not be merged (as described above)
- Linkage : measure the distances that determine whether or not to merge the clusters
  - Single: The minimum pairwise distance between clusters
  - Complete: The maximum pairwise distance between clusters
  - Average: The distance between each cluster’s centroid and other clusters’ centroids.
  - Ward: This is not a distance measurement. Instead, it merges the two clusters whose merging will result in the lowest inertia.

### Decision Tree
- Pros
  - Require relatively few pre-processing steps
  - Can work easily with all types of variables (continuous, categorical, discrete)
  - Do not require normalization or scaling
  - Decisions are transparent
  - Not affected by extreme univariate values
- Cons
  - Can be computationally expensive relative to other algorithms
  - Small changes in data can result in significant changes in predictions
- Choosing Splits : Gini impurity, entropy, information gain, log loss
- Gini impurity : $1 - \sum_{i=1}^N P(i)^2$
  - $P(i)$ = the probability of samples belonging to class i in a given node.
  - ex) $1 - P(apple)^2 - P(grape)^2$
- Hyperparameters
  - max_depth : how deep the tree is allowed to grow
  - min_samples_split :  the minimum number of samples that a node must have for it to split into more nodes
  - min_samples_leaf : the minimum number of samples that must be in each **child** node after the parent splits
- Grid search : Finding the optimal set of hyperparameters
```
rf = RandomForestClassifier(random_state=0)

cv_params = {'max_depth': [2,3,4,5, None], 
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'max_features': [2,3,4],
             'n_estimators': [75, 100, 125, 150]
             }  
scoring = {'accuracy', 'precision', 'recall', 'f1'}

rf_cv = GridSearchCV(estimator=rf, param_grid=cv_params, scoring=scoring, cv=5, refit='f1')

rf_cv.fit(X_train, y_train)
```
- Bagging : bootstrap aggregating
  - bootstrapping refers to sampling with replacement
  - Why to use it
    - Reduces variance: Standalone models can result in high variance. Aggregating base models’ predictions in an ensemble help reduce it.
    - Fast: Training can happen in parallel across CPU cores and even across different servers.
    - Good for big data
      - Bagging doesn’t require an entire training dataset to be stored in memory during model training.
      - You can set the sample size for each bootstrap to a fraction of the overall data, train a base learner, and string these base learners together without ever reading in the entire dataset all at once. 
- Random Forest : Bagging + random feature sampling
  - It randomizes **samples** and the data by **features**
  - This randomization from sampling often leads to both better performance scores and faster execution times,
    - making random forest a powerful and relatively simple tool in the hands of any data professional.
  - sklearn.ensemble : RandomForestRegressor, RandomForestClassifier
- Gradient Boosting : uses an ensemble of weak learners to make a final prediction.
  - One of the most powerful supervised learning techniques
  - It works by building an ensemble of decision tree base learners
    - wherein each base learner is trained successively,
    - attempts to predict the error—also known as “residual”—of the previous tree,
    - and therefore compensate for it.
  - Its base learner trees are known as “weak learners” or “decision stumps.”
    - They are generally very shallow. 
  - More resilient to high variance that results from overfitting the data due to being comprised of high-bias, low-variance weak learners.
  - xgboost : XGBClassifier, XGBRegressor



