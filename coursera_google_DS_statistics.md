### Basic Concept
- Mean : the average value in a dataset.
- Median : the middle value in a dataset.
- Mode : the most frequently occurring value in a dataset.
- Range : the difference between the largest and smallest value in a dataset.
- Variance : the average of the squared difference of each data point from the mean. 
- Standard deviation : how spread out your values are from the mean of your dataset.
  - $s = \sqrt{\frac{\sum (x - \bar{x})^2}{n - 1}}$
- Percentile : the value below which a percentage of data falls. 
- Quartile divides the values in a dataset into four equal parts.
  - Q1 : 25th percentile, Q2 : 50th percentile, Q3 : 75th percentile
- Interquartile range (IQR) : Q3 - Q1
  - the distance between the first quartile (Q1) and the third quartile (Q3)

### The probability of multiple events
- Mutually exclusive events : If 2 events cannot occur at the same time.
- Independent events : If the occurrence of one event does not change the probability of the other event.
- 3 basic rules
  - Complement rule : P(A') = 1 - P(A)
    - the probability that event A does not occur is 1 minus the probability of A
  - Addition rule : P(A or B) = P(A) + P(B)
    - if events A and B are mutually exclusive
    - ex) die roll : eitehr 2 or 4
  - Multiplication rule : P(A and B) = P(A) $\times$ P(B)
    - if events A and B are independent
- Conditional Probability
  - two events are dependent if the occurrence of one event changes the probability of the other event. 
  - P(A and B) = P(A) $\times$ P(B|A)
  - P(B|A) = P(A and B) / P(A)
- Calculate conditional probability with Bayes's theorem
  - $P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$
    - P(A|B) : Posterior probability
    - P(B|A) : Likelihood
    - P(A) : Prior probability
    - P(B) : Evidence
  - Posterior = Likelihood * Prior / Evidence

### Discrete probability distributions
- Uniform Distribution
  - events whose outcomes are all equally likely, or have equal probability
- Binomial Distribution
  - the probability of events with only two possible outcomes: success or failure.
  - This definition assumes the following
    - Each event is independent, or does not affect the probability of the others.
    - Each event has the same probability of success. 
  - Use case
    - A new medication generates side effects
    - A credit card transaction is fraudulent
    - A stock price rises in value 
- Bernoulli Distribution
  - similar to the binomial distribution
    - as it also models events that have only two possible outcomes (success or failure)
  - The only difference is that the Bernoulli distribution refers to only a single trial of an experiment,
    - while the binomial refers to repeated trials.
  - A classic example of a Bernoulli trial is a single coin toss.
- Poisson Distribution
  - the probability that a certain number of events will occur during a specific time period. 
  - Use case
    - Calls per hour for a customer service call center
    - Customers per day at a shop
    - Thunderstorms per month in a city
    - Financial transactions per second at a bank

### Model data with the normal distribution









