Predictors reason in a way of **abduction** - with regularization or by adding a prior, the simpler explanations of complex phenomenons are found. 
## Predictors as functions
Function-predictor takes an input of D-dim vector x and returns prediction:
	 $f(x)=\theta*x^2 + \theta_0$ <- For linear funcs only
This is like a value head, goal is to minimise error.

Predictor-functions minimise risk empirically.
For ($(x_0,y_0),...,(x_n,y_n)$) the $f(\cdot, \theta): R^D \rightarrow R$ is estimated for $\theta$, so that for $\forall y_n \in R : f(x_n, \theta^{*})\approx y_n$ (The actual output of a predictor is notated $\hat y_n$)

**If $y_n \in R$ , then $f(x_n, \theta)$ can be shown as a linear func:**
- actually as an affine function, but is referred to as linear
$x_n = [1, x_n^{(1)}, x_n^{(2)}, ... , x_n^{(D)}]^T$ <- Additional unit features $x_n^{(0)} = 1$ are concatenated

$\theta = [\theta_0, \theta_1, \theta_2, ... ,\theta_D]^T$ and 
--> $f(x_n,\theta)=\theta^T x_n$ <--

$\Leftrightarrow f(x_n,\theta)=\theta_0 + \displaystyle\sum_{d=1}^D \theta_d   x_n^{(d)}$ ( Here $f : R^{D+1} \rightarrow R$ )
The function before is affine.
### Loss function for training
To define whether prediction $\hat y_n$ based on $x_n$ fits the data well, loss function is needed: $\ell(y_n, \hat y_n)$. It compares the prediction and actual feature and outputs a non-negative loss. Goal then is to find a good param-vector $\theta^{*}$ to minimise the loss.

It is commonly assumed that the set of examples $(x_i,y_i),...,(x_N,y_N)$ is independent and identically distributed (i.i.d), so no 2 datapoints are statistically dependent, and empirical mean is a viable estimate of the population mean.

For a training set $\{ (x_1,y_1),...,(x_N,y_N) \}$ 
$X:=[x_1,...,x_N]^T \in R^{N\times D}$ is an example matrix
$y:=[y_1,...,y_N]^T\in R^N$ is a label vector
The average loss is given by:
$R_{emp}(f,X,y)=\cfrac{1}{N}\displaystyle\sum_{n=1}^N \ell(y_n, \hat y_n)$ <- this is the **empirical risk**

To find the predictor that minimises the *expected risk*, the population (true) risk is computed as the expectation of the loss:
$R_{true}(f)=E_{x,y} [\ell(y_n, f(x))]$ <- this is the true risk if we have infinite data
### Regularization (Reduce Overfitting)
Part of the dataset is not used in training to be used later as unseen data to test predictions and to reduce overfitting. (test set)

Overfitting tends to occur with small datasets and complex classes.
Overfitting is happening when $R_{emp}(f,X_{train},y_{train})$ underestimates $R_{true}(f)$ = $R_{emp}(f,X_{test},y_{test})$, and so if $R_{true} > R_{emp}$ considerably, this is a sign of overfitting.

Regularization is a penalty term for the minimiser, that makes it harder for the optimiser to become overly flexible.
### Cross-Validation 
One of the problems with ml is the "large training set, large validation set" problem. We want to keep as much data as possible for training, but we also want to keep the validation set $V$ as big as possible, to lower the variance of prediction. 

K-fold cross-validation splits the data into K chunks, $K-1 \in R$ (training set) and last chunks serves as the validation set $V$. Ideally cross-val iterates through all combos of chunks for $R$ and $V$. 

So, the dataset $D = R \lor V$, while $(R \land V) = \emptyset$ .
We train on $R$, assess the predictor $f$ on $V$:
For each partition $k$ the training set $R^{(k)}$ produces a predictor $f^{(k)}$, which is applied to $V^{(k)}$ and thus empirical risk $R(f^{(k)},V^{(k)})$ is computed. After going through all possible partitionings of $V$ and $R$, the average is put as a generalization error:
	$E_V[R(f,V)] \approx \cfrac{1}{K}\displaystyle\sum_{k=1}^K R(f^{(k)},V^{(k)})$

For instance, with K=5, there are 5 possible partitionings. The computing cost increases.

## Predictors as probabilistic models
The goal is to find the function of the parameters that matches the distribution of the data. 

For data represented by rand-var $x$ and for family of prob-densities $p(x| \theta)$ parametrised by $\theta$: $\mathcal{L}_x(\theta) = -log p(x|\theta)$ 
$\mathcal{L}_x(\theta)$ is the negative log-likelihood. It is the function of $\theta$ and the data $x$ is seen as fixed for it. 
$\mathcal{L}_x(\theta)$ tells how likely is the setting $\theta$ for the observations $x$. 
> The likelihood $p(x|\theta)$ measures how probable the observed data is under parameter $\theta$. When viewed as a function of $\theta$, it is called the likelihood function.

**Maximum Likelihood Estimation** (MLE) maximises the likelihood by finding the most likely parameter setting $\theta$

For example, if observation corresponds to the Gaussian with zero mean $\epsilon_n \sim \mathcal{N}(0,\sigma^2)$ . So, for each label pair $(x_n, y_n)$ the Gaussian likelihood looks like:
	$p(y_n|x_n,\theta)=\mathcal{N}(y_n|x_n^T \theta, \sigma^2)$ 

If the set of examples $(x_1,y_1),...,(x_N,y_N)$ is **i.i.d**, the likelihood involving the whole dataset of $\mathcal{X}=({x_1,...,x_N})$ and $\mathcal{Y}=(y_1,...,y_N)$ can be factorised into a product of likelihoods of each example:
	$p(\mathcal{Y}|\mathcal{X},\theta) = \displaystyle\Pi_{n=1}^N p(y_n|x_n,\theta)$ 

For i.i.d datasets, the negative log-likelihood can be decomposed:
	$\mathcal{L}_x(\theta) = -log p(\mathcal{Y}|\mathcal{X},\theta) = - \displaystyle\sum_{n=1}^N log p(y_n|x_n,\theta)$ 
Then, the best setting is found by minimising $\mathcal{L}(\theta)$ with respect to $\theta$.

$\theta_{\text{MLE}} = \arg\min_\theta [-\log p(x|\theta)]$
### Maximising the posterior (MAP)
The prior knowledge is the distribution $p(\theta)$ of the parameters, and we observe the data $x$ (so margin is $p(x)$). To represent how we need to update the distribution of $\theta$ given new observations, we see the corrected $p(\theta|x)$ as a posterior:
	$p(\theta|x)=\cfrac{p(x|\theta) p(\theta)}{p(x)}$ , where $p(x|\theta)$ shows how likely $x$ is given the current $\theta$ 

Since $\theta$ doesn't depend on $p(x)$, it can be removed, so that 
	$p(\theta|x) ∝ p(x|\theta) p(\theta)$ , so we maximise it.

$\theta_{\text{MAP}} = \arg\min_\theta \left[-\log p(x|\theta) \log p(\theta)\right]$

Both of these methods return a single parameter vector, a point estimate. So, while the resulting $p(\theta|x)$ is a probability distribution, we lose uncertainty over parameters.
### Model Fitting

In the process of learning, the model $M_{\theta}$ is optimised to be as close as possible to unseen model $M^*$ that describes the data.

In this process, **overfitting** happens, when $M_{\theta}$ is too rich for the dataset, and could model more complicated datasets. (e.g. $M^*$ is linear, and $M_{\theta}$ is a polynome $ax^3+bx^2+cx+d$ ) Overfitted model fits its parameters $\theta$ to reduce the training error, and therefore to reduce the noise. Because model is overcomplicated, it fits the params to the noise, and therefore works badly on real data.

On the opposite, **underfitting** happens, when the $M_{\theta}$ is not rich enough for the dataset (e.g $M^*$ is $ax^2+bx+c$ and $M_{\theta}$ is $ax+b$)

For **fitting** to happen, the model class must have about the same complexity as the dataset.


## Inference

**Probabilistic modelling** can be used to learn something about the un-observed distribution from the observed outcomes. E.g. to understand the unseen $p(y)$ from the dataset $\mathcal{X}$, which is formed by $p(x|y)$.

A probabilistic model is specified by the joint distribution of all random variables. The joint distribution $p(x,\theta)$ of the observed $x$ and hidden params $\theta$ encapsulates the information from the prior, likelihood, marginal likelihood $p(x)$, and the posterior (obtained by dividing the joint by the marginal likelihood).