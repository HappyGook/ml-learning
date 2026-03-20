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

By optimising for the posterior $p(x|\theta)$, the information about the distribution of $\theta$ gets lost. Therefore, to get the full posterior distribution (so not the maximum, and rather the whole distribution), the Bayesian inference is used:
	$p(\mathbb{\theta}|\mathcal{X})=\cfrac{p(\mathcal{X}|\mathbb{\theta}) p(\mathbb{\theta})}{p(\mathcal{X})}$  , where $p(\mathcal{X})=\int p(\mathcal{X}|\mathbb{\theta}) p(\mathbb{\theta}) d\theta$  
The Bayesian theorem is used to invert the relation between $\theta$ and $\mathcal{X}$, to obtain the posterior distribution.

By finding posterior and $p(\theta)$, the uncertainty can be propagated from the parameters to the data, and predictors get the form:
	$p(x)=\int p(x|\mathbb{\theta}) p(\mathbb{\theta}) d\theta = \mathbb{E}_{\theta}[p(x|\theta)]$  
Here, the prediction is the average over all plausible parameter values $\theta$ (plausibility determined by distribution $p(\theta)$)

## Latent variables

Latent variables $z$ are added to the model, while not directly parametrizing it like $\theta$. They can help model be more interpretable, simplify its structure without precision losses, but generally make learning harder.

For data $x$, model params $\theta$ and latent variables $z$, the conditional distribution looks like
	$p(x|z,\theta)$ 

In such model, to find the predictive function based on parameters, latent variables need to be marginalised via putting a prior on them ( $p(z)$ ) and integrating:
	$p(x|\theta)=\int p(x|z,\mathbb{\theta}) p(z) dz$ <--- as stated, likelihood must not depend on $z$

The posterior distribution is then found the same way with Bayes' 

Similar to the posterior on parameters $\theta$, the posterior on $z$ is computed via Bayes'
	$p(z|\mathcal{X})=\cfrac{p(\mathcal{X}|z) p(z)}{p(\mathcal{X})}$,    $p(\mathcal{X}|z)=\int p(\mathcal{X}|z,\mathbb{\theta}) p(\mathbb{\theta}) d\theta$ 

Because of the computing difficulty, both model params and latent vars can't be both marginalised at the same time. Easier computed is the posterior distribution on the latent vars $z$, conditioned by params $\theta$ :
	$p(z|\mathcal{X}, \theta)=\cfrac{p(\mathcal{X}|z,\theta) p(z)}{p(\mathcal{X}|\theta)}$ 

**Some more context to the usage**: latent variables allow us to simplify the model by assuming: "conditioned on the situation $z$, the model is simple". By this the parameters don't have to be overcomplicated to explain the data.

Diagram as an example
![[latent_vars.svg|510]]
## Graphs to visualise dependencies

The joint distribution $p(a,b,c)$ doesn't tell anything about the independence relations of distributions. A way to show this is to construct a directed graph.

E.g. for the joint distribution $p(a,b,c)=p(c|a,b) p(b|a) p(a)$ the graph would look like
	$(a) \longrightarrow (b)$ 
	   $\searrow$     $\swarrow$
		 $(c)$ 
Is also fully connected

The joint distribution can also be extracted from the graph, like:
	$p(x) = \displaystyle\Pi_{k=1}^K p(x_k|Pa_k)$, where $Pa_k$ means the parent nodes of $x_k$, i.e. the nodes that have arrows pointing to $x_k$ 

Conditional independence and d-Separation can also be portrayed via directed graphs.
There are 3 main types of probabilistic graphical models: 
- Directed graphical models (Bayesian networks)
- Undirected graphical models (Markov random fields)
- Factor graphs


## Model Selection

Model selection presents mechanisms to assert how a model generalises to unseen data.

If for each split in cross-validation, it is again split and cross-validated, the process is called a **nested cross-validation.** 

The inner level in it is used to estimate a performance of a particular choice of model or hyperparameter on the internal validation set. The outer layer estimates the generalisation performance of the best choice of model (chosen in inner loop). 

The set used to estimate the generalisation performance is called the **test set**, and the one used to choose the best model is called the **validation set**.

The model is chosen in inner loop based on the expected value of the generalisation error $R(\mathcal{V}|M)$ for it. The error $R(\mathcal{V}|M)$ is approximated with the empirical error on the validation set $\mathcal{V}$ for model $M$:
	$\mathbb{E}_{\mathcal{V}} R(\mathcal{V}|M) \approx \cfrac{1}{K} \displaystyle\sum_{k=1}^{K}R(\mathcal{V}^{(k)}|M)$   

The goal is always to find the simplest model that explain the data well (Occam's razor) <- Can also be explained as finding the simplest hypothesis that is consistent.

**Bayesian model selection** for a set of models $M=\lbrace M_1, ..., M_K\rbrace$, where each model $M_K$ has params $\theta$ , goes as follows:
1. The prior is placed on the set of models $p(M)$ - model-distro becomes a random variable with each $p(M_k)$ showing how well the model explains the data
2. The data is generated from the model
	$M_k \sim p(M)$ <-- uncertainty of model
	$\downarrow$
	$\theta_k \sim p(\theta|M_k)$ <-- uncertainty of parameters within the model
	$\downarrow$
	$\mathcal{D} \sim p(\mathcal{D}|\theta_k)$ <-- Data-generating distro under $M$ and $\theta$ 
	
	>The full joint distribution looks as $p(\mathcal{D}, \theta, M) = p(M)\, p(\theta \mid M)\, p(\mathcal{D} \mid \theta, M)$

Given a training set $\mathcal{D}$, the Bayes' is applied to compute the posterior over models:
	$p(M_k|\mathcal{D}) ∝ p(M_k)p(\mathcal{D}|M_k)$ , this posterior is independent of $\theta$, which was marginalised in the following marginal likelihood $\downarrow$
	
	$p(\mathcal{D}|M_k) = \int p(\mathcal{D}|\theta_k)p(\theta_k|M_k)d\theta_k$
	where $p(\theta_k|M_k)$ is the prior distro of parameters $\theta_k$ of the model $M_k$ 
	Marginal likelihood automatically embodies a trade-off between the complexity of model and data fit.

Then, from the posterior $p(M_k|\mathcal{D})$ the MAP is estimated as
	$M^*=argmax_{M_k} p(M_k|\mathcal{D})$ 

Two probabilistic models can be compared with **Bayes Factors**. For this, the ratio of their posteriors is computed
	$\underbrace{\cfrac{p(M_1|\mathcal{D})}{p(M_2|\mathcal{D})}}_{posterior~odds} = \underbrace{\cfrac{p(M_1)}{p(M_2)} }_{prior~odds} ~\underbrace{\cfrac{p(\mathcal{D}|M_1)}{p(\mathcal{D}|M_2)}}_{Bayes factor}$ 

Based on the posterior odds, a model is chosen.