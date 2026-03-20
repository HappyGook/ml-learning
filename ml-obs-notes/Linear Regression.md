In regression problems, the inputs $x \in \mathbb{R}^D$ are mapped to the corr. function values $f(x) \in \mathbb{R}$. The inputs are set of training values $x_n$ and noisy observations $y_n = f(x_n) + \epsilon$ , where $\epsilon$ is an i.i.d. rand-var describing noise.

To solve a regression problem, the following problems must be solved:
	- **Choice of the model & parametrisation** - Which polynomial (and of what degree) passes good to model the data?
	- **Finding good parameters** - Optimise params $\theta$ based on loss functions
	- **Overfitting & model selection** 
	- **Relationship between loss-funcs & parameter priors** - Prior assumptions induce losses
	- **Uncertainty modelling**

### Problem formulation

The noise is modelled using a likelihood function. For the following examples, it will be
	$p(y|x) = \mathcal{N}(y|f(x),\sigma^2)$ 
	with $x \in \mathbb{R}^D$ - inputs and $y=f(x) + \epsilon \in \mathbb{R}$ - targets, and with $\epsilon \sim \mathcal{N}(0,\sigma^2)$ - i.i.d. noise distribution

Linear regression is given by:
	$p(y|x,\theta) = \mathcal{N}(y|x^T\theta, \sigma^2)$ 
		$p(y|x,\theta)$ is the prob-density func of $y$ evaluated at $x^T\theta$ 
	$\Leftrightarrow y=x^T\theta +\epsilon, ~ \epsilon \sim \mathcal{N}(0,\sigma^2)$ , where $\theta \in \mathbb{R}^D$ are the searched params
	

For the input $\mathcal{D} := {(x_1,y_1),...,(x_N,y_N)}$ the observations $y_i$ and $y_j$ are conditionally independent given respective inputs $x_i$ and $x_j$. Therefore the likelihood factorises:
	$p(\mathcal{Y}|\mathcal{X},\theta) = p(y_1,...,y_N|x_1,...,x_N,\theta) = \displaystyle\prod_{n=1}^N p(y_n|x_n,\theta)$
	$\Leftrightarrow \displaystyle\prod_{n=1}^N \mathcal{N}(y_n|x_n^T\theta,\sigma^2))$ 

The goal is to find the optimal parameters $\theta^*$. When these are found, function values are predicted from distribution
	$p(y_*|x_*,\theta^*)=\mathcal{N}(y_*|x_*^T\theta^*,\sigma^2)$ 

### Maximum Likelihood Estimation

The optimal params $\theta_{ML}$ are chosen by maximising the likelihood 
	$\theta_{ML} \in argmax_{\theta}~p(\mathcal{Y}|\mathcal{X},\theta)$ 

Instead of direct maximisation of the likelihood, the log transformation is applied and negative log-likelihood is minimised.
	Optimum of the $f$ is the same as the optimum of $logf$ 
	Moreover, log-transform doesn't suffer from numerical underflow and has simpler differentiation rules.

Log-transform makes the optimal function a sum of logs, instead of a product:
	$-log~p(\mathcal{Y}|\mathcal{X},\theta) = -log~\displaystyle\prod_{n=1}^N~p(y_n|x_n,\theta)$ 
	$\Leftrightarrow -\displaystyle\sum_{n=1}^N~log~p(y_n|x_n,\theta)$  

For linear regressions with Gaussian likelihood, we get
	$log~p(y_n|x_n,\theta)=-\cfrac{1}{2\sigma^2}(y_n-x_n^T\theta)^2 + const$ 
Thus, the negative log-likelihood becomes
	$\mathcal{L}(\theta)=\cfrac{1}{2\sigma^2}\displaystyle\sum_{n=1}^N~(y_n-x_n^T\theta)$
		$=\cfrac{1}{2\sigma^2} ||y-X\theta||^2$  

For this quadratic function, the minimum is searched, by computing the gradient of $\mathcal{L}$, setting it to 0 and solving for $\theta$
	$\cfrac{d\mathcal{L}}{d\theta}=...=\cfrac{1}{\sigma^2}(-y^TX+\theta^TX^TX) \in \mathbb{R}^{1\times D}$ , with $rk(X)=D$ 

Setting it to 0 and solving yields
	$\cfrac{d\mathcal{L}}{d\theta}=0^T\Leftrightarrow \theta_{ML}^TX^TX=y^TX$ 
	$\Leftrightarrow \theta_{ML}=(X^TX)^{-1}X^Ty$


For the more complex data, the data can be expressed via a non-linear transformation of the inputs $\phi(x)$, to fir within the linear regression framework (regression must be linear only in the parameters)
	$p(y|x,\theta)=\mathcal{N}(y|\phi^T(x)\theta,\sigma^2)$
	$\Leftrightarrow y=\phi^T(x)\theta + \epsilon = \displaystyle\sum_{k=0}^{K-1}~\theta_k~\phi_k~(x)+\epsilon$
	where $\phi:\mathbb{R}^D \rightarrow \mathbb{R}^K$ is a (nonlinear) transformation of the inputs $x$ and $\phi_k:\mathbb{R}^D \rightarrow \mathbb{R}$ is the kth component of the **feature vector ${\phi}$**. The model params $\theta$ still only appear linearly

For training inputs $x_n \in \mathbb{R}^D$ and targets $y_n \in \mathbb{R}, n=1,...,N$, the feature matrix is defined as:

$\begin{bmatrix} \phi^T(x_1) \\ \vdots \\ \phi^T(x_N) \end{bmatrix}$ $=$ $\begin{bmatrix} \phi_0(x_1) & \cdots & \phi_{K-1}(x_1) \\ \vdots & \ddots & \vdots \\ \phi_0(x_N) & \cdots &\phi_{K-1}(x_N) \end{bmatrix}$ $\in \mathbb{R}^{N\times K}$ 

Where $\Phi_{ij}=\phi_j(x_i)$ and $\phi_j:\mathbb{R}^D \rightarrow \mathbb{R}$ 

With the feature matrix $\Phi$, the negative log-likelihood for the linear regression has the form
	$-log~p(\mathcal{Y}|\mathcal{X},\theta)$ $=$ $\cfrac{1}{2\sigma^2}(y-\Phi \theta)^T(y-\Phi \theta) + const$ 
And the MLE
	$\theta_{ML}=(\Phi^T \Phi)^{-1}~\Phi^Ty$  <- MLE for linear regression with nonlinear features.

MLE can also be used to estimate the variance $\sigma^2_{ML}$ , for that the derivative of the log-likelihood with respect to $\sigma^2>0$ is computed, set to 0, and solved.
Log-likelihood 
$log~p(\mathcal{Y}|\mathcal{X},\theta,\sigma^2) = \displaystyle\sum_{n=1}^{N}~log~\mathcal{N}(y_n|\phi^T(x_n)\theta,\sigma^2)$ 
	$= \displaystyle\sum_{n=1}^{N}(-\cfrac{1}{2}log(2\pi)-\cfrac{1}{2}log~\sigma^2 - \cfrac{1}{2\sigma^2}(y_n-\phi^T(x_n)\theta)^2)$
	$=\cfrac{N}{2}~log~\sigma^2-\cfrac{1}{2\sigma^2}~\underbrace{\displaystyle\sum_{n=1}^{N}(y_n-\phi^T(x_n)\theta)^2}_{=:s}+const$

Then the partial derivative with respect to $\sigma^2$ is computed
	$\cfrac{\partial log~p(\mathcal{Y}|\mathcal{X},\theta,\sigma^2)}{\partial\sigma^2}$ $=$ $-\cfrac{N}{2\sigma^2}+\cfrac{1}{2\sigma^4}s$ $= 0$ 
	$\Leftrightarrow \cfrac{N}{2\sigma^2}=\cfrac{s}{2\sigma^4}$ 

And then, the most likely noise variance $\sigma_{ML}^2$
	$\sigma_{ML}^2 = \cfrac{s}{N}=\cfrac{1}{N}\displaystyle\sum_{n=1}^N~(y_n-\phi^T(x_n)\theta)^2$ 
	so, the MLE of the noise variance is the empirical mean of the distance between the noisy observations $y_n$ and corresponding function values $\phi^T(x_n)\theta$ 
