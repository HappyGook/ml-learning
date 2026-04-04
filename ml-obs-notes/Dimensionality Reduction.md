High-dimensional data is difficult to interpret, work with and store. Therefore, the amount of dimensions should be reduced. Dimensionality reduction exploits structure and correlation and ideally represents data more compactly without losing information. (So it is a compression technique)

The known algorithm for it is the **Principal Component Analysis** (PCA)
In PCA, the projections $\tilde x_n$ of data points $x_n$ are found, which are as close as possible, but have lower dimensionality.

For an i.i.d. dataset $\mathcal X= \lbrace x_1,...,x_N \rbrace x_n ∈\mathbb R^D$ with mean 0 that's data covariance matrix is defined as
	$S= \cfrac{1}{N} \displaystyle\sum_{n=1}^N~x_nx_n^⊤$ ^8542b8

It is also assumed that the data has a low-dimensional compressed representation of $x_n$
	$z_n= B^⊤x_n ∈ \mathbb R^M$
	The projection matrix $B$ is defined as $B:= [b_1,...,b_M] ∈ \mathbb R^{D×M}$ 
	assumed columns of $B$ are orthonormal

The goal is to find an $M$-dim subspace $U \subseteq \mathbb R^D$ with $dim(U)=M<D$ onto which the data is projected.   
The columns $b_1,...,b_M$ of $B$ for a basis of the subspace $U$. 
	$\tilde x=BB^Tx \in \mathbb R^D$

The scheme of PCA:
	$\underbrace{[x \in \mathbb R^D]}_{original}$ $\rightarrow$ $\underbrace{[z \in \mathbb R^M]}_{compressed}$ $\rightarrow$ $\underbrace{[\tilde x \in \mathbb R^D]}_{reconstructed}$ 

## Maximum Variance Perspective

Variance of the low-dimensional code does not depend on the mean of the data.
	$\mathbb V_z[z]=\mathbb V_x[B^T(x-\mu)] = \mathbb V_x[B^Tx]$ 
Therefore, it can be assumed w/o loss that mean of data is 0. This also makes the mean of the code 0
	$\mathbb E_z[z] = \mathbb E_x[B^Tx]=B^T \mathbb E_x[x]=0$ 

PCA determines which dimensions are more important by their variance (since variance captures data spread), therefore PCA finds a low-dim projection that maintains as much variance as possible.

### Approach

The maximisation is led sequentially. For the first dimension, the vector $b_1 \in \mathbb R^D$ is found (first column of $B$ & first of $M$ orthonormal vectors), that maximises the variance of projected data in this dimension (variance of $z_1$ of $z \in \mathbb R^M$) so that
	$V_1 :=\mathbb V[z_1]=\cfrac{1}{N}\displaystyle\sum_{n=1}^N~z_{1n}^2$ 
	
With first component $z_{1n}$ of $z_n$ defined as $z_{1n} = b_1^⊤x_n$  
	$V_1 =\cfrac{1}{N}\displaystyle\sum_{n=1}^N~(b^⊤_1 x_n)^2$ $=$ $\cfrac{1}{N}\displaystyle\sum_{n=1}^N~b^⊤_1 x_n~x_n^⊤b_1$ 
	$=$  $b^⊤_1(\cfrac{1}{N}\displaystyle\sum_{n=1}^N~x_n~x_n^⊤)b_1$ $=$ $b_1^⊤Sb_1$ 
	Where $S$ is the Covariance matrix [[#^8542b8]] 

The $b_1$ is restricted with $||b_1||^2 = 1$, because $V_1$ features a square of $b_1$ and a change of $b_1$'s magnitude changes $V_1$'s magnitude even stronger.

So, the constraint optimisation problem is defined as
	$max_{b_1}~b_1^⊤Sb_1$ 
	$subject~to~~||b_1||^2=1$ 
	
The Lagrangian for it is
	$\mathcal L (b_1,\lambda) = b_1^⊤Sb_1 + \lambda_1(1-b_1^⊤b_1)$   
	
Then, derivatives are found
	$\cfrac{\partial \mathcal L}{\partial b_1} =2b_1^⊤S - 2\lambda_1b_1^⊤$ , $~~~~$ $\cfrac{\partial \mathcal L}{\partial \lambda_1} =1 - b_1^⊤b_1$
	
Setting them to 0 yields relations
	$Sb_1=\lambda_1b_1$
	$b_1^⊤b_1=1$ 
	
Comparing this relation with eigenvalue decomposition, $b_1$ is an eigenvector of $S$ and $\lambda_1$ is an eigenvalue. Therefore, variance can be rewritten as
	$V_1=b_1^⊤Sb_1=\lambda_1b_1^⊤b_1=\lambda_1$ 
	The variance of the data projected onto a one-dim subspace = the eigenvalue associated with basis vector $b_1$ that spans the subspace.

To maximise the variance of low-dim code, the basis vector associated with the largest eigenvalue of $S$ is chosen. This eigenvector is called first **principal component**. Its effect in the original data space can be determined by mapping $z_{1n}$ back into data space
	$\tilde x_n = b_1 z_{1n} = b_1b_1^⊤x_n \in \mathbb R^D$ 
	$\tilde x_n \in \mathbb R^D$ requires only single coordinate $z_{1n}$ to represent it with respect to $b_1 \in \mathbb R^D$ 

Eigenvectors can be used to construct the orthonormal eigenbasis of an ($m-1$)-dimensional subspace of $\mathbb R^D$ (spectral theorem). 
The $m$th principal component can be found by substracting effect of first $m-1$ p.c-s $b_1,...,b_{m-1}$ from the data, i.e. trying to find those p.c.-s that compress the remaining information. The new data matrix in this case is defined as
	$\hat X:=X-\displaystyle\sum_{i=1}^{m-1}~b_ib_i^⊤X = X - B_{m-1}X$
	with $X=[x_1,..,x_N] \in \mathbb R^{D \times N}$   containing data points as column vectors
	and $B_{m-1} :=\displaystyle\sum_{i=1}^{m-1}b_ib_i^⊤$ - projection matrix onto subspace spanned by $b_1,...,b_{m-1}$ 
	and $\hat X:=[\hat x_1,...,\hat x_N] \in \mathbb R^{D \times N}$ containing information that has not yet been compressed

The $m$th principal component is found via maximising the variance
	$V_m =\mathbb V[z_m] = \cfrac{1}{N} \displaystyle\sum_{n=1}^N~z_{mn}^2$  $=$ $\cfrac{1}{N} \displaystyle\sum_{n=1}^N~(b_m^⊤ \hat x_n)^2$ $=$ $b_m^⊤ \hat S b_m$ 
	as before, constrained optimisation problem is solved, with solution $b_m$ being an eigenvector of $\hat S$ with the largest eigenvalue.

Eigenvector sets of $S$ and $\hat S$ are identical, so $b_m$ is also an eigenvector of $S$. Both $S$ and $\hat S$ are symmetrical and have $D$ distinct eigenvectors. Every eigenvector of $S$ is an eigenvector of $\hat S$ 
	$\hat S b_m = S b_m = \lambda_m b_m$
	$\lambda_m$ is the largest eigenvalue of $\hat S$ and the $m$th largest of $S$, with both having an associated eigenvector $b_m$. 

Every eigenvector of $S$ is the eigenvector of $\hat S$, but if the eigenvectors of $S$ are a part of $(m-1)$ dim principal subspace, the associated eigenvalue of $\hat S$ is 0 (since $\hat S$ shows the infos outside of the subspace)

With $b_m^⊤b_m=1$ the variance of the data projected onto $m$th principal component is
	$V_m = b_m^⊤Sb_m=\lambda_mb_m^⊤b_m=\lambda_m$ 
	Variance of projected data (onto $M$-dim subspace) is the sum of eigenvalues associated  with cor. eigenvectors of the $S$ 

First $M$ principal components capture the $V_M$ amount of variance
	$V_M=\displaystyle\sum_{m=1}^{M}\lambda_m$

The variance lost by data compression via PCA is
	$J_M :=\displaystyle\sum_{j=M+1}^{D}\lambda_j=V_D-V_M$ 

Relative variance captured as $\cfrac{V_M}{V_D}$ and relative variance lost by compression as $1 - \cfrac{V_M}{V_D}$ 

## Projection Perspective

Here, PCA is seen as an algorithm that directly minimises the average reconstruction error (minimises distance between $x_n$ and $\hat x_n$)

For an orthonormal basis $b_1,...,b_D$ of $\mathbb R^D$ any $x \in \mathbb R^D$ can be written as a linear combo of the basis vectors of $\mathbb R^D$ 
	$x=\displaystyle\sum_{d=1}^D~\zeta_db_d$ $=$$\displaystyle\sum_{m=1}^M~\zeta_mb_m + \displaystyle\sum_{j=M+1}^D~\zeta_jb_j$ 
	with $\zeta_d \in \mathbb R$ being suitable coordinates for the linear combo.

Vectors $\tilde x \in \mathbb R^D$ which exist in $U \subseteq \mathbb R^D$ with dim($U$)=$M$ are defined as
	$\tilde x=\displaystyle\sum_{m=1}^M~z_mb_m \in U \subseteq \mathbb R^D$ 

The goal is to find the best linear projection of the dataset $\mathcal X = \lbrace x_1, . . . , x_N \rbrace$ onto $U$ with orthonormal basis vectors $b_1,...,b_M$. $U$ is called the **principal subspace**.
Projections are denoted by
	$\tilde x:=\displaystyle\sum_{m=1}^M~z_{mn}b_m = Bz_n\in \mathbb R^D$ 
	$z_n := [z_{1n}, . . . , z_{Mn}]^⊤ ∈ \mathbb R^M$ is the coordinate vector of $\tilde x_n$ with respect to basis $b_1,...,b_M$ 

The similarity measure between $\tilde x_n$ and $x_n$ is the Euclidean norm $||x-\tilde x||^2$ . The objective is to minimise the average Euclidean distance (reconstruction error)
	$J_M:=\cfrac{1}{N} \displaystyle\sum_{n=1}^N ||x-\tilde x_n||^2$ 

To find the coordinates $z_n$ and the ONB of $U$
	1: optimise $z_n$ for a given ONB ($b_1,...,b_M$)
	2: find optimal ONB

The optimal coordinates $z_{1n},...,z_{Mn}$ are those that make the projection orthogonal.
	*partial derivatives are found ($J_M / z_{in}$, $J_M / \tilde x_n$, $\tilde x_n / z_{in}$) with chain rule, computed, etc. This yields optimal coordinates*
	$z_{in} = x^⊤_n b_i = b^⊤_i x_n$ for $i=1,...,M$ and $n=1,...,N$ 

So, the orthogonal projection of $x$ onto the $M$-dim subspace is
	$\tilde x = B (\underbrace{B^⊤B}_{=I})^{-1}B^⊤x = BB^⊤x$ 
	with $B:=[b_1,...,b_M] \in \mathbb R^{D \times M}$ 
	Coordinates of the projection are $z:=B^⊤x$ 

### Basis of the Principal Subspace

The basis vectors $b_1,...,b_m$ can be found via rephrasing the loss function $J_M$ 

	$\tilde x_n = \displaystyle\sum_{m=1}^M~z_{mn}b_m$$=$$\displaystyle\sum_{m=1}^M~(x_n^⊤b_m)b_m$
	
	$\tilde x_n = (\displaystyle\sum_{m=1}^M~b_mb_m^⊤)x_n$

The original vector can be written ac the linear combo of all basis vectors
	$x_n=(\displaystyle\sum_{m=1}^M~b_mb_m^⊤)x_n$ $+$$(\displaystyle\sum_{j=M+1}^D~b_jb_j^⊤)x_n$ 

From that the difference vector $x_n - \tilde x_n$ can be defined
	$x_n - \tilde x_n=(\displaystyle\sum_{j=M+1}^D~b_jb_j^⊤)x_n$ 
	$= \displaystyle\sum_{j=M+1}^D~(x_n^⊤b_j)b_j$ 
	Difference is the projection onto the orthogonal complement 

The average squared reconstruction error of the projection onto M-dim principal subspace is
	$J_M=\displaystyle\sum_{j=M+1}^D~\lambda_j$ 
	with $\lambda_j$ being the eigenvalues of the data covar matrix.
To minimise this, the smallest $D-M$ eigenvalues are chosen.

PCA is not feasible in high dimensions, since it scales cubically in $D$. 
For a centered dataset $x_1,...,x_n \in \mathbb R^D$ covar matrix is given as
	$S = \cfrac{1}{N}~XX^⊤ \in \mathbb R ^{D \times D}$ 
	$X = [x_1,...,x_n] \in \mathbb R^{D \times N}$ 

Assuming that number of data points is notably smaller than the dimensionality of the data $N << D$, the $rk(S) = N$ if no duplicate data points exist, so it has $D-N+1$ null eigenvalues, which implies redundancies.

With eigenvector equation
	$Sb_m=\lambda_mb_m, ~~ m=1,...,M$ 
	$b_m$ is the eigenvector of the principal subspace.

$S$ can be substituted, so that
	$Sb_m = \cfrac{1}{N}XX^⊤b_m=\lambda_mb_m$ 

With $c_m:=X^⊤b_m$ 
	$\cfrac{1}{N}X^⊤Xc_m=\lambda_mc_m$ 

This implies that $\cfrac{1}{N}X^⊤X \in \mathbb R^{N \times N}$ has the same eigenvalues as $S$ and is symmetric, but smaller ($N << D$) This allows a more effective computation.

To recover original eigenvectors for the PCA, the equation is left-multiplied by $X$:
	$\underbrace{\cfrac{1}{N}X^⊤X}_SXc_m=\lambda_mXc_m$ 
	$Xc_m$ is recovered as an eigenvector of $S$. 

## PCA Process 

To project the given dataset (e.g. two-dimensional) onto a lower dim. subspace (e.g. one-dimensional), the following steps are taken
	1. **Mean subtraction**: Data is centered by computing the mean $\mu$ of the dataset and subtracting it from every data point.
	2. **Standardisation**: Data points are divided by the standard deviation $\sigma_d$ of the dataset for every dimension $d=1,...,D$. Now data is unit free and has $\mathbb V=1$ along each axis
	3. **Eigendecomposition of the covariance matrix**: Covar matrix of the data is computed, along with its eigenvalues and corresponding eigenvectors. Then an ONB of eigenvectors is found.
	4. **Projection**: All data points $x_* \in \mathbb R^D$ are projected onto the principal subspace. For it, $x_*$ is standardised using the $\mu_d$ and $\sigma_d$ of the training data in the $d$th dimension
		$x_*^{(d)}\leftarrow \cfrac{x_*^{(d)}-\mu_d}{\sigma_d}, ~~~ d=1,...,D$ 
		$x_*^{(d)}$ is the $d$th component of $x_*$.
	Obtained projection is $\tilde x_*=BB^⊤x_*$ with coordinates $z_*=B^⊤x_*$ . $B$ is the matrix that contains the eigenvectors with the largest eigenvalues of $S$ (as columns).
	To get the projections in the original data space, the standardisation needs to be undone
		$\tilde x^{(d)}_* ← \tilde x^{(d)}_* σ_d + μ_d, ~~~ d = 1, . . . , D$ 

## PCA as a Probabilistic Model (Latent Var Perspective)
> A latent variable is a variable that cannot be directly observed or measured but can be inferred from other observable variables.

By viewing PCA as a probabilistic model we have more flexibility and useful insights
	- Likelihood function allows to deal with noisy observations explicitly
	- Bayesian model comparison via marginal likelihood
	- View PCA as a generative model (to simulate new data)
	- Deal with random missing dimensions by Bayes'
	- Principled way to extend the model (to a mixture of PCAs)
	- Marginalise out model parameters

We assume a continuous latent variable $z \in \mathbb R^M$ with standard-normal prior $p(z)=\mathcal N(0,I)$ and a linear relation between $z$ and observed $x$
	$x=Bz+\mu+\epsilon \in \mathbb R^D$ 
	$\epsilon \sim \mathcal N(0,\sigma^2I)$ is Gaussian observation noise
	$B \in \mathbb R ^{D \times M}$ and $\mu \in \mathbb R^D$ describe linear/affine mapping from latent to observed vars. 

	Latent and observed variables are linked via
	$p(x|z,B,\mu,\sigma^2)=\mathcal N (x|Bz + \mu,\sigma^2I)$ 

And generative process looks like this overall
	$z_n \sim \mathcal N (z|0,I)$ 
	$x_n|z_n \sim \mathcal N(x|Bz_n + \mu,\sigma^2I)$ 
	To generate a data point that is typical given the params, we first sample $z_n$ from $p(z)$ and then a data point conditioned on the sampled $z_n$ , i.e. $x_n \sim \mathcal p(x|z_n,B, \mu,\sigma^2)$ *(ancestral sampling)* 

The whole probabilistic model can be written as
	$p(x,z|B,\mu,\sigma^2)=p(x|z,B,\mu,\sigma^2)p(z)$ 

By integrating out $z$, the likelihood is obtained 
	$p(x,z|B,\mu,\sigma^2)=\int p(x,z|B,\mu,\sigma^2)p(z) dz$ 
	$=\int \mathcal N (x|Bz+\mu,\sigma^2I)~\mathcal N(z|0,I)dz$ 
	
The solution to this is Gaussian with mean $\mu$ 
	$\mathbb E_x[x]=\mathbb E_z[Bz+\mu]+\mathbb E_{\epsilon}[\epsilon]=\mu$ 
	
And with covariance matrix
	$\mathbb V[x] = BB^⊤ +\sigma^2I$ 

The joint distribution of latent and observed rand-vars is given by
	$p(x,z|B,\mu,\sigma^2)=$ $\mathcal N(\begin{bmatrix} x \\ z \end{bmatrix} | \begin{bmatrix} \mu \\ 0 \end{bmatrix}), \begin{bmatrix} BB^⊤+\sigma^2I & B \\ B^⊤ & I \end{bmatrix})$ 
	the length of mean vector is $D+M$ and the size of covar matrix is $(D+M) \times (D+M)$ 

The posterior distribution of a latent variable is
	$p(z|x) = \mathcal N (z|m,C)$
	with 
	$m= B^⊤(BB^⊤+\sigma^2I)^{-1}(x-\mu)$ 
	$C=I-B^⊤(BB^⊤+\sigma^2I)^{-1}B$ <- tells how confident is the embedding

