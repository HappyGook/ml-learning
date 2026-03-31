(Support Vector Machines)

Binary classification is a task of classifying the object in one of two possible outcomes, i.e. the output of the predictor is a binary value.

The set of possible values in such case is denoted by
	$f: \mathbb R^D \rightarrow \lbrace +1, -1 \rbrace$ 

As in regression, the learning task is supervised, and has a set of examples $x_n \in \mathbb R^D$ with labels $y_n \in \lbrace +1, -1 \rbrace$ . The nonlinearity of the data is hidden in a transformation $\phi$, so the model is linear.

SVM provides a geometric view on supervised ml, and starts by designing a particular func that is then optimised during training. (loss function)

Classes can be separated by a hyperplane. Given two examples (as vectors) $x_i$ and $x_j$, the similarity between them can be computed via their inner product $\langle x_i, x_j \rangle$ .

Main idea of many classification algorithms is to represent the data in $\mathbb R^D$ and then partition the space in a way that examples with the same label are in the same partition.

Linear split of the space into two halves using a hyperplane can be defined by the following function, for an example $x \in \mathbb R^D$ 
	$f: \mathbb R^d \rightarrow \mathbb R$
	$x \mapsto f(x):=\langle w,x \rangle +b$ 
	parametrised by $w \in \mathbb R^D$ and $b \in \mathbb R$ 
	$w$ is orthogonal to any vector on the hyperplane

Hyperplanes are affine subspaces and for the problem of separating two classes can be defined as
	$\lbrace x \in \mathbb R^D:f(x)=0 \rbrace$ 
Geometrically thinking, the the positive examples lie above the hyperplane and negative under it.

When training, it is desired to ensure that examples with positive labels are on the positive side of the hyperplane and ones with negative labels are on the negative side
	$\langle w,x_n \rangle + b \geqslant 0~~~$ when $~~~y_n=+1$ 
	$\langle w,x_n \rangle + b \leqslant 0~~~$ when $~~~y_n=-1$ 

Single equation for both conditions is $y_n(\langle w,x_n \rangle +b) \geqslant 0$ 

For a dataset $\lbrace (x_1,y_1),...,(x_N,y_N) \rbrace$ that is linearly separable, there are infinitely many candidate hyperplanes (and therefore classifiers), that solve the problem without any training errors. To choose a unique solution, the hyperplane that maximises the margin between positive & negative examples is chosen. (via orthogonal projections)

The overall optimisation problem is defined as ^maxmargin
	$max_{w,b,r}~\underbrace{r}_{margin}$ 
	subject to $\underbrace{y_n(\langle w,x_n \rangle +b)\geqslant r}_{data~fitting},~\underbrace{||w||=1}_{normalisation},~~r>0$ 
	Only the direction of $w$ matters and not its length.
The goal is to maximise the margin r while ensuring that the data lies on the correct size of the hyperplane. ^e8a0e4

A different assumption can be made. Instead of choosing that the parameter vector is normalised, a scale for the data can be chosen. The scale is chosen so that the value of the predictor $\langle w,x \rangle +b =1$ at the closest example. The example in the data set, that is closest to the hyperplane is denoted by $x_a$ ^eaaf31

Then, $x_a^{'}$ is the orthogonal projections of $x_a$ onto the hyperplane
	$\langle w,x_a \rangle +b = 1$
	$\langle w,x_a^{'} \rangle +b =0$ 

$x_a$ can be displayed as its projection + distance, which is also normalised
	$x_a = x_a^{'} + r\cfrac{w}{||w||}$ 

By substituting this into the previous equation
	$\langle w,x_a - r\cfrac{w}{||w||} \rangle + b= 0$   
	$\Leftrightarrow \underbrace{\langle w,x_a \rangle +b}_{=1} -r\underbrace{\cfrac{\langle w,w \rangle}{||w||}}_{=r||w||}$  
	
	$r = \cfrac{1}{||w||}$ <- distance to the hyperplane

With this, the overall optimisation problem can be formed as^scaledata
	$max_{w,b}~\cfrac{1}{||w||}$ 
	subject to $y_n(\langle w,x_n \rangle +b)\geqslant 1$, for all $n=1,...,N$ 
	(1 since [[#^eaaf31]])
	
Instead of maximising the reciprocal norm, often the squared norm is minimised. Also a constant is often included, that doesn't affect the optimal $w$ an $b$ but makes the form more tidy when computing gradients (here $\cfrac{1}{2}$). Thus, the **hard margin SVM** (hard since no violations of margin are allowed) is defined as
	$min_{w,b}~\cfrac{1}{2}||w||^2$  
	subject to $y_n(\langle w,x_n \rangle +b)\geqslant 1$, for all $n=1,...,N$ 

Maximising the margin r with normalised weights [[#^maxmargin]] is equivalent to scaling the data [[#^scaledata]], such that the margin is unity.

### Soft Margin SVM

In case data is not linearly separable, some examples may be allowed to fall within the margin region, or even to the wrong side of the hyperplane. Soft Margin SVM allows for some classification errors. There are many ways to view this algorithm.

**Geometric idea** introduces a slack variable $\xi_n$ for each example-label ($x_n,y_n$) that allows an example to be within a margin / on the wrong side of the hyperplane. $\xi_n$ is added to the objective to encourage the correct classification and subtracted from the margin, so that the optimisation problem becomes 
	$min_{w,b,\xi}~\cfrac{1}{2}||w||^2+C\displaystyle\sum_{n=1}^N~\xi_n$ 
	subject to $y_n(\langle w,x_n \rangle +b) \geqslant 1 - \xi_n$ for $n=1,...,N$ 
	and $\xi_n$ is constrained $\xi_n \geqslant 0$ 
	
The regularisation parameter $C>0$ trades off the size of the margin and the total amount of slack. The margin term $||w||^2$ is called regulariser.  ^778a89

Following the empirical risk minimisation, the hyperplanes are chosen as the hypothesis class
	$f(x)=\langle w,x \rangle +b$ 
	
The ideal loss function between binary labels is a **zero-one loss**, which is 1 if $y_n=f(x_n)$ (correct classification) and 0 if $y_n \not = f(x_n)$ (incorrect classification) 
	$1(f(x)\not = y_n)$ 
This function, however, is hard to use for $w,b$ optimisation.

The loss describes the error that is made of training data. For this **hinge loss** can be used
	$l(t)= \begin{cases} 0 &\text{if } t \geqslant 1 \\ 1-t &\text{if } t<1 \end{cases}$   for the Soft SVM
	and
	$l(t)= \begin{cases} 0 &\text{if } t \geqslant 1 \\ ∞ &\text{if } t<1 \end{cases}$   for the Hard SVM

For a given training set the total loss is minimised, while regularising the objective with $l_2$-regularisation. This, and hinge loss, gives the unconstrained optimisation problem
	$min_{w,b}~\underbrace{\cfrac{1}{2}||w||^2}_{regulariser}+\underbrace{C\displaystyle\sum_{n=1}^N~max\lbrace 0,1 - y_n(\langle w,x_n\rangle +b)\rbrace}_{error~term}$  
This unconstrained problem can be solved via gradient descent ^softprimal

The minimisation of hinge loss over $t$ can be replaced with minimisation of slack-var $\xi$ with two constraints.
		$\underset {t}{min}~max \lbrace 0,1-t \rbrace$ 
	is equivalent to
		$\underset{\xi,t}{min~~\xi}$
		subject to  $\xi \geqslant 0$, $~~\xi \geqslant 1-t$ 
	By substituting this into the [[#^778a89]], we get the soft SVM

## Dual SVM

Primal SVM's number of parameters grows linearly with the number of features.

Dual SVM is independent of the number of features and instead increases with the number of examples in the training set. It is useful if amount of examples is smaller than amount of features. It also allows easy kernel application.

Primal soft margin SVM ([[#^softprimal]]) has $w,b, \xi$ as its primal variables. The Lagrange multiplier $\alpha_n \geqslant 0$ is used, corresponding to the constraint that examples are classified correctly and $\gamma_n \geqslant 0$. $\alpha_n$ corresponds to the non-negativity constraint of the slack-var. The Lagrangian is given by:
	$\mathfrak L(w,b,\xi,\alpha,\gamma)=\cfrac{1}{2}||w||^2+C\displaystyle\sum_{n=1}^N~\xi_n -$
	$\underbrace{-\displaystyle\sum_{n=1}^N~\alpha_n(y_n(\langle w, x_n \rangle +b)-1 +\xi_n)}_{constraint~1}$   $\underbrace{-\displaystyle\sum_{n=1}^N~\gamma_n\xi_n}_{constraint~2}$ 
	With constraint 1: 
		subject to $y_n(\langle w,x_n \rangle +b) \geqslant 1 - \xi_n$ for $n=1,...,N$ 
	and constraint 2:
		 $\xi_n$ is constrained $\xi_n \geqslant 0$ 

Differentiating the Lagrangian with respect to three primal variables
	$\cfrac{\partial \mathfrak L}{\partial w}=w^⊤ - \displaystyle\sum_{n=1}^N~\alpha_n y_n x_n^⊤$ 
	$\cfrac{\partial \mathfrak L}{\partial b}= - \displaystyle\sum_{n=1}^N~\alpha_n y_n$
	$\cfrac{\partial \mathfrak L}{\partial \xi_n}=C -\alpha_n - \gamma_n$ 
	
The maximum is found by setting each to zero and from it $w$ should be
	$w=\displaystyle\sum_{n=1}^N~\alpha_ny_nx_n$ 
	This is an instance of the representer theorem, that states that the solution of minimising empirical risk lies in the subspace defined by the examples. ^representer

By inserting the $w$ as a sum defined before, the dual Lagrangian can be defined
	$\mathfrak D(\xi,\alpha,\gamma)=-\cfrac{1}{2}~\displaystyle\sum_{i=1}^N\displaystyle\sum_{j=1}^N~y_iy_j\alpha_i\alpha_j \langle x_i,x_j \rangle$ $+\displaystyle\sum_{i=1}^N \alpha_i$ $+ \displaystyle\sum_{i=1}^N(C-\alpha_i -\gamma_i)\xi_i$ 
	
The last term here can be set to 0, since it is a partial derivative $\cfrac{\partial \mathfrak L}{\partial \xi_n}$ 
Since Lagrangian multipliers $\gamma_i$ are non-negative, $\alpha_i \leqslant C$ 

The dual SVM is defined as
	$\underset{\alpha}{min}~$ $\cfrac{1}{2}~\displaystyle\sum_{i=1}^N\displaystyle\sum_{j=1}^N~y_iy_j\alpha_i\alpha_j \langle x_i,x_j \rangle$ $- \displaystyle\sum_{i=1}^N \alpha_i$ 
	subject to $\displaystyle\sum_{i=1}^N y_i\alpha_i =0$
	$0 \leqslant \alpha_i \leqslant C$ for all $i=1,...,N$ 

The set of inequality constraints in SVM are called box constraints, because the limit vector $\alpha=[\alpha_1,...,\alpha_N]^⊤ \in \mathbb R^N$ of Lagrange multipliers to be inside the box defined by 0 and $C$ on each axis.

From $\alpha$, the primal params $w$ can be recovered with [[#^representer]] 
The optimal $w$ is denoted as $w^*$, and to find $b^*$ we exploit that for $x_n$ that lies exactly on the margin boundary, i.e. $\langle w^*,x_n \rangle +b = y_n$ with $y_n =\lbrace -1, +1 \rbrace$, the $b$ is
	$b^*=y_n-\langle w^*,x_n \rangle$ 

If there are no examples directly on the margin, $|y_n-\langle w^*,x_n \rangle|$ is computed for all support vectors and the median of this absolute value difference is the value of $b^*$ 
	Support vectors are examples for which $\alpha_n > 0$, since they support the hyperplane.
	And examples $x_n$ for which $\alpha_n=0$ don't contribute to the solution $w$.

386