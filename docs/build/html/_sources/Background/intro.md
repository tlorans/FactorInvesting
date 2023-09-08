# Background

## The Risk Factor Framework

### Arbitrage Pricing Theory

Ross (1976) {cite:p}`ross2013arbitrage` proposed an alternative model to the CAPM, which is called the arbitrage pricing theory (APT).

In this model, the return of asset $i$ is driven by a linear factor model:

\begin{equation}
R_i = \alpha_i + \sum^K_{k=1} \beta^k_i \mathcal{F}_k + \epsilon_i
\end{equation}

where $\alpha_i$ is the intecept, $\beta^k_i$ is the sensitivity of asset $i$ to factor $k$ and $\mathcal{F}_k$ is the innovation of factor $k$. $\epsilon_i$ is the idiosyncratic risk of asset $i$.

Using APT, we can show that the risk premium of asset $i$ is a function of the risk premia of the factors:

\begin{equation}
\pi_i = \mathbb{E}[R_i] - R_f = \sum^K_{k=1} \beta^k_i \pi (\mathcal{F}_k)
\end{equation}

We can rewrite it with some linear algebra as:

\begin{equation}
R = \alpha + B \mathcal{F} + \epsilon
\end{equation}

where $R$ is a $(N \times 1)$ vector of asset excess returns, $\alpha$ is a $(N \times 1)$ vector, $B$ is a $(N\times K)$ matrix, $\mathcal{F}$ is a $(K \times 1)$ vector of factor returns and $\epsilon$ is a $(N \times 1)$ vector of erros.

Assuming the no-arbitrage condition stated by the APT theory, we have $\alpha = 0$.

### The Great Divide and How Machine Learning Can Help in Factor Modelling
#### Observable vs. Latent Factors

FF model: factors are observable. Characteristic of assets not directly used.

Other stream: factors are latent (PCA)

#### Static vs. Conditional Models

FF and PCA: exposures (beta) are invariant, only ad-hoc procedure for having time-variant estimates (ie. running rolling regressions), it thus ignore covariate

IPCA: conditional model with time-varying beta, taking into account covariates

#### Machine Learning as a Tool for Asset Pricing Model Generalization

Dimensionality reduction

## Singular Value Decomposition Tutorial

Singular Value Decomposition is a theorem for linear algebra which says that a rectangular matrix A can be broken down into the product of three matrices:
- an orthogonal matrix $U$, 
- a diagonal matrix $S$
- the transpose of an orthogonal matrix $V$.

The theorem can be presented like this:

\begin{equation}
A = U S V^T
\end{equation}

where $U^TU = I$, $V^TV = I$. The columns of $U$ are orthonormal eigenvectors of $AA^T$. The columns of $V$ are orthonormal eigenvectors $A^TA$ and $S$ is a diagonal matrix containing the square roots of eigenvalues from $U$ or $V$ in descending order.

