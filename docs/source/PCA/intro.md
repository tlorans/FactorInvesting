# Principal Component Analysis: Linear Static Models with Latent Factors
## PCA via Singular Value Decomposition

We start again with our linear formula:

\begin{equation}
R = B \mathcal{F} + \epsilon
\end{equation}

Unlike the static models with observable factors, an other approach is to assume that factors are implicit variables (ie. not observable). These latent factors can be estimated with the Principal Component Analysis (PCA) method on the covariance matrix of returns.

In that approach, both $\mathcal{F}$ and $B$ needs to be estimated. 

The first stage of the approach involves applying an eigen-decomposition to the sample covariance matrix, such as:

\begin{equation}
\hat{\Sigma}_R = V \Lambda V^T
\end{equation}

Where $V$ is the matrix comprised of the eigenvectors and $\Lambda$ is the diagonal matrix comprised of the eigen values.

Then, principal components are sorted according to their eigenvalue. Each $kth$ principal component has the the $kth$ largest variance, conditional on being orthogonal to all previous principal components. 



The vector of factors $\mathcal{F}$ is now to be estimated:

\begin{equation}
\hat{\mathcal{F}} = R\hat{B}(\hat{B}^T \hat{B})^{-1}
\end{equation}



---
**Example X**

*To better understand the process involved with dimensionality reduction using PCA. let's consider a simple case where $K=1$. It means that we are interest into finding the most relevant dimension. We need to finding the direction $p$ which allows the best possible average reconstruction of the returns, that is the solution of the problem:*

---




## PCA K-Factors Asset Pricing