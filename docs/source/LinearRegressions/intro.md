# Linear Regression: Static Models with Observable Factors
## Estimating Static Models with Observable Factors with Linear Regression

We start with the previous linear model expressed previously as:

\begin{equation}
R = \alpha + B\mathcal{F} + \epsilon
\end{equation}

In our case, because we want to test if asset pricing models price correctly each characteristic-sorted portfolio, we are going to estimate:

\begin{equation}
R = B \mathcal{F} + \epsilon
\end{equation}

We can estimate the matrix of factor exposure $B$ with a linear regression model, such as:

\begin{equation}
\hat{B} = (\mathcal{F}^T\mathcal{F})^{-1}\mathcal{F}^T R
\end{equation}

---
**Example X**

*Let's assume we want to estimate a 3-factors model (Fama and French, 1992):*


\begin{equation}
R_i = \beta^m_i R_m + \beta^{smb}_i R_{smb} + \beta^{hml}_i R_{hml}
\end{equation}


*with $R_{smb}$ is the return of the Size factor and $R_{hml}$ is the return of the Value factor, we can rewrite it in matrix form as:*

\begin{equation}
\begin{pmatrix}
R_1 \\
R_2 \\
\vdots \\
R_n
\end{pmatrix} = 
\begin{pmatrix}
\beta_1^m & \beta_1^{smb} & \beta_1^{hml} \\
\beta_2^m & \beta_2^{smb} & \beta_2^{hml} \\
\vdots & \vdots & \vdots \\
\beta_n^m & \beta_n^{smb} & \beta_n^{hml} \\
\end{pmatrix}
\begin{pmatrix}
R_m \\
R_{smb} \\ 
R_{hml}
\end{pmatrix}
+ \begin{pmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_n
\end{pmatrix}
\end{equation}

We can estimate the matrix $B$ with linear regression such as:

\begin{equation}
\begin{pmatrix}
\hat{\beta}_1^m & \hat{\beta}_1^{smb} & \hat{\beta}_1^{hml} \\
\hat{\beta}_2^m & \hat{\beta}_2^{smb} & \hat{\beta}_2^{hml} \\
\vdots & \vdots & \vdots \\
\hat{\beta}_n^m & \hat{\beta}_n^{smb} & \hat{\beta}_n^{hml} \\
\end{pmatrix} = 
(\begin{pmatrix}
R_m \\
R_{smb} \\ 
R_{hml}
\end{pmatrix}^T \begin{pmatrix}
R_m \\
R_{smb} \\ 
R_{hml}
\end{pmatrix})^{-1}
\begin{pmatrix}
R_m \\
R_{smb} \\ 
R_{hml}
\end{pmatrix}^T
\begin{pmatrix}
R_1 \\
R_2 \\
\vdots \\
R_n
\end{pmatrix}
\end{equation}
---




## FF K-Factors Asset Pricing