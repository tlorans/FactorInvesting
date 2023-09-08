# Linear Regression: Linear Static Models with Observable Factors
## Linear Regression via Singular Value Decomposition and Pseudo-Inverse

We start with the previous linear model expressed previously as:

\begin{equation}
R = B \mathcal{F} + \epsilon
\end{equation}


With observable factors, the risk premia of a factor $\mathcal{F}_k$ corresponds to the excess return of the tradable portfolio ($\mathcal{F}_k = R_k)$, with $R_k$ the excess return of the corresponding sorted portfolio).
In this framework, $\mathcal{F}$ is known, only $B$ the matrix of exposure to these factors need to be estimated. 

We thus have a system of linear equations where $B$ is a matrix of unknown exposure to the vector of factors $\mathcal{F}$. We need to find a solution where the values for $B$ in the model minimize the squared error between the approximation of returns $\hat{R}$ by the resulting model and the realized returns $R$:

\begin{equation}
|| B \mathcal{F} - R ||^2
\end{equation}

This is called the linear least squares.

In matrix notation, this problem is formulated with the so-called normal equation:

\begin{equation}
\mathcal{F}^T \mathcal{F} B = \mathcal{F}^T R
\end{equation}

And reformulated in order to specify the solution for $B$ as:

\begin{equation}
\hat{B} = (\mathcal{F}^T\mathcal{F})^{-1}\mathcal{F}^T R
\end{equation}

It can be solved directly by computing the inverse, or it can be solved through the Singular Value Decomposition and the Pseudo-Inverse, such as:

\begin{equation}
\hat{B} = \mathcal{F}^{+} R
\end{equation}

where $\mathcal{F}^{+}$ is the pseudoinverse of the vector of factors $\mathcal{F}$. 

---
**Example X**

*Let's assume we want to estimate a 3-factors model (Fama and French, 1992):*

\begin{equation}
R_i = \beta^m_i \mathcal{F}_m + \beta^{smb}_i \mathcal{F}_{smb} + \beta^{hml}_i \mathcal{F}_{hml}
\end{equation}

*Because $\mathcal{F}_j$ is assumed to be observable with the excess return of the corresponding sorted portfolio, we have $\mathcal{F}_j = R_j$. We can thus rewrite the model as:*

\begin{equation}
R_i = \beta^m_i R_m + \beta^{smb}_i R_{smb} + \beta^{hml}_i R_{hml}
\end{equation}


*with $R_{smb}$ is the return of the Size factor and $R_{hml}$ is the return of the Value factor.*

*In the matrix form we have:*

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

*We can estimate the matrix $B$ with linear regression such as:*

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

## Linear Regression K-Factors Asset Pricing