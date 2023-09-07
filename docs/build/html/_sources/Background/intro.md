# The Risk Factor Framework

## Capital Asset Pricing Model

The capital asset pricing model (CAPM) was introduced by Sharpe in 1964 {cite:p}`sharpe1964capital`, and can be viewed as an equilibrium model based on the framework defined by Markowitz (1952 {cite:p}`markowitz1952portfolio`).

In his paper, Markowitz develop the efficient frontier concept, i.e. the set of optimal mean-variance portfolios.

\begin{equation}
\mathbb{E}[R_i] - R_f = \beta^m_i(\mathbb{E}[R_m] - R_f)
\end{equation}

where $R_i$ and $R_m$ are the asset and market returns, $R_f$ is the risk-free rate and the coefficient $\beta^m_i$ is the beta of the asset $i$ with respect to the market portfolio:

\begin{equation}
\beta^m_i = \frac{cov(R_i, R_m)}{\sigma^2(R_m)}
\end{equation}

## Arbitrage Pricing Theory and Factor Models

Ross (1976) {cite:p}`ross2013arbitrage` proposed an alternative model to the CAPM, which is called the arbitrage pricing theory (APT).

In this model, the return of asset $i$ is driven by a linear factor model:

\begin{equation}
R_i = \alpha_i + \sum^m_{j=1} \beta^j_i \mathcal{F}_j + \epsilon_i
\end{equation}

where $\alpha_i$ is the intecept, $\beta^j_i$ is the sensitivity of asset $i$ to factor $j$ and $\mathcal{F}_j$ is the innovation of factor $j$. $\epsilon_i$ is the idiosyncratic risk of asset $i$.

Using APT, we can show that the risk premium of asset $i$ is a function of the risk premia of the factors:

\begin{equation}
\pi_i = \mathbb{E}[R_i] - R_f = \sum^m_{j=1} \beta^j_i \pi (\mathcal{F}_j)
\end{equation}

We can rewrite it with some linear algebra as:

\begin{equation}
R = \alpha + B \mathcal{F} + \epsilon
\end{equation}

where $R$ is a $(n \times 1)$ vector of asset excess returns, $\alpha$ is a $(n \times 1)$ vector, $B$ is a $(n\times m)$ matrix and $\mathcal{F}$ is a $(m \times 1)$ vector of factor returns.

zero intercept no-arbitrage restiction

## The Great Divide and How Machine Learning Can Help in Factor Modelling
### Observable vs. Latent Factors

### Static vs. Conditional Models

### Machine Learning as a Tool for Asset Pricing Model Generalization