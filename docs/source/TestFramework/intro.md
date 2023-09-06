# Testing Asset Pricing Models

Objective asset pricing models: explaining cross-sections of returns.

## Characteristic-Sorted Portfolios

We are going to construct characteristic-sorted portfolios to form test assets. Portfolio sorting has been popularized by Fama and French (1993) {cite:p}`fama1993common` to test the impact of characteristics in asset pricing.  The basic premise involves categorizing individual assets into different portfolios based on a chosen variable. If these portfolios are roughly equal in size and vary mainly in the value of this sorting variable, then any performance differences can be ascribed to the influence of that variable. Typically, portfolios are balanced either equally or by asset value to optimize diversification. 

The general approach can be described as the following, as summarized by Roncalli (2021) {cite:p}`roncallilecture3assetmanagement`:

- start with a universe $\mathcal{U}$ of stocks
- define a rebalancing period (e.g. every month, every quarter, every year)
- for each rebalancing date $t_{\tau}$:
    - define a score $\mathbb{S}_i(t_{\tau})$ for each stock $i$
    - stocks with high scores are selected to form the long exposure $\mathcal{L}(t_{\tau})$ of the characteritstic
    - stocks with low scores are selected to form the short exposure $\mathcal{S}(t_{\tau})$ of the characteristic
- specify a weighting scheme $w_i(t_{\tau})$, for example value weighted or equally weighted
- compute the returns performance of the characteristic-managed portfolio $\mathcal{C}(t)$
### Scoring

Let's write $x_i(t_{\tau})$ at the rebalancing date $t_{\tau}$, and $\mathbb{S}_i(t_{\tau})$ is the resulting score.

We can score the characteristics following a min-max rescaling over $[-1, 1]$, such as:

\begin{equation}
\mathbb{S}_i(t_{\tau}) = 2 \frac{x_{i}(t_{\tau}) - \min(x(t_{\tau}))}{\max(x(t_{\tau}))-\min(x(t_{\tau}))} - 1
\end{equation}

The rescaling method must be applied separately for each rebalancing date. 

---
**Example 1**

*Let's have a numerical example. We have 10 assets and the corresponding characteritic data. We find the following parameters of our scoring function: $\max(x) = 120.9$ and $\min(x) = 2.6$. We then apply the scoring function to each characteristic value.*

| Asset | Characteristic | Score  | 
|---|---|---|
| $A_1$  | 34.2  | -0.4658  |
| $A_2$  |  65.4 | 0.0617  |
| $A_3$ | 12.3  | -0.8360  |
| $A_4$ | 32.7  |  -0.4911 |
| $A_5$ |  98.1 | 0.6145  |
| $A_6$ |  7.4 |  -0.9189 |
| $A_7$ |  2.6 | -1  |
| $A_8$ | 120.9  | 1  |
| $A_9$ |  12.4 |  -0.8343 |
| $A_{10}$ |  56.0 |  -0.0972 |

[Matlab code](https://github.com/tlorans/FactorInvesting/blob/main/materials/matlab/chap1/example1.m)


[Julia code](https://github.com/tlorans/FactorInvesting/blob/main/materials/julia/chap1/example1.jl)

---
### Sorting Procedure

As a simple approach, we can form five quintile portfolios:
- $\mathcal{Q}_1$ corresponds to the stocks the highest scores (top 20\%)
- $\mathcal{Q}_2$,$\mathcal{Q}_3$,$\mathcal{Q}_4$, are the second, third and fourth quintile portfolios
- $\mathcal{Q}_5$ corresponds to the stocks with the lowest scores (bottom 20\%)

The long portolio $\mathcal{L}$ will comprise the stocks in the first quintile $\mathcal{Q}_1$, while the short portfolio $\mathcal{S}$ will comprise the stocks in the last quintile $\mathcal{Q}_5$.

---
**Example 2**

*We continue the previous example 1. We rank stocks according to their score and assign the corresponding quantile. The two stocks with the highest scores belong to the first quantile (the long portfolio), while the two stocks with the lowest scores belong to the last quantile (the short portfolio).*

| Asset | Score | Rank | Quintile  | $\mathcal{L}$ / $\mathcal{S}$ | 
|---|---|---|---|---|
| $A_1$  | -0.4658  | 5 | $\mathcal{Q}_3$ | |
| $A_2$  | 0.0617  | 3 | $\mathcal{Q}_2$ | |
| $A_3$ | -0.8360  | 8 | $\mathcal{Q}_4$ | |
| $A_4$ | -0.4911  | 6 | $\mathcal{Q}_3$ | |
| $A_5$ | 0.6145 | 2 | $\mathcal{Q}_1$ | $\mathcal{L}$ |
| $A_6$ | -0.9189 | 9 |  $\mathcal{Q}_5$ | $\mathcal{S}$ |
| $A_7$ | -1  | 10 | $\mathcal{Q}_5$ | $\mathcal{S}$ |
| $A_8$ | 1  | 1 | $\mathcal{Q}_1$ | $\mathcal{L}$
| $A_9$ | -0.8343 | 7 | $\mathcal{Q}_4$ |
| $A_{10}$ | -0.0972 | 4 | $\mathcal{Q}_2$ |

[Matlab code](https://github.com/tlorans/FactorInvesting/blob/main/materials/matlab/chap1/example2.m)


[Julia code](https://github.com/tlorans/FactorInvesting/blob/main/materials/julia/chap1/example2.jl)

---

### Weighting Scheme

We now move to the weighting scheme. Generally, each portfolio is equally- or value-weighted in order to maximize the diversification (Roncalli, 2023) {cite:p}`Roncalli2023`. 

In the case of an equal weighting scheme, we have:

\begin{equation}
w_i = w_j = \frac{1}{n}
\end{equation}

where $w_i$ is the weight of the asset $i$, $w_j$ the weight of the asset $j$ and $n$ the number of assets in the portfolio.

---
**Example 3**

*We continue the previous example 2. We define the weights of assets in the long and short portfolios as equally-weighted. We have $n = 2$ for both portfolios.*

| Asset |  $\mathcal{L}$ / $\mathcal{S}$ | Weight |
|---|---|---|
| $A_1$  | | |
| $A_2$  |  | |
| $A_3$ |  | |
| $A_4$ |  | |
| $A_5$ | $\mathcal{L}$ |50\% |
| $A_6$ |  $\mathcal{S}$ | 50% |
| $A_7$ | $\mathcal{S}$ | 50% |
| $A_8$ |  $\mathcal{L}$ | 50% |
| $A_9$ |  | |
| $A_{10}$ | | |

[Matlab code](https://github.com/tlorans/FactorInvesting/blob/main/materials/matlab/chap1/example3.m)


[Julia code](https://github.com/tlorans/FactorInvesting/blob/main/materials/julia/chap1/example3.jl)

---

### Characteristic-Sorted Portfolio Performance

We first need to compute $R_t(\mathcal{L})$ and $R_t(\mathcal{S})$, the returns of the long and short portfolios between $t-1$ and $t$:

\begin{equation}
R_{\mathcal{L}}(t) = w_{\mathcal{L}}^T r_t 
\end{equation}

\begin{equation}
R_{\mathcal{S}}(t) = w_{\mathcal{S}}^T r_t 
\end{equation}

where $w_{\mathcal{L}}$ and $w_{\mathcal{S}}$ are the vectors of weights in the long and short portfolios respectively (within two rebalancing dates $\tau$ and $\tau +1$, such as the weights are fixed), and $r_t$ is the vector of assets returns between $t-1$ and $t$.

The performance of the long short porfolio $\mathcal{L} - \mathcal{S}$ satisfies the following definition (Roncalli, 2023 {cite:p}`Roncalli2023`):

\begin{equation}
(1 + R_{\mathcal{L}}(t)) = (1 + C(t))(1 + R_{\mathcal{S}}(t))
\end{equation}

Then we can deduce that:

\begin{equation}
C(t) = \frac{R_{\mathcal{L}}(t) - R_{\mathcal{S}}(t)}{1 + R_{\mathcal{S}}(t)}
\end{equation}

---
**Example 4**

*We continue the previous example 3. We have the following vector of assets returns:*

\begin{equation}
r_t^T = \begin{pmatrix}
0.03 &
-0.04 &
0.01 &
-0.02 &
-0.18 &
0.06 &
0.02 &
0.08 &
-0.01 &
0.06
\end{pmatrix}
\end{equation}

and the following vectors of weights:

\begin{equation}
w_{\mathcal{L}}^T = \begin{pmatrix}
0.0 & 0.0 & 0.0 & 0.0 & 0.5 & 0.0 & 0.0 & 0.5 & 0.0 & 0.0
\end{pmatrix}
\end{equation}

\begin{equation}
w_{\mathcal{S}}^T = \begin{pmatrix}
0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.5 & 0.5 & 0.0 & 0.0 & 0.0
\end{pmatrix}
\end{equation}

We thus have:

\begin{equation}
R_t(\mathcal{L}) = 0.5 \times -0.18 + 0.5 \times 0.08 = -0.05
\end{equation}

\begin{equation}
R_t(\mathcal{S}) = 0.5 \times 0.06 + 0.5 \times 0.02 = 0.04
\end{equation}

Then, we can now obtain the characteristic-sorted portfolio performance:

\begin{equation}
C(t) = \frac{-0.08 - 0.04}{1 + 0.04} = - 0.0865
\end{equation}


---

## Statistical Performance

We can measure the statistical performance of the asset pricing model with the resulting $\mathfrak{R}^2$, which can be defined as:

\begin{equation}
\mathfrak{R}^2 = 1 - \frac{\sigma^2(\epsilon)}{\sigma^2(r)}
\end{equation}

where $\sigma^2(\epsilon)$ corresponds of the sum squared of the residuals of the model (ie. the sum of the difference between actual and estimated returns, all squares) and $\sigma^2(r)$ corresponds to the total sum of squares (ie. the sum of the distance the returns are away from the mean, all squared).

More formally we have the sum squared of the residuals defined as:

\begin{equation}
\sigma^2(\epsilon) = \sum^T_{t = 1}(r(t) - \hat{r}(t))^2
\end{equation}

with $\hat{r}(t)$ the estimated return according to the asset pricing model.

And the total sum of squares defined usually as:

\begin{equation}
\sigma^2(r) = \sum^T_{t=1}(r(t) - \mu)^2
\end{equation}

where $\mu$ is the sample average of returns.

Following Kelly et al. (2019) {cite:p}`kelly2019characteristics`, we assume the sample average of the returns $\mu$ to be equal to 0. Thus, $\sigma^2(r) = \sum^T_{t=1}(r(t))^2$.


In this framework, the estimated $\hat{r}(t)$ to be compared with the realized return $r(t)$ is defined as:

\begin{equation}
\hat{r}(t) = \hat{\beta}'(t-1)\hat{f}(t)
\end{equation}

with $\hat{\beta}$ the vector of estimated betas.

Our full formula is thus:

\begin{equation}
\mathfrak{R}^2 = 1 - \frac{\sum^T_{t = 1}(r(t) - \hat{\beta}'(t-1)\hat{f}(t))}{\sum^T_{t=1}(r(t))^2}
\end{equation}

## Risk Premia v.s. Mispricing

This test investigate whether ou factor models accurately "price" characteristics-managed portfolios (or anomaly portfolios) uncoditionally, following Gu et al. (2021) {cite:p}`gu2021autoencoder`.

As we implement all our models without intercepts, we can directly test whether the zero-intercept no-arbitrage restriction is satisfied. If it is, the time series average of model residuals for each portfolios, that is the pricing errors, should be indistinguishable from zero (two side test). The uncoditional pricing errors are defined as:

\begin{equation}
\alpha_i = \mathbb{E}[u_{i,t}] = \mathbb{E}[r_i,t] - \mathbb{E}[\beta_{i,t-1}'f_t]
\end{equation}