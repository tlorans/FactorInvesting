# Testing Asset Pricing Models

Objective asset pricing models: explaining cross-sections of returns.

## Characteristic-Managed Portfolios

We are going to construct characteristic-managed portfolios to form test assets.

The general approach is the following:

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

*We continue the previous example 1. We rank stocks according to their score and assign the corresponding qunatile.*

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

### Portfolio Returns

## Statistical Performance with Total and Predictive R-Squared

We follow Kelly et al. (2019) {cite:p}`kelly2019characteristics` to test the asset pricing performance of the factor models with "total $R^2$" and "predictive $R^2$. 

To understand the difference between both measures, let's start with the definition of the "traditional" $R^2$. 

The coefficient of determination, or $R^2$, is computed as:

\begin{equation}
R^2 = 1 - \frac{SSR}{SST}
\end{equation}

where $SSR$ stands for sum squared of the residuals and $SST$ stands for the total sum of squares, that is the su of the distance the data is away from the mean all squared.

The sum squared regression (SSR) is defined as:

\begin{equation}
SSR = \sum_i(y_i -\hat{y_i})^2
\end{equation}

where $y_i$ is the true value, $\hat{y}_i$ is the estimated value.

The total sum of squares (SST) is defined as:

\begin{equation}
SST = \sum_i(y_i - \bar{y})^2
\end{equation}

where $\bar{y}$ is the mean of the observations.

In the factor models, the "true" values are the realized returns $r_{i,t}$, where $i$ is the return of the characteristic-managed portfolio $i$.

Following Kelly et al. (2019) {cite:p}`kelly2019characteristics`, we assume the sample average of the observations $\bar{r_i}$ to be equal to 0. Thus, the SST is common to both the total $R^2$ and predictive $R^2$ and defined as:

\begin{equation}
SST = \sum_{i,t} r_{i,t}^2
\end{equation}

### Total R-Squared

The total $R^2$ quantifies the explanatory power of contemporaneous factor realizations. It summarizes how well the systematic factor risk in a given model specificiation describes the realized riskiness in the panel of individual stocks.

In this framework, the estimated $\hat{r}_{i,t}$ to be compared with the realized return $r_{i,t}$ is defined as:

\begin{equation}
\hat{r}_{i,t} = \hat{\beta}'_{i,t-1}\hat{f}_t
\end{equation}

The SSR is thus defined as:

\begin{equation}
SSR = \sum_{i,t}(r_{i,t} - \hat{\beta}'_{i,t-1}\hat{f}_t)^2
\end{equation}

The full formula of the total $R^2$ is thus:

\begin{equation}
\text{Total $R^2$} = 1 - \frac{\sum_{i,t}(r_{i,t} - \hat{\beta}'_{i,t-1}\hat{f}_t)^2}{\sum_{i,t} r_{i,t}^2}
\end{equation}

### Predictive R-Squared

The second measure that we refer to as the "predictive $R^2$" represents the the accuracy of the asset pricing model predictions of future returns. 

In that case, the estimated $\hat{r}_{i,t}$ is the predicted one from previous month information (including the factor return). We have:

\begin{equation}
\hat{r}_{i,t} = \hat{\beta}'_{i,t-1} \hat{\lambda}_{t-1}
\end{equation}

where $\hat{\lambda}_{t-1}$ is the sample average of $\hat{f}$ up to the previous month ($t-1$).

In that case, we have the following SSR:

\begin{equation}
SSR = \sum_{i,t}(r_{i,t} - \hat{\beta}'_{i,t-1}\hat{\lambda}_{t-1})^2
\end{equation}

And thus the following complete formula for the predictive $R^2$:

\begin{equation}
\text{Predictive $R^2$} = 1 - \frac{\sum_{i,t}(r_{i,t} - \hat{\beta}'_{i,t-1}\hat{\lambda}_{t-1})^2}{\sum_{i,t} r_{i,t}^2}
\end{equation}
## Risk Premia v.s. Mispricing

This test investigate whether ou factor models accurately "price" characteristics-managed portfolios (or anomaly portfolios) uncoditionally, following Gu et al. (2021) {cite:p}`gu2021autoencoder`.

As we implement all our models without intercepts, we can directly test whether the zero-intercept no-arbitrage restriction is satisfied. If it is, the time series average of model residuals for each portfolios, that is the pricing errors, should be indistinguishable from zero (two side test). The uncoditional pricing errors are defined as:

\begin{equation}
\alpha_i = \mathbb{E}[u_{i,t}] = \mathbb{E}[r_i,t] - \mathbb{E}[\beta_{i,t-1}'f_t]
\end{equation}