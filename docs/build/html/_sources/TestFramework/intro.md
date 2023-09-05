# Testing Asset Pricing Models

Objective asset pricing models: explaining cross-sections of returns.

## Characteristics-Managed Portfolios

### Characteristics Rescaling

Let's write $x_i$ for the raw input and $\tilde{x}_i$ for the transformed data. 

We normalize the characteristics following a min-max rescaling over $[-1, 1]$, such as:

\begin{equation}
\tilde{x}_{i,t} = 2 \frac{x_{i,t} - \min(x_t)}{\max(x_t)-\min(x_t)} - 1
\end{equation}

Note the $t$ subscript. The rescaling method is applied separately month by month. This ensures that for each rebalancing date of our characteristics-managed portfolios, the characterisics will have a similar shape and carrying information about the cross-section fo returns.
### Sorting Procedure

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