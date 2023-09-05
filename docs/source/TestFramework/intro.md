# Testing Asset Pricing Models

Objective asset pricing models: explaining cross-sections of returns and finding the true tangency portfolio (ie. with the maximum sharpe ratio).

## Characteristics-Managed Portfolios

### Rank-Normalization

### Sorting Procedure

## Statistical Performance with Total and Predictive R-Squared

We follow Kelly et al. (2019)  {cite:p}`kelly2019characteristics` to test the asset pricing performance of the factor models with "total $R^2$" and "predictive $R^2$. 

To understand the difference between both measures, let's start with the definition of the "traditional" $R^2$. 

The coefficient of determination, or $R^2$, is computed as:

\begin{equation}
R^2 = 1 - \frac{SSR}{SST}
\end{equation}

where $SSR$ stands for sum squared of the residuals and $SST$ stands for the total sum of squares, that is the su of the distance the data is away from the mean all squared.

The sum squared regression (SSR) is defined as:

\begin{equation}
SSR = \sum(y_i -\hat{y_i})^2
\end{equation}

where $y_i$ is the true value, $\hat{y}_i$ is the estimated value.

The total sum of squares (SST) is defined as:

\begin{equation}
SST = \sum(y_i - \bar{y})^2
\end{equation}

where $\bar{y}$ is the mean of the observations.
### Total R-Squared

The model performance 

### Predictive R-Squared

## Risk Premia v.s. Mispricing
