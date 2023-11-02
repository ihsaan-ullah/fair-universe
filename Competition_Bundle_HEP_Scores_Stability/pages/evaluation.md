# Evaluation
***

To evaluate the participants of the competition, several metrics are used for experimentation purposes.
## **Mean Squared Error (MSE): for $\hat{\mu}$ and $\Delta{\hat{\mu}}$**
    

$MSE_{\hat{\mu}} = \frac{1}{n} \sum_{i=1}^{n} (\mu - \hat{\mu}_i)^2 ~~~~~~~~~ (1)$

<br>

$MSE_{\Delta{\hat{\mu}}} = \frac{1}{n} \sum_{i=1}^{n} (\Delta{\hat{\mu}}_i - \widehat{\Delta{\mu}})^2 ~~~~~~~~~ (2)$

<br>

where $\Delta{\hat{\mu}_i} = |\mu - \hat{\mu}_i |$

***
    
## **Mean Absolute Error (MAE): for $\hat{\mu}$ and $\Delta{\hat{\mu}}$**

$MAE_{\hat{\mu}} = \frac{1}{n} \sum_{i=1}^{n} |\mu - \hat{\mu}_i| ~~~~~~~~~ (3)$

<br>

$MAE_{\Delta{\hat{\mu}}} = \frac{1}{n} \sum_{i=1}^{n} |\Delta{\hat{\mu}}_i - \widehat{\Delta{\mu}}| ~~~~~~~~~ (4)$

where $\Delta{\hat{\mu}_i} = |\mu - \hat{\mu}_i |$

***

## **Coverage**
Coverage is described as the number of times, true $\mu$ is between $\hat{\mu}\_{i}^+$ and $\hat{\mu}\_{i}^-$ (for experiment $i$). $\hat{\mu}\_{i}^+$ and $\hat{\mu}\_{i}^-$ can be computed in two ways:

*  $\hat{\mu}\_{i}^+ = \hat{\mu}\_{i} + \widehat{\Delta{\mu}}$ and $\hat{\mu}\_{i}^- = \hat{\mu}\_{i} - \widehat{\Delta{\mu}}$ where $\widehat{\Delta{\mu}}$ is computed by the participant's code
*  $\hat{\mu}\_{i}^+ = \hat{\mu}\_{i} + \frac{C}{2}$ and $\hat{\mu}\_{i}^- = \hat{\mu}\_{i} - \frac{C}{2}$ where $C$ is a constant e.g. 0.02 provided by the organizers. 

$\text{Coverage}\_{\mu, \widehat{\Delta{\mu}}}$ = $\frac{1}{n} \sum_{i=1}^{n} 1, \quad \hat{\mu}\_{i}^- \leq \mu \leq \hat{\mu}\_{i}^+; \quad 0, \quad \text{otherwise} ~~~~~~~~~ (5)$

where $\hat{\mu}\_{i}^+ = \hat{\mu}\_{i} + \widehat{\Delta{\mu}}$ and $\hat{\mu}\_{i}^- = \hat{\mu}\_{i} - \widehat{\Delta{\mu}}$

<br>

$\text{Coverage}\_{\mu, C} = \frac{1}{n} \sum_{i=1}^{n} 1, \quad \hat{\mu}\_{i}^- \leq \mu \leq \hat{\mu}\_{i}^+; \quad 0, \quad \text{otherwise} ~~~~~~~~~ (6)$

where $\hat{\mu}\_{i}^+ = \hat{\mu}\_{i} + \frac{C}{2}$ and $\hat{\mu}\_{i}^- = \hat{\mu}\_{i} - \frac{C}{2}$

***

## **Score MAE and MSE**

$\text{Score}\_{MAE} = MAE_{\hat{\mu}} + MAE_{\Delta{\hat{\mu}}} ~~~~~~~~~ (7)$

<br>

$\text{Score}\_{MSE} = MSE_{\hat{\mu}} + MSE_{\Delta{\hat{\mu}}} ~~~~~~~~~ (8)$

***

## Score Quantile
For a given dataset, $\mathbf d$ and unknown parameter $\mu$ we define the $q^\mathrm{th}$ quantile of the posterior distribution $p(\mu | \mathbf d)$ as the value $\mu_q$ such that the probability of $\mu$ being smaller than $\mu_q$ is $q$. The problem posed to participants is to determine the two quantiles $\mu_{25}, \mu_{75}$, i.e. the central region of the parameter space that contains the true value of $\mu$ with a probability of 50%.

We grade participants' algorithms on a test set of $N_\mathrm{test}$ datasets $\mathbf d_i$ where $i = 1~\dots N_\mathrm{test}$ by computing the following metric:

$J_q = \left(\Delta \mu + \epsilon\right) \times f\left(\frac{N_\in}{N_\mathrm{test}} \right), ~~~~~~~~~ (9)$

where:

$N_{\in} = \sum_{i = 1}^{N_\mathrm{ test}} I\left(\mu_{25} \leq \mu \leq \mu_\mathrm{75} \right),~~~~~~~~~ (10)$

and

$\Delta \mu = \frac{1}{N_\mathrm{test}} \sum_{i = 1}^{N_\mathrm{test}}\left(\mu_{75}^i - \mu_{25}^i \right).~~~~~~~~~ (11)$

$\epsilon$ is a small number that regularizes the behavior as $\Delta \mu$ and $N_\in$ as $\Delta \mu$ tends to zero. 
$f(x)$ is a function that penalizes departure from the target coverage of 50%. The simplest choice for $f(x)$ is the reciprocal of a polynomial $p(x)$ with roots at 0 and 1 and a minimum at 0.5, so we choose $p(x) = x(1 - x)$: 

$f(x) = \frac{p(x=0.5)}{p(x)}.$