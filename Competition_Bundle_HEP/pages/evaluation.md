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