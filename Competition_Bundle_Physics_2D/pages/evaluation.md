# Evaluation

The problem to estimate $\mu$, the signal strength as described in the physics theory.



### Metric
The submissions are evaluated using the following `score`:

$$
Score = \Delta(\mu) + \Delta(\Delta(\mu))
$$
where

$$
\Delta(\mu) = \mu - \hat(\mu)
$$

and 

$$
\Delta(\Delta(\mu)) = \Delta(\mu) - \Delta(\hat(\mu))
$$

Note that $\hat(\mu)$ and $\Delta(\hat(\mu))$ are computed by the ingestion program.

