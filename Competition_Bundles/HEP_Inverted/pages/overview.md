# Problem Setting
***
Collisions occuring in the LHC yield **low energy** and **high energy events**, the latter occuring rarely and being the events of interest. The occurrence of **high energy events is modeled by a Poisson process of know arrival rate expectation $\nu$** (which can be interpreted as the mean number of high energy events per experiment carried out). Among high energy events, physicists search for interesting events that might give them some information about possible new discoveries. Most of the events seen are "background", events that are expected and unrelated to new physics. The interesting "signal" events are far less frequently seen. Background events are well characterized from past experiments and simulations. The "known" **arrival rate expectation of background events is called** $\beta$. The arrival rate expectation of signal events is to be measured. It is denoted as $\mu \gamma$, where $\gamma$ is the estimated arrival rate expectation of signal events obtained from the Standard Model (SM, our current theoretical model to describe particle physics, which is known to be incomplete), and $\mu$ **is a "signal strength" factor, which we must evaluate from the new experiment at hand**.

The three arrival processes (high energy events, signal, and background) are all assumed to be Poisson processes and their arrival rate expectations are linked by the equation: $\nu = \beta + \mu \gamma$.

Each collision event results in a big fireworks, a snapshot of it is taken on the detectors of the LHC. After some preprocessing and pre-filtering of events, a feature vector $\bf x$ (signature) is extracted. Such feature vector can be used to "recognize" background events and "signal" events. However, the signatures of signal and some background events closely resemble each others. In other words the distributions overlap. Furthermore, there are very few signals compared to the number of backgrounds. Thus, there is no real hope to very accurately classify events. Since all we want is to estimate $\mu$, classifying accurately signals and backgrounds may not actually be necessary. 

# Estimation of $\mu$
***
### **Counting Estimator**
A simple maximum likelihood estimator based on Poisson counting process can be used as a baseline given as:  
$\Large \hat{\mu} = \frac{N - \beta}{\gamma} ~~~~~~~~ (1)$  
where $N \sim Poisson(\nu)$ is a random variable: the observed number of high energy events in an experiment.

Two methods for estimating $\mu$ are commonly used by physicists:

### **1. Histogram method**
Projecting $\bf x$ onto **one single feature**, constructed from $x$ in a smart way (usually from expert knowledge); create a histogram of events in that 1-dimensional projection (i.e. bin the events); apply formula (1) in each bin. Carrying out this method relies on the fact that $\mu$ can be estimated in each bin because $\Large \mu = \frac{\nu_i - \beta_i}{\gamma_i} = \frac{\nu - \beta}{\gamma}$, where $\beta_i$ and $\gamma_i$ are the SM expected values of background and signal events in each bin, which can be obtained by extensive Monte Carlo simulations using an accurate simulator of the data generating process, for $\mu=1$. This can yield an estimator of $\mu$:

$\Large \hat{\mu} = \sum_{i=1}^m w_i \frac{N_i - \beta_i}{\gamma_i}$  
It can be shown that this estimator has a lower variance than that of the plain Poisson counting process of the previous section: 
$\Large \sigma^2_{\hat{\mu}} \simeq \left( \sum_{i=1}^m \frac{\gamma_i^2}{\nu_i}\right)^{-1}$ 

Here, $\gamma_i$ and $\beta_i$ are generally NOT assumed to be known constants, only $\gamma$ and $\beta$ are. They must be estimated in each bin, e.g., using a simulator (which can be rather precise since we can generate a lot of data from the simulator). However, in the presence of systematics, the estimation will be biased. A re-estimation hypothesizing a given systematic error will be needed.

### **2. Classifier method**
Narrowing down the number of events to be considered to a Region Of Interest (ROI), rich in signal events, putting a threshold on the output of a classifier providing $Proba(y=signal|{\bf x})$, then apply the estimator:

$\Large \hat{\mu} = \frac{N_{ROI} - \beta_{ROI}}{\gamma_{ROI}}$

This estimator has variance:

$\Large \sigma^2_{\hat{\mu}} = \frac{\nu_{ROI}}{\gamma_{ROI}^2} $

which is lower than that of the plain Poisson counting process, iff  $\gamma_{ROI}$ . $\gamma_{ROI} / \nu_{ROI}$ > $\gamma$ . $\gamma / \nu$. We see that $\gamma_{ROI} / \nu_{ROI}$ > $\gamma / \nu$ is NOT a sufficient condition to lower the variance of the estimator of $\mu$. There is a tradeoff between increasing $\gamma_{ROI} / \nu_{ROI}$ and not decreasing $\gamma_{ROI}$ too much, that is going into regions "enriched" in signal, but in which the total number of signal events approaches 0.

Here $\gamma_{ROI}$ and $\beta_{ROI}$ are NOT assumed to be known constants (like in the histogram method); they need to be estimated with the simulator, and, likewise, could be plagued with systematic error. Thus, in the presence of systematics, this simple estimator under-estimates the variance of $\hat{\mu}$. **This is the problem we want to solve.**


# How to join this challenge?
***
- Go to the "Starting Kit" tab
- Download the "Dummy sample submission" or "sample submission"
- Go to "My Submissions" tab
- Submit the downloaded file


# Credits
***
- Isabelle Guyon
- David Rousseau
- Ihsan Ullah
- Mathis Reymond
- Ragansu Chakkappai