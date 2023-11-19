# Overview 
***

## Introduction 
In 2012, the discovery of the Higgs Boson in the CMS and ATLAS detectors of the Large Hadron Collider (LHC) at CERN in Geneva, Switzerland was a significant milestone in the history of physics. However, despite the validation it provided to the Standard Model (SM), there are still numerous questions in physics that the SM is yet to answer. One approach to uncover some of these mysteries is to study the Higgs Boson in detail, as numerous Beyond Standard Model physics rely on precise measurements of the rate that the Higgs Boson decays to other particles.

The LHC collides protons together at high energy and at a high rate. Each proton collision produces other particles, and, a small fraction of the time, a Higgs boson is produced. This Higgs boson then decays to other particles which can be detected in detectors such as ATLAS and CMS. In this challenge we particularly focus on the Higgs decay to two $\tau$ particles which themselves further decay into other particles before being detected. Measuring this “signal” of interest is complicated by the presence of “background” decays which can produce the same detector signatures. The aim of this challenge is to determine methods to improve the “signal strength” of the Higgs boson decay ($\mu$), which is defined below, based on feature variables that are derived from simulated measurements made in the ATLAS detector. This builds on a previous Kaggle challenge but adds a larger dataset as well as requirements on determining both $\mu$ and the uncertainty on the determination in the realistic scenario where there are additional systematic uncertainties that shift measurements of certain features. Technical details on the signal and background decay channels and features of the dataset are given below, but participants not familiar with particle physics can consider any of these simply as input feature variables for their proposed method except the EventId, Weight and Label (which is essentially the target).

## Problem Setting
***
The objective of this challenge is to improve the signal strength of the Higgs boson decay, specifically in the $H \rightarrow \tau \tau$ decay mode and in the 1 lepton 1 hadron decay channel and in the presence of the background $Z \rightarrow \tau \tau$ process. Additionally, there are systematic uncertainties, specifically in this dataset applied to the Tau Energy Scale (TES).

**The following are the features in this data set.**


|     | Variable                     | Description                                                                                       |
| --- | ---------------------------- | ------------------------------------------------------------------------------------------------- |
| 1.  | EventId                      | A unique integer identifier of the event. {should **NOT** be used for training} |
| 2.  | DER_mass_MMC                 | The estimated mass $m_H$ of the Higgs boson candidate, obtained through a probabilistic phase space integration. |
| 3.  | DER_mass_transverse_met_lep  | The transverse mass between the missing transverse energy and the lepton.                         |
| 4.  | DER_mass_vis                 | The invariant mass of the hadronic tau and the lepton.                                           |
| 5.  | DER_pt_h                     | The modulus of the vector sum of the transverse momentum of the hadronic tau, the lepton and the missing transverse energy vector. |
| 6.  | DER_deltaeta_jet_jet         | The absolute value of the pseudorapidity separation between the two jets (undefined if PRI_jet_num ≤ 1). |
| 7.  | DER_mass_jet_jet             | The invariant mass of the two jets (undefined if PRI_jet_num ≤ 1).                                |
| 8.  | DER_prodeta_jet_jet          | The product of the pseudorapidities of the two jets (undefined if PRI_jet_num ≤ 1).              |
| 9.  | DER_deltar_had_lep           | The R separation between the hadronic tau and the lepton.                                        |
| 10. | DER_pt_tot                   | The modulus of the vector sum of the missing transverse momenta and the transverse momenta of the hadronic tau, the lepton, the leading jet (if PRI_jet_num ≥ 1) and the subleading jet (if PRI_jet_num = 2) (but not of any additional jets). |
| 11. | DER_sum_pt                   | The sum of the moduli of the transverse momenta of the hadronic tau, the lepton, the leading jet (if PRI_jet_num ≥ 1) and the subleading jet (if PRI_jet_num = 2) and the other jets (if PRI_jet_num = 3). |
| 12. | DER_pt_ratio_lep_tau         | The ratio of the transverse momenta of the lepton and the hadronic tau.                           |
| 13. | DER_met_phi_centrality       | The centrality of the azimuthal angle of the missing transverse energy vector w.r.t. the hadronic tau and the lepton. |
| 14. | DER_lep_eta_centrality       | The centrality of the pseudorapidity of the lepton w.r.t. the two jets (undefined if PRI_jet_num ≤ 1). |
| 15. | Eletron Flag                 | 1 if it is an electron 0 if it not.                                                               |
| 16. | muon Flag                    | 1 if it is a muon 0 if it not.                                                                   |
| 17. | PRI_had_pt                   | The transverse momentum $\sqrt{{p_x}^2 + {p_x}^2}$ of the hadronic tau.                          |
| 18. | PRI_had_eta                  | The pseudorapidity $\eta$ of the hadronic tau.                                                    |
| 19. | PRI_had_phi                  | The azimuth angle $\phi$ of the hadronic tau.                                                     |
| 20. | PRI_lep_pt                   | The transverse momentum $\sqrt{{p_x}^2 + {p_x}^2}$ of the lepton (electron or muon).             |
| 21. | PRI_lep_eta                  | The pseudorapidity $\eta$ of the lepton.                                                           |
| 22. | PRI_lep_phi                  | The azimuth angle $\phi$ of the lepton.                                                            |
| 23. | PRI_met                      | The missing transverse energy $\overrightarrow{E}^{miss}_{T}$.                                    |
| 24. | PRI_met_phi                  | The azimuth angle $\phi$ of the missing transverse energy.                                        |
| 25. | PRI_jet_num                  | The number of jets (integer with a value of 0, 1, 2 or 3; possible larger values have been capped at 3). |
| 26. | PRI_jet_leading_pt           | The transverse momentum $\sqrt{{p_x}^2 + {p_x}^2}$ of the leading jet, that is the jet with the largest transverse momentum (undefined if PRI_jet_num = 0). |
| 27. | PRI_jet_leading_eta          | The pseudorapidity $\eta$ of the leading jet (undefined if PRI_jet_num = 0).                     |
| 28. | PRI_jet_leading_phi          | The azimuth angle $\phi$ of the leading jet (undefined if PRI_jet_num = 0).                      |
| 29. | PRI_jet_subleading_pt        | The transverse momentum $\sqrt{{p_x}^2 + {p_x}^2}$ of the leading jet, that is, the jet with the second largest transverse momentum (undefined if PRI_jet_num ≤ 1). |
| 30. | PRI_jet_subleading_eta       | The pseudorapidity $\eta$ of the subleading jet (undefined if PRI_jet_num ≤ 1).                  |
| 31. | PRI_jet_subleading_phi       | The azimuth angle $\phi$ of the subleading jet (undefined if PRI_jet_num ≤ 1).                   |
| 32. | PRI_jet_all_pt               | The scalar sum of the transverse momentum of all the jets of the events.                          |
| 32. | Weight                       | The event weight $w_i$.  {should **NOT** be used for training}                                                                         |
| 33. | Label                        | The event label $y_i \in \{1,0\}$ (1 for signal, 0 for background). {should **NOT** be used for training}                               |


The Large Hadron Collider (LHC) produces high energy events which can be modeled as a Poisson process with an expected arrival rate of $\nu$.Among these high-energy events, physicists search for interesting events that may provide insights into new discoveries. However, most of the events observed are "background" events that are expected and unrelated to new physics. The expected arrival rate of background events, denoted as $\beta$, is well characterized from past experiments and simulations. On the other hand, the arrival rate of signal events, denoted as $\mu \gamma$, is to be measured. Here, $\gamma$ represents the estimated arrival rate of signal events obtained from the Standard Model (SM), which is known to be incomplete. The factor $\mu$ represents the "signal strength" that must be evaluated from the new experiment. All three arrival processes (high energy events, signal, and background) are assumed to be Poisson processes, and their arrival rate expectations are related by the equation $\nu = \beta + \mu \gamma$.

## Estimation of $\mu$
***
### **Counting Estimator**
A simple maximum likelihood estimator based on the Poisson counting process can be used as a baseline given as:  
$\Large \hat{\mu} = \frac{N - \beta}{\gamma} ~~~~~~~~ (1)$  
where $N \sim Poisson(\nu)$ is a random variable: the observed number of high-energy events in an experiment.

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

In the presence of weights, the number of events is given by the sum of weights

$ w_{pseudo} = Poisson(w_i)$, where $w_i$ the weights of event i 

$N_{pseudo} = \sum_{i = 0}^l w_{pseudo}$ where l is the number of elements in one pseudo dataset

$N_{ROI_{BS}} = \frac{\sum_{i = 0}^m N_{pseudo}}{m} $  where m is the number of pseudo datasets

$\sigma_{ROI_{BS}} =  \frac{\sum_{i = 0}^m (N_{pseudo} - N_{ROI_{BS}} )^2}{m}$

$\gamma_{ROI} = \sum_{i = 0}^n w_{eval}  \forall i \in {S} $ number of signal events in ROI in evaluation set

$\beta_{ROI} = \sum_{i = 0}^n w_{eval}   \forall i \in {B} $ number of Background events in ROI in evaluation set

$\Large \hat{\mu} =  \frac{N_{ROI_{BS}} - \beta_{ROI}}{\gamma_{ROI}}   $

$\Delta \hat{\mu} = |\frac{\sigma_{ROI_{BS}} - \beta_{ROI}}{\gamma_{ROI}}|$

This estimator has variance:

$\Large \sigma^2_{\hat{\mu}} = \frac{\nu_{ROI}}{\gamma_{ROI}^2} $

which is lower than that of the plain Poisson counting process, if and only if  $\gamma_{ROI}$. $\gamma_{ROI} / \nu_{ROI}$ > $\gamma$ . $\gamma / \nu$. We see that $\gamma_{ROI} / \nu_{ROI}$ > $\gamma / \nu$ is NOT a sufficient condition to lower the variance of the estimator of $\mu$. There is a tradeoff between increasing $\gamma_{ROI} / \nu_{ROI}$ and not decreasing $\gamma_{ROI}$ too much, that is going into regions "enriched" in signal, but in which the total number of signal events approaches 0.

Here $\gamma_{ROI}$ and $\beta_{ROI}$ are NOT assumed to be known constants (like in the histogram method); they need to be estimated with the simulator, and, likewise, could be plagued with systematic error. Thus, in the presence of systematics, this simple estimator underestimates the variance of $\hat{\mu}$. **This is the problem we want to solve.**



## How to join this challenge?
***
- Go to the "Starting Kit" tab
- Download the "Dummy sample submission" or "sample submission"
- Go to the "My Submissions" tab
- Submit the downloaded file


## Credits
***
#### Lawrence Berkeley National Laboratory 
- Benjamin Nachman
- Ben Thorne
- Chris Harris
- Sascha Diefenbacher
- Steven Farrell
- Wahid Bhimji

#### University of Washington
- Elham e Khoda
- Shih-Chieh Hsu

#### ChaLearn
- Isabelle Guyon
- Ihsan Ullah

#### Université Paris-Saclay
- David Rousseau
- Ragansu Chakkappai

#### UC Irvine
- Aishik Ghosh
