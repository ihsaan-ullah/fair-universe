# Overview

High energy physicists at CERN use simulations in order to reproduce collisions that occurs in the [Large Hadron Collider](https://www.home.cern/science/accelerators/large-hadron-collider) (LHC). Number of particles created in a single collision can range from a few to several hundred. Once they developped a theory which predicts the existence of a new particle, physicists run simulations and seek evidence of new particle. To do so, they classify all the particles resulting from a collision between background particles (uninteresting ones that they already know) and signal particles (the ones they are interested in). This is why high-energy physicists are working increasingly closely with machine learning scientists.

To perform this classifications task there are tens of available features about each particle (such as momentum, energy, direction). Nevertheless, simulations are susceptible to systematic biases, as are the data used for classification, which adds to the complexity of the task. As a result, a major challenge is to mitigate these biases from the data to enhance the accuracy of classification.

The Fair Universe challenge is a toy-exemple for this problem. nstead of working in a high-dimensional feature space, we consider 2D points (2-dimensional feature vectors) that belongs either to signal or background class.  The aim is to build a model that classifies them correctly.
***
More formaly, this challenge uses 2-features and 2-class datasets

The datasets conist of some points (or events) features:
1. `x1` for x_1-coordinate
2. `x2` for x_2-coordinate

The `y` column shows the class of the point : either 1 for signal or 0 for background. 

<img src="https://raw.githubusercontent.com/ihsaan-ullah/fair-universe/master/Competition_Bundle/pages/train_test_plots.png" width=800 > 

In training data, signal and background clusters are generated with Gaussian distribution. In test data, there is a "distribution shift": the center of both clusters may be slightly translated and the variance of the Gaussian distributions may be modified too. But you do not know by how much.
	
***
	
The goal is to overcome this nuisance and classify the points in there correct class in the presence of bias.

Note that you get all the test points at once, but without class labels. This gives you a chance to estimate the distribution shift.




### Task
The task of the this competition is classificaiton of signal vs background where signal is an event (creation of a particle of interest) and background consists of all other events.


### Installation
In order to run this bundle locally, you need to install the following packages:
- pyyaml
- numpy
- pandas

It is recommended to use virtual environments e.g. conda env

### How to Join this challenge:
**Register**  
- Click **"My Submission"** tab in the top Menu
- Accept terms and conditions
- Click **"Register"** button

**Test**  
- To test submissions, dowload [THIS SAMPLE SUBMISSION]
- If all goes well, you should see the score of your submission appear on the “Results" page.
- If you pass this test:

**Get Starting Kit**  
- Click **"Getting Started Kit"** tab in the side menu
- download the starting kit in the “Getting started” tab and start the README.ipynb notebook. Run it 
- Download Starting Kit
- Run `README.ipynb` locally to get familiar with the problem and to create submission
- Submit the zipped submission in **My Submission** tab

**Ready to Contribute**  
- Click the **"Data"** tab in the side menu
- Download the Public Data and replace sample data by it.
- Re-run the notebook and create your own submission.



### Credits
- Isabelle Guyon
- David Rousseau
- Ihsan Ullah
- Mathis Reymond
- Jacobo Ruiz
- Stefano BAVARO
- Shah Fahad HOSSAIN