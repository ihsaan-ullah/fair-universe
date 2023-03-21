# Data

### [Download Public Data](https://www.codabench.org/datasets/download/814aa909-c8f2-4308-b119-ea2cb2f49d99/)

We provide sample_data with the starting kit, but to prepare your submission, you must fetch the public_data from the challenge website and point to it.

We mentioned earlier that the value of the nuisance parameter remains unknown. It will be true for the test set : your model will not be provided the value of the nuisance parameter. Thus, we encourage you not to let your model know about the value you use to generate the data you train it on. Moreover, when they run the simulations, physicists don't know the value of the nuisance parameter, and since this challenge is about helping physicists to solve a problem, it's better to come up with a solution that fits their working conditions as much as possible.


train_sets_list is a list of 3 train sets each of which is generated with a different value of the nuisance parameter. The dataset at index :

- 0 was generated with z=0.5
- 1 was generated with z=1.4
- 2 was generated with z=1.8


test_sets_list is a list of 3 test sets each of which is generated with a different value of the nuisance parameter that we don't provide you with.

Each dataset contains two clusters of points, one of which corresponds to the signal points while the other one contains background points. To generate the points of each cluster, we use 2D-Gaussian random distribution.

Just as high energy physicists can run their simulations as many time as they need, a particularity of this challenge is that along with some sample pieces of data, we provide you with python functions that allow you to generate your own data. It means you will be able to introduce the bias into the data with any value you want of the nuisance parameter.

Go to the FILES tab to download the data and the starting kit. The starting kit contains a very small subset of the data for debug purposes. It introduces you to the problem. To prepare a challenge submission, you need to use the large dataset that you download separately or to generate your own data with the function provided. For the test dataset, you do NOT have the labels!

