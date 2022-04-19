```
---
title: Labo3
author: Axel Vallon and Nicolas Hungerbühler
date: 18.04.22
---
```

## Introduction 

The work is consisted on applying a methodology to evaluate the performence of a trained neural network on new datas.  In order to chose a final model with :

- the right complexity
	- number of parameters
	- syneptics weights

- configurations 
	- activation function
	- learning rate 
	- momentum rate 
	- training iterations

The neural network will have this configurations :

- 1 hidden layer
- 2 hidden neurons
- tanh as activation function
- learning rate = 0.001
- momentum rate = 0.7
- training dataset = 80% / testing dataset = 20%

These are the different steps sumarrized :

1. split randomly the dataset to define a training and a test subsets.
2. chose how many times we iterate over the training set (number of Epochs)

As the split is random, we need to do it many times (N_SPLITS) where the goal is to calculate the mean of error (MSE).</br>
As the training is done with random weights, we have to train several times, we need to do it many times with different randomly initialized wights (N_INITS).

**MSE** = The Mean Squared Error (MSE)  is perhaps the simplest and most common loss function, often taught in  introductory Machine Learning courses. To calculate the MSE, you **take the difference between your model's predictions and the ground truth,  square it, and average it out across the whole dataset**.

**Code**

```create_dataset(DATASET_SIZE, spread)```</br>
==> Create dataset and we can change the spread to make them superpose in order to make the classification more difficult. 

```split_dataset(dataset, train_test_ratio = 0.8)```</br>
==> For hold-out validation it splits the dataset in *dataset_train* and *dataset_test*. We can chose the size of the proportion by giving the size of the *dataset_train*.

```split_dataset(dataset, n_parts=5)```</br>
==> For k-fold cross-validation it splits the dataset in *n_parts* where the last one is used for 

## 1 Hold-out validation

**Q1**. Determine where do we define all the parameters mentioned above.

We define them as constant in the Jupyter Notebook. 

![](./img/parameters.png)

Observe that we run the evaluation procedure on four different  problems. Each problem is a two-class two-dimensional problem, where the two sets are more and more overlapped (e.g., the synthetic datasets are randomly generated using variances of 0.4, 0.5, 0.6 and 0.7).

We can see that we iterate over ```DATA_PARAMS[4] = [0.4, 0.5, 0.6, 0.7]``` and we put the index in the ```create_dataset``` spread parameter. 

![](./img/create_dataset.png)

**Q2**. What are the cyan and red curves in those plots ? Why are they different ?

Cyan is for the training dataset and red is for the testing dataset. 

![](./img/cian_red.png)

We need to see the difference to have an idea of the MSE in both dataset. Both dataset should have a good MSE. We need all the informations possible to chose the right model. 

**Q3**. What happens with the training and test errors (MSE) when we have the two sets more overlapped ?

The mean square error is getting bigger for both dataset as the spread is getting bigger as well. This is logical behavior because as the datas are closer it gets harder to classify them correctly.

**Q4**. Why sometimes the red curves indicate a higher error than the cyan ones ?

The red one is the testing dataset which has only 20% of the datas in our case. As it has less datas it is more prone to have different results. This is why sometimes the red one has hiher error. We tested it ourself and we saw that the difference can be quite big. For exemple with 2 different tries :

![](./img/error1.png) ![](./img/error2.png)

**Q5**. What is showing the boxplot summarizing the validation errors of the preceding experiments ?

It shows the distribution of the MSE for each spread value for the last epoch of the test dataset. 

## K-fold cross-validation

**Q1**. Determine where do we define all the above mentioned parameters.

The same as in the hold-out validation the parameters are constant in the jupyter notebook. 

![](./img/configuration2.png)

**Q2**. What is the difference between hold-out and cross-validation ? What is the new parameter that has to be defined for  cross-validation ?

We saw in the parameter that we used the parameter **K = 5**, this parameter is used to split the the dataset in cross-validation. We will divide the dataset in **K** parts where **K-1** will be used for testing and the rest for the training. The 80%/20% separation is kept like in the cross-validation method (5 - 1 = 4, 100/5 * 4 = 80%). In cross-validation we used the parameter **TRAIN_TEST_RATIO = 0.8** to split the data in two parts only.

For cross-validation we do not test with different initialization where in hold-out we tested it **N_INITS = 2** times and the number of *EPOCHS* is different as well. 

**Q3**. Observe the boxplots summarizing the validation errors obtained using the cross-validation method and compare them with  those obtained by hold-out validation

In the cross-validation we can observe that each partition generates values of model error which are closer than the results found when using hold-out validation.

![cross-validation](./img/box2.png)

![](./img/box1.png)

## Speaker recognition experiments

### Man vs woman - natural

To train our model we use the mean features. Standard deviation tells you how spread out the data is, it tells us how different the voice are. which doesn't interest us. We even tested it and the MSE was way too big and it needed more epochs and neurons to start being viable. That's why we ended up by using the mean feature to characterize the MFCC values.

We have 36 women and 36 man in our dataset. We fixed the learning rate to `0.001` and the momentum term to `0.5`. These values are just initial guesses which work well in most of the cases so we did not change them. We first explored the number of epochs to estimate how many iterations of the backpropagation algorithm are enough to find a low training error. This is the result :

**Number of epochs**![](img/test1.png)

Between 2 neurons to 16 neurones the training error does improve a little bit and between 16 to 32 neurons it slightly improves. The result of this test indicates that the minimum number of iterations needed are around 16. 

We explored further by testing the complexity of the model (number of hidden neurons) :

**Number of hidden neurons**![](./img/test2.png)

On these graphs we can even see that after 15 neurons there is some overfitting. 

**Number of hidden neurons**![](./img/training1.png)

![](./img/test3.png)

With this different visualization it is easier to make a choice about the number of epochs and the number of neurons. We made some test as well and we finished with 60 epochs and 15 neurons. (red square)

![](./img/conf_mat1.png)

![](./img/F1_score_1.png)

We first calculate the F1 score for both man and woman and then make a mean of them two. The resuts are quite accurate as we have an F1 score of 0.97.

### Man vs woman . natural and synthetic

Epoch = 90
Neurons = 10

![](./img/neuron2.png)

![](./img/hidden_neuron2.png)

![](./img/training3.png)

![](./img/testing3.png)

![](./img/conf_mat2.png)

![](./img/F1_score_2.png)



###  Man vs woman vs children - natural

Epoch = 80
Neurons = 10

![](./img/neurons3.png)

![](./img/hidden_neurons3.png)

![](./img/training5.png)

![](./img/testing5.png)

![](./img/conf_matrix3.png)

![](./img/f1_score3.png)

### Custom

