# Forest-Coverage-Type

Given the Cover Type dataset (file "covtype.data") we try to predict the type of vegetation that covers a forest area based on its characteristics. The dataset is described in the file 'covtype.info'. The first 15120 records are considered to be used for training and the rest for evaluation, whilst the following points are examined : 

A. Construction of a logistic regression model using the "LBFGS" solution algorithm with a maximum number of 10000 iterations, convergence at 10-3, normalization 𝐿2 and weight 𝐶=1.0. Evaluation of the model's accuracy. The algorithm takes a long time to converge, so set the parameter verbose=1 to monitor its progress. Repeat the process with different solving algorithms and parameter choices of the LogisticRegression function of scikit-learn. Potential insight? Comment on how sensitive the training process is to parameter choice.

B. As in sub-question B, using Linear Discriminant Analysis (without the parameter sensitivity analysis).

C. Compare the results and the speed of convergence of the 2 models and the ease with which models with similar performance were produced in each case. The example is instructive: In algorithms that have a limited theoretical background we often have to experiment without guidance to get a good result (on the other hand, it is often our only choice).

The dataset [1] can be found at : https://archive.ics.uci.edu/ml/datasets/covertype

Acknowledgement : 

[1]. Joao Gama and Ricardo Rocha and Pedro Medas. Accurate decision trees for mining high-speed data streams. KDD. 2003.
