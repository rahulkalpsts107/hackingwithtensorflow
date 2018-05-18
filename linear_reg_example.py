from __future__ import absolute_import, division, print_function

#http://tflearn.org/models/dnn/
import tflearn

""" Linear Regression Example """
""" As per http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm
Linear regression attempts to model the relationship between two variables
by fitting a linear equation to observed data.
The variables need to have so some sort of association even if they dont literally
mean. Having no association will yield no result in linear regression.

In Linear Regression the equation used to fit the variables is
Y = aX + B
where Y is the dependent variable and X is the explanatory variable
B is the intercept and a is the slope
"""


# Regression data
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

#tflearn provides api's that are easy to work with layers

#input_date is a layer to input data to the network
input_ = tflearn.input_data(shape=[None])

#single_unit - a single unit (Linear) layer
linear = tflearn.single_unit(input_)

#estimator layer
#an estimator provides a way to define a loss function
#based on the loss function the layer can provide a way for the net
#to learn weights as it aims to minimize loss.
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)

#deep neural net model
#pass network as a parameter
m = tflearn.DNN(regression)

#run the model
m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

print("\nRegression result:")

print("Y = " + str(m.get_weights(linear.W)) +
      "*X + " + str(m.get_weights(linear.b)))

print("\nTest prediction for x = 3.2, 3.3, 3.4:")
print(m.predict([3.2, 3.3, 3.4]))