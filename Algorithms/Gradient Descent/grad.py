# Gradeint descent for linear regression
# basic format of the linear regression function is : "y = wx + b"
# implement gradesc with respect to loss:
# loss = (y - yhat) ** 2 / N
"""
Gradiant descent will try to find values for "w" and "b".
"""
import numpy as np

# initialised parameters
w = 0.0
b = 0.0
# training data
x = np.random.randn(10, 1) # create data using random distribution
w_train = 2
b_train = round(np.random.randn(), 4)
y = w_train*x + b_train

#hyperparameter
learning_rate = 0.01 # how fast our algorithms learns, i.e one step towards solution

# create gradient descent function
def descent(x,y,w,b,learning_rate):
    # initialised partial derivatives
    # loss = (y-yhat)**2 = (y-(wx+b))**2
    # calculate derivatives of loss wrt "b" and loss wrt "w"
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0] # we need average values, hence the len of data
    for xi,yi in zip(x,y):
        # partial derivates:
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))
    # update the parameters
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    return w, b

# iteratively make updates
for epoch in range(400):
    # run gradient descent
    w, b = descent(x,y,w,b,learning_rate)
    yhat = w*x+b
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])
    # print(f'{epoch} loss = {loss} w= {w} b= {b}')
print(f'loss = {round(loss[0], 6)*100} w= {round(w[0],2)} b= {round(b[0],4)}')
print(f'Original constants w:{w_train} b:{b_train}')