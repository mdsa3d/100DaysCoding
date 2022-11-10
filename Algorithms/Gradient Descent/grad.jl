# initialised params
w = 0.0
b = 0.0
# random training data
x = rand(10)
w_train = 2
b_train = rand()
y = w_train.*x .+ b_train

learning_rate = 0.01

function descent(x,y,w,b,learning_rate)
    dldw = 0.0
    dldb = 0.0
    global N = size(x,1)
    for (xi,yi) in zip(x,y)
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))
    end
    w = w - learning_rate.*(1/N).*dldw
    b = b - learning_rate.*(1/N).*dldb
    return w, b
end

for epoch in 1:2000
    global w, b = descent(x,y,w,b,learning_rate)
    yhat = w .* x .+ b
    loss = sum((y-yhat).^2)/N
    println("loss = $loss w=$w b=$b")
end
