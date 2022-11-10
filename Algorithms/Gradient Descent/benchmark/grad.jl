# initialised params
using BenchmarkTools
# random training data
x = rand(10)
w_train = 2
b_train = 3 #rand()
y = w_train.*x .+ b_train

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

function grad_descent(x,y;learning_rate = 0.01)
    w = 0.0
    b = 0.0
    loss = 0.0
    while round(loss, digits=6)*100 >= 0
        w, b = descent(x,y,w,b,learning_rate)
        yhat = w .* x .+ b
        loss = sum((y-yhat).^2)/N
        # println("loss = $loss w=$w b=$b")
        if round(loss, digits=6)*100 == 0
            break
        end
    end
    return loss, w, b
end
loss, w, b = @btime grad_descent(x,y)
print("loss = $(round(loss, digits=6)*100) w = $(round(w, digits=2)) b = $(round(b, digits=2))")

