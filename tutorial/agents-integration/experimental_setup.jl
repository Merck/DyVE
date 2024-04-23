# This is taken from https://github.com/Merck/CEEDesigns.jl/blob/34588ae0e5563cb93f6818e3a9c8b3a77c5e3c47/tutorials/SimpleGenerative.jl

make_friedman3 = function (U, noise = 0, friedman3 = true)
    size(U, 2) == 4 || error("input U must have 4 columns, has $(size(U,2))")
    n = size(U, 1)
    X = DataFrame(zeros(Float64, n, 4), :auto)
    for i = 1:4
        X[:, i] .= U[:, i]
    end
    ϵ = noise > 0 ? rand(Distributions.Normal(0, noise), size(X, 1)) : 0
    if friedman3
        X.y = @. atan((X[:, 2] * X[:, 3] - 1 / (X[:, 2] * X[:, 4])) / X[:, 1]) + ϵ
    else
        ## Friedman #2
        X.y = @. (X[:, 1]^2 + (X[:, 2] * X[:, 3] - 1 / (X[:, 2] * X[:, 4]))^2)^0.5 + ϵ
    end
    return X
end

p12, p13, p14, p23, p24, p34 = 0.8, 0.5, 0.3, 0.5, 0.25, 0.4
Σ = [
    1 p12 p13 p14
    p12 1 p23 p24
    p13 p23 1 p34
    p14 p24 p34 1
]

X1 = Distributions.Uniform(0, 100)
X2 = Distributions.Uniform(40 * π, 560 * π)
X3 = Distributions.Uniform(0, 1)
X4 = Distributions.Uniform(1, 11)

C = GaussianCopula(Σ)
D = SklarDist(C, (X1, X2, X3, X4))

X = rand(D, 1000)

data = make_friedman3(transpose(X), 0.01)

data[1:10, :]

# We can check that the empirical correlation is roughly the same as the specified theoretical values: 

cor(Matrix(data[:, Not(:y)]))

# We ensure that our algorithms know that we have provided data of specified types. 

types = Dict(
    :x1 => ScientificTypes.Continuous,
    :x2 => ScientificTypes.Continuous,
    :x3 => ScientificTypes.Continuous,
    :x4 => ScientificTypes.Continuous,
    :y => ScientificTypes.Continuous,
)

data = coerce(data, types);
