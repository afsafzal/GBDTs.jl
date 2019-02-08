include("./src/GBDTsfuzzy.jl")
include("./MyGrammar.jl")

using .GBDTsfuzzy
using .MyGrammar
using ArgParse
using MultivariateTimeSeries
using Printf
import JLD
using Random; Random.seed!(1)

println("Ready to accept")
while !eof(stdin)
    input = readline(stdin)
    splitted = split(input)
    data = String(splitted[1])
    model_path = String(splitted[2])
    println("Reading data")
    X, y = read_data_labeled(data)
    println("Finished reading")

    println("Reading model")

    m = JLD.load(model_path)

    println(m)
    global v = m["v"]
    model = m["model"]
    mode = m["mode"]

    println("Model loaded")

    println("Classify")
    if mode == "normal"
        pred = classify(model, X)
    else
        pred = classifyfuzzy(model, X)
    end

    println("Classification done")
    println(pred)

    T = Vector{Float64}(undef, length(X))

    if mode == "normal"
        for i in eachindex(T)
            if pred[i] == model.void_label
                T[i] = 1.0
            else
                T[i] = 0.0
            end
        end
    else
        for i in eachindex(T)
            val = get(pred[i], model.void_label, 0.0)
            T[i] = val
        end
    end

    println("Wrongness:")
    println(T)
    println("Done")
end
