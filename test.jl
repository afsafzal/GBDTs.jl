include("./src/GBDTsfuzzy.jl")
include("./MyGrammar.jl")

using .GBDTsfuzzy
using .MyGrammar
using ArgParse
using MultivariateTimeSeries
using Printf
import JLD
using Random; Random.seed!(1)


s = ArgParseSettings()
@add_arg_table s begin
    "data"
        help = "path to test data"
        arg_type = String
        required = true
    "model"
        help = "path to model jld file"
        arg_type = String
        required = true
end

args = parse_args(ARGS, s)

println("Reading data")
X, y = read_data_labeled(args["data"])
println("Finished reading")

println("Reading model")

m = JLD.load(args["model"])

v = m["v"]
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
