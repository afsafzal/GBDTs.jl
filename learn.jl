include("./src/GBDTsfuzzy.jl")
include("./MyGrammar.jl")

using .GBDTsfuzzy
using .MyGrammar
using MultivariateTimeSeries
using Printf
import TikzPictures
import JLD
using Random; Random.seed!(1)

if length(ARGS) < 3
    println("Insufficient arguments: <path-to-data> <fuzzy-or-normal> <max-depth-of-tree> [output-dir]")
    throw(Exception)
end

mode = ARGS[2]

@assert mode == "fuzzy" || mode == "normal"

depth = parse(Int64, ARGS[3])
output_dir = "."
if length(ARGS) == 4
    output_dir = ARGS[4]
end

println("Reading data")
#X, y = read_data_labeled(joinpath("..", "data", ARGS[1]));
X, y = read_data_labeled(ARGS[1])
println("Finished reading")


println("Grammar defined")

const v = Dict{Symbol,Vector{Float64}}()
mins, maxes = minimum(X), maximum(X)
for (i,xid) in enumerate(names(X))
    v[xid] = collect(range(mins[i],stop=maxes[i],length=10))
end;

println("Constants are set")

p = MonteCarlo(2000, 5)

println("MonteCarlo")

println("Learning decision tree")
if mode == "normal"
    model = induce_tree(grammar, :b, p, X, y, depth, afsoon_loss);
else
    model = induce_tree(grammar_fuzzy, :b, p, X, y, depth, afsoon_loss_fuzzy);
end
println("Finished learning")
println("Show")
show(model)
println(v)
println("Done show")


println("Storing model")
JLD.save(joinpath(output_dir, "model.jld"), "model", model, "v", v, "mode", mode)
println("Done saving")

try
    println("Display")
    t = display(model; edgelabels=false) #suppress edge labels for clarity (left branch is true, right branch is false)
    cd(output_dir)
    TikzPictures.save(TikzPictures.PDF("graph"), t)
    println("Done display")
catch y
    println("Display failed")
    println(y)
end


