include("./src/GBDTsfuzzy.jl")
include("./MyGrammar.jl")

using .GBDTsfuzzy
using .MyGrammar
using ArgParse
using MultivariateTimeSeries
using Printf
import TikzPictures
import JLD
using Random; Random.seed!(1)

s = ArgParseSettings()
@add_arg_table s begin
    "data"
        help = "path to data"
        required = true
        arg_type = String
    "--fuzzy"
        help = "run in fuzzy more"
        action = :store_true
    "depth"
        help = "maximum depth of the tree"
        arg_type = Int
        required = true
    "--output_dir"
        help = "output directory"
        arg_type = String
        default = "."
    "--name"
        help = "name of the model"
        arg_type = String
        default = "model"
    "--seed"
        help = "random seed"
        arg_type = Int
        default = 0
end
args = parse_args(ARGS, s)

Random.seed!(args["seed"])

mode = "normal"
if args["fuzzy"]
    mode = "fuzzy"
end


println("Reading data")
X, y = read_data_labeled(args["data"])
println("Finished reading")

g = grammar
if mode == "fuzzy"
    g = grammar_fuzzy
end


for xid in names(X)
    if startswith(string(xid), "p_")
        add_xid(g, xid)
    end
end

println("Grammar defined")

const v = Dict{Symbol,Vector{Float64}}()
mins, maxes = minimum(X), maximum(X)
for (i,xid) in enumerate(names(X))
    v[xid] = collect(range(mins[i],stop=maxes[i],length=3))
end;

println("Constants are set")

p = MonteCarlo(2000, 5)

println("MonteCarlo")

println("Learning decision tree")
#if mode == "normal"
#    model = induce_tree(g, :b, p, X, y, args["depth"], afsoon_loss);
#else
model = induce_tree(g, :b, p, X, y, args["depth"], afsoon_loss_fuzzy);
#end
println("Finished learning")
println("Show")
show(model)
println(v)
println("Done show")


println("Storing model")
JLD.save(joinpath(args["output_dir"], string(args["name"], ".jld")), "model", model, "v", v, "mode", mode)
println("Done saving")

try
    println("Display")
    t = display(model; edgelabels=false) #suppress edge labels for clarity (left branch is true, right branch is false)
    cd(args["output_dir"])
    TikzPictures.save(TikzPictures.PDF(args["name"]), t)
    println("Done display")
catch y
    println("Display failed")
    println(y)
end


