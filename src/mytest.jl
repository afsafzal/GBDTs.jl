include("./GBDTsfuzzy.jl")

using .GBDTsfuzzy
using MultivariateTimeSeries
using Printf
import TikzPictures
import HDF5
import JLD
using Random; Random.seed!(1)

println("Reading data")
X, y = read_data_labeled(joinpath("..", "data", "out"));
println("Finished reading")


grammar = @grammar begin
    b = G(bvec) | F(bvec) | G(implies(bvec,bvec))
    bvec = and(bvec, bvec)
    bvec = or(bvec, bvec)
    bvec = not(bvec)
    bvec = lt(rvec, rvec)
    bvec = lte(rvec, rvec)
    bvec = gt(rvec, rvec)
    bvec = gte(rvec, rvec)
    bvec = f_lt(x, xid, v, vid)
    bvec = f_lte(x, xid, v, vid)
    bvec = f_gt(x, xid, v, vid)
    bvec = f_gte(x, xid, v, vid)
    rvec = x[xid]
    xid = |([:altitude,:roll,:vx,:home_latitude,:vz,:yaw,:groundspeed,:longitude,:home_longitude,:pitch,:vy,:latitude,:time_offset,:airspeed])
    vid = |(1:10)
end

G(v) = all(v)                                                #globally
F(v) = any(v)                                                #eventually
f_lt(x, xid, v, vid) = lt(x[xid], v[xid][vid])               #feature is less than a constant
f_lte(x, xid, v, vid) = lte(x[xid], v[xid][vid])             #feature is less than or equal to a constant
f_gt(x, xid, v, vid) = gt(x[xid], v[xid][vid])               #feature is greater than a constant
f_gte(x, xid, v, vid) = gte(x[xid], v[xid][vid])             #feature is greater than or equal to a constant

#workarounds for slow dot operators:
implies(v1, v2) = (a = similar(v1); a .= v2 .| .!v1)         #implies
not(v) = (a = similar(v); a .= .!v)                          #not
and(v1, v2) = (a = similar(v1); a .= v1 .& v2)               #and
or(v1, v2) = (a = similar(v1); a .= v1 .| v2)                #or
lt(x1, x2) = (a = Vector{Bool}(undef,length(x1)); a .= x1 .< x2)   #less than
lte(x1, x2) = (a = Vector{Bool}(undef,length(x1)); a .= x1 .≤ x2)  #less than or equal to
gt(x1, x2) = (a = Vector{Bool}(undef,length(x1)); a .= x1 .> x2)   #greater than
gte(x1, x2) = (a = Vector{Bool}(undef,length(x1)); a .= x1 .≥ x2)  #greater than or equal to

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
model = induce_tree(grammar, :b, p, X, y, 10, afsoon_loss);
println("Finished learning")
println("Show")
show(model)
println(v)
println("Done show")


println("Storing model")
JLD.save("model.jld", "model", model, "v", v)
println("Done saving")

try
    println("Display")
    t = display(model; edgelabels=false) #suppress edge labels for clarity (left branch is true, right branch is false)
    TikzPictures.save(TikzPictures.PDF("graph"), t)
    println("Done display")
catch y
    println("Display failed")
    println(y)
end


