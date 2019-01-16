include("./src/GBDTsfuzzy.jl")

using .GBDTsfuzzy
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


grammar_fuzzy = @grammar begin
    b = Gf(bvec) | Ff(bvec) | Gf(impliesf(bvec,bvec))
    bvec = andf(bvec, bvec)
    bvec = orf(bvec, bvec)
    bvec = notf(bvec)
    bvec = ltf(rvec, rvec, tr)
    bvec = ltef(rvec, rvec, tr)
    bvec = gtf(rvec, rvec, tr)
    bvec = gtef(rvec, rvec, tr)
    bvec = f_ltf(x, xid, v, vid, tr)
    bvec = f_ltef(x, xid, v, vid, tr)
    bvec = f_gtf(x, xid, v, vid, tr)
    bvec = f_gtef(x, xid, v, vid, tr)
    rvec = x[xid]
    xid = |([:altitude,:roll,:vx,:home_latitude,:vz,:yaw,:groundspeed,:longitude,:home_longitude,:pitch,:vy,:latitude,:time_offset,:airspeed])
    vid = |(1:10)
    tr = 1.0
end

Gf(v) = minimum(v)                                                #globally
Ff(v) = maximum(v)                                                #eventually
f_ltf(x, xid, v, vid, tr) = ltf(x[xid], ones(length(x[xid])) * v[xid][vid], tr)               #feature is less than a constant
f_ltef(x, xid, v, vid, tr) = ltef(x[xid], ones(length(x[xid])) * v[xid][vid], tr)             #feature is less than or equal to a constant
f_gtf(x, xid, v, vid, tr) = gtf(x[xid], ones(length(x[xid])) * v[xid][vid], tr)               #feature is greater than a constant
f_gtef(x, xid, v, vid, tr) = gtef(x[xid], ones(length(x[xid])) * v[xid][vid], tr)             #feature is greater than or equal to a constant


#workarounds for slow dot operators:
impliesf(v1, v2) = (a = similar(v1); a .= orf(notf(v1), v2))         #implies
notf(v) = (a = similar(v); a .= 1 .- v)                          #not
andf(v1, v2) = (a = similar(v1); a .= [min(v1[i], v2[i]) for i in 1:length(v1)])               #and
orf(v1, v2) = (a = similar(v1); a .= [max(v1[i], v2[i]) for i in 1:length(v1)])                #or
ltf(x1, x2, tr) = (a = Vector{Float64}(undef,length(x1)); a .= [x1[i] < x2[i] ? 1.0 : x1[i] < x2[i] + tr ? 0.5 : 0.0 for i in 1:length(x1)])   #less than
ltef(x1, x2, tr) = (a = Vector{Float64}(undef,length(x1)); a .= [x1[i] <= x2[i] ? 1.0 : x1[i] <= x2[i] + tr ? 0.5 : 0.0 for i in 1:length(x1)])
gtf(x1, x2, tr) = (a = Vector{Float64}(undef,length(x1)); a .= [x1[i] > x2[i] ? 1.0 : x1[i] > x2[i] - tr ? 0.5 : 0.0 for i in 1:length(x1)])   #less than
gtef(x1, x2, tr) = (a = Vector{Float64}(undef,length(x1)); a .= [x1[i] >= x2[i] ? 1.0 : x1[i] >= x2[i] - tr ? 0.5 : 0.0 for i in 1:length(x1)])   #less than

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
    model = induce_tree(grammar_fuzzy, :b, p, X, y, depth, afsoon_loss);
end
println("Finished learning")
println("Show")
show(model)
println(v)
println("Done show")


println("Storing model")
JLD.save(joinpath(output_dir, "model.jld"), "model", model, "v", v)
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


