module MyGrammar

export
	grammar,
	grammar_fuzzy,
        G,
        Gf,
        F,
        Ff,
        f_lt,
        f_ltf,
        f_lte,
        f_ltef,
        f_gt,
        f_gtf,
        f_gte,
        f_gtef,
        f_e,
        f_ef,
        implies,
        impliesf,
        not,
        notf,
        and,
        andf,
        or,
        orf,
        lt,
        ltf,
        lte,
        ltef,
        gt,
        gtf,
        gte,
        gtef,
        e,
        ef,
        add_xid

using ExprRules

grammar = @grammar begin
    b = Gf(bvec) | Ff(bvec) | Gf(impliesf(bvec,bvec))
    bvec = andf(bvec, bvec)
    bvec = orf(bvec, bvec)
    bvec = notf(bvec)
    bvec = lt(rvec, rvec)
#    bvec = lte(rvec, rvec)
    bvec = gt(rvec, rvec)
#    bvec = gte(rvec, rvec)
    bvec = e(rvec, rvec)
    bvec = f_lt(x, xid, v, vid)
#    bvec = f_lte(x, xid, v, vid)
    bvec = f_gt(x, xid, v, vid)
#    bvec = f_gte(x, xid, v, vid)
    bvec = f_e(x, xid, v, vid)
    rvec = x[xid]
    xid = |([:altitude,:roll,:vx,:home_latitude,:vz,:yaw,:groundspeed,:longitude,:home_longitude,:pitch,:vy,:latitude,:time_offset,:airspeed])
    vid = |(1:3)
end

G(v) = all(v)                                                #globally
F(v) = any(v)                                                #eventually
f_lt(x, xid, v, vid) = lt(x[xid], v[xid][vid])               #feature is less than a constant
f_lte(x, xid, v, vid) = lte(x[xid], v[xid][vid])             #feature is less than or equal to a constant
f_gt(x, xid, v, vid) = gt(x[xid], v[xid][vid])               #feature is greater than a constant
f_gte(x, xid, v, vid) = gte(x[xid], v[xid][vid])             #feature is greater than or equal to a constant
f_e(x, xid, v, vid) = e(x[xid], v[xid][vid])

#workarounds for slow dot operators:
implies(v1, v2) = (a = similar(v1); a .= v2 .| .!v1)         #implies
not(v) = (a = similar(v); a .= .!v)                          #not
and(v1, v2) = (a = similar(v1); a .= v1 .& v2)               #and
or(v1, v2) = (a = similar(v1); a .= v1 .| v2)                #or
lt(x1, x2) = (a = Vector{Float64}(undef,length(x1)); a .= x1 .< x2)   #less than
lte(x1, x2) = (a = Vector{Float64}(undef,length(x1)); a .= x1 .≤ x2)  #less than or equal to
gt(x1, x2) = (a = Vector{Float64}(undef,length(x1)); a .= x1 .> x2)   #greater than
gte(x1, x2) = (a = Vector{Float64}(undef,length(x1)); a .= x1 .≥ x2)  #greater than or equal to
e(x1, x2) = (a = Vector{Float64}(undef,length(x1)); a .= x1 .== x2)


grammar_fuzzy = @grammar begin
    b = Gf(bvec) | Ff(bvec) | Gf(impliesf(bvec,bvec))
    bvec = andf(bvec, bvec)
    bvec = orf(bvec, bvec)
    bvec = notf(bvec)
    bvec = ltf(rvec, rvec, tr)
#    bvec = ltef(rvec, rvec, tr)
    bvec = gtf(rvec, rvec, tr)
#    bvec = gtef(rvec, rvec, tr)
    bvec = ef(rvec, rvec, tr_s)
    bvec = f_ltf(x, xid, v, vid, tr)
#    bvec = f_ltef(x, xid, v, vid, tr)
    bvec = f_gtf(x, xid, v, vid, tr)
#    bvec = f_gtef(x, xid, v, vid, tr)
    bvec = f_ef(x, xid, v, vid, tr_s)
    rvec = x[xid]
    xid = |([:altitude,:roll,:vx,:home_latitude,:vz,:yaw,:groundspeed,:longitude,:home_longitude,:pitch,:vy,:latitude,:time_offset,:airspeed])
    vid = |(1:3)
    tr = 1.0
    tr_s = 0.3
end

Gf(v) = minimum(v)                                                #globally
Ff(v) = maximum(v)                                                #eventually
f_ltf(x, xid, v, vid, tr) = ltf(x[xid], ones(length(x[xid])) * v[xid][vid], tr)               #feature is less than a constant
f_ltef(x, xid, v, vid, tr) = ltef(x[xid], ones(length(x[xid])) * v[xid][vid], tr)             #feature is less than or equal to a constant
f_gtf(x, xid, v, vid, tr) = gtf(x[xid], ones(length(x[xid])) * v[xid][vid], tr)               #feature is greater than a constant
f_gtef(x, xid, v, vid, tr) = gtef(x[xid], ones(length(x[xid])) * v[xid][vid], tr)             #feature is greater than or equal to a constant
f_ef(x, xid, v, vid, tr) = ef(x[xid], ones(length(x[xid])) * v[xid][vid], tr)


#workarounds for slow dot operators:
impliesf(v1, v2) = (orf(notf(v1), v2))         #implies
notf(v) = (1 .- v)                          #not
andf(v1, v2) = (map((x, y) -> min(x, y), v1, v2))               #and
orf(v1, v2) = (map( (x, y) -> max(x, y), v1, v2))                #or
ltf(x1, x2, tr) = (map((x, y) -> x < y - tr ? 1.0 : x < y + tr ? 0.5 + (y-x)/2*tr : 0.0, x1, x2))   #less than
ltef(x1, x2, tr) = (map((x, y) -> x <= y - tr ? 1.0 : x <= y + tr ? 0.5 + (y-x)/2*tr : 0.0, x1, x2))
gtf(x1, x2, tr) = (map((x, y) -> x > y + tr ? 1.0 : x > y - tr ? 0.5 + (x-y)/2*tr : 0.0, x1, x2))   #less than
gtef(x1, x2, tr) = (map((x, y) -> x >= y + tr ? 1.0 : x >= y - tr ? 0.5 + (x-y)/2*tr : 0.0, x1, x2))   #less than
ef(x1, x2, tr) = map(a -> abs(a) < tr ? 1.0 : 0.0, x1 .- x2)


function add_xid(g::Grammar, sym::Symbol)
    push!(g.isterminal, true)
    push!(g.iseval, false)
    push!(g.rules, QuoteNode(sym))
    push!(g.types, :xid)
    push!(g.childtypes, [])
    push!(g.bytype[:xid], length(g.types))
end

end
