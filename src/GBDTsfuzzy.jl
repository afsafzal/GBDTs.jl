"""
    GBDTs

Grammar-Based Decision Tree (GBDT) is a machine learning model that can be used for the interpretable classification and categorization of heterogeneous multivariate time series data.
"""
module GBDTsfuzzy

export 
        GBDTNode, 
        GBDT,
        induce_tree, 
        partition,
        members_by_bool,
        classify,
        classifyfuzzy,
        node_members,
        children_id,
        id,
        label,
        gbes_result,
        isleaf,
        children,
        gini,
        gini_loss,
        afsoon_loss,
        afsoon_loss_fuzzy,
        print_rules

using Discretizers
using Reexport
using StatsBase
using TikzGraphs, LightGraphs
@reexport using AbstractTrees
@reexport using ExprRules
@reexport using ExprOptimization

NEGLIGIBLE = 0.1

"""
    GBDTNode

Node object of a GBDT.
"""
struct GBDTNode
    id::Int
    label::Int
    gbes_result::Union{Nothing,ExprOptResult}
    children::Vector{GBDTNode}
end
function GBDTNode(id::Int, label::Int) 
    GBDTNode(id, label, nothing, GBDTNode[])
end

"""
    GBDT

GBDT model produced by induce_tree.
"""
struct GBDT
    tree::GBDTNode
    catdisc::Union{Nothing,CategoricalDiscretizer}
    void_label::Union{Nothing,Int}
end

"""
    Counter

Mutable int counter
"""
mutable struct Counter
    i::Int
end

"""
    id(node::GBDTNode)

Returns the node id.
"""
id(node::GBDTNode) = node.id
"""
    label(node::GBDTNode)

Returns the node label.
"""
label(node::GBDTNode) = node.label
"""
    gbes_result(node::GBDTNode) 

Returns the result from GBES.
"""
gbes_result(node::GBDTNode) = node.gbes_result
"""
    isleaf(node::GBDTNode) 

Returns true if node is a leaf.
"""
isleaf(node::GBDTNode) = isempty(node.children)

#AbstractTrees interface
AbstractTrees.children(node::GBDTNode) = node.children
function AbstractTrees.printnode(io::IO, node::GBDTNode) 
    print(io, "$(node.id): label=$(node.label)")
    r = node.gbes_result
    if r != nothing
        print(io, ", loss=$(round(r.loss;digits=2)), $(r.expr)")
    end
end

"""
    ishomogeneous(v::AbstractVector{T}) where T

Returns true if all elements in v are the same.
"""
ishomogeneous(v::AbstractVector{T}) where T = length(unique(v)) == 1

function is_nearly_homogenous(y_truth::AbstractVector{Int}, members::AbstractVector{Float64})
# FIXME
    return length([y_truth[i] for i in findall(x->x > NEGLIGIBLE, members)]) == 1
end

"""
    gini_loss(node::RuleNode, grammar::Grammar, X::AbstractVector{T}, y_truth::AbstractVector{Int}, 
                     members::AbstractVector{Int}, eval_module::Module; 
                     w1::Float64=100.0, 
                     w2::Float64=0.1) where T

Default loss function based on gini impurity and number of nodes in the derivation tree.  
See Lee et al. "Interpretable categorization of heterogeneous time series data"
"""
function gini_loss(node::RuleNode, grammar::Grammar, X::AbstractVector{T}, y_truth::AbstractVector{Int}, 
                     members::AbstractVector{Int}, eval_module::Module, exprs::Array{Expr,1}; 
                     w1::Float64=100.0, 
                     w2::Float64=0.1) where T
    ex = get_executable(node, grammar)
    y_bool = partition(X, members, ex, eval_module)
    members_true, members_false = members_by_bool(members, y_bool)
    return w1*gini(y_truth[members_true], y_truth[members_false]) + w2*length(node)
end
function afsoon_loss(node::RuleNode, grammar::Grammar, X::AbstractVector{T}, y_truth::AbstractVector{Int},
                        members::AbstractVector{Int}, eval_module::Module, exprs::Array{Expr,1};
                        w1::Float64=10.0,
                        w2::Float64=0.1) where T
    ex = get_executable(node, grammar)
    if ex in exprs
        return 1000000.0
    end
    y_bool = partition(X, members, ex, eval_modul)
    members_true, members_false = members_by_bool(members, y_bool)
    if ishomogeneous(y_truth[members])
        return w1*length(members_true)*length(members_false) + w2*length(node)
    else
        return w1*gini(y_truth[members_true], y_truth[members_false]) + w2*length(node)
    end
end
function afsoon_loss_fuzzy(node::RuleNode, grammar::Grammar, X::AbstractVector{T}, y_truth::AbstractVector{Int},
                        members::AbstractVector{Float64}, eval_module::Module, exprs::Array{Expr,1};
                        w1::Float64=1000.0,
                        w2::Float64=0.1) where T
    ex = get_executable(node, grammar)
    if ex in exprs
        return 1000000.0
    end
    y_fuzz = partitionfuzzy(X, members, ex, eval_module)
    members_true, members_false = members_by_fuzz(members, y_fuzz)
    if is_nearly_homogenous(y_truth, members)
        return w1*sum(members_true)*sum(members_false) + w2*length(node)
    else
        return w1*gini_afsoon(y_truth, members_true, members_false) + w2*length(node)
    end
end

"""
    induce_tree(grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, X::AbstractVector{XT}, 
                        y::AbstractVector{YT}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main) where {XT,YT}

Learn a GBDT from labeled data.  Categorical labels are converted to integers.
# Arguments:
- `grammar::Grammar`: grammar
- `typ::Symbol`: start symbol
- `p::ExprOptAlgorithm`: Parameters for ExprOptimization algorithm
- `X::AbstractVector{XT}`: Input data features, e.g., a MultivariateTimeSeries
- `y::AbstractVector{YT}`: Input (class) labels.
- `max_depth::Int`: Maximum depth of GBDT.
- `loss::Function`: Loss function.  See gini_loss() for function signature.
- `eval_module::Module`: Module in which expressions are evaluated.
"""
function induce_tree(grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, X::AbstractVector{XT}, 
                        y::AbstractVector{YT}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main; kwargs...) where {XT,YT}
    catdisc = CategoricalDiscretizer(y)
    y_truth = encode(catdisc, y)
    induce_tree(grammar, typ, p, X, y_truth, max_depth, loss, eval_module; 
                catdisc=(catdisc), kwargs...)
end
"""
    induce_tree(grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, X::AbstractVector{T}, 
                        y_truth::AbstractVector{Int}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main; 
                        catdisc::Union{Nothing,CategoricalDiscretizer}=nothing),
                        verbose::Bool=false) where T

Learn a GBDT from labeled data.  
# Arguments:
- `grammar::Grammar`: grammar
- `typ::Symbol`: start symbol
- `p::ExprOptAlgorithm`: Parameters for ExprOptimization algorithm
- `X::AbstractVector{XT}`: Input data features, e.g., a MultivariateTimeSeries
- `y_truth::AbstractVector{Int}`: Input (class) labels.
- `max_depth::Int`: Maximum depth of GBDT.
- `loss::Function`: Loss function.  See gini_loss() for function signature.
- `eval_module::Module`: Module in which expressions are evaluated.
- `catdisc::Union{Nothing,CategoricalDiscretizer}`: Discretizer used for converting the labels.
- `min_members_per_branch::Int`: Minimum number of members for a valid branch.
- `prevent_same_label::Bool`: Prevent split if both branches have the same dominant label 
- `verbose::Bool`: Verbose outputs
"""
function induce_tree(grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, X::AbstractVector{T}, 
                        y_truth::AbstractVector{Int}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main; 
                        catdisc::Union{Nothing,CategoricalDiscretizer}=nothing,
                        min_members_per_branch::Float64=0.0,
                        prevent_same_label::Bool=true,
                        verbose::Bool=false) where T
    verbose && println("Starting...")
    @assert length(X) == length(y_truth)
    members = ones(length(y_truth))
    node_count = Counter(0)
    cluster_count = length(unique(y_truth))
    void_label = cluster_count + 1
    exprs = Expr[]
    node = _split(node_count, grammar, typ, p, X, y_truth, members, max_depth, loss, eval_module, void_label, exprs,
                 min_members_per_branch=min_members_per_branch, 
                 prevent_same_label=prevent_same_label, verbose=verbose)
    if typeof(catdisc) == CategoricalDiscretizer #FIXME
        catdisc.n2d[cluster_count + 1] = cluster_count + 1
        catdisc.d2n[cluster_count + 1] = cluster_count + 1
    end
    return GBDT(node, catdisc, void_label)
end
function _split(node_count::Counter, grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, 
                       X::AbstractVector{T}, y_truth::AbstractVector{Int}, 
                       members::AbstractVector{Float64},
                       d::Int, loss::Function, eval_module::Module, void_label::Int,
                       exprs::Array{Expr,1};
                       min_members_per_branch::Float64=0.0,
                       prevent_same_label::Bool=true,
                       verbose::Bool=false) where T
    println(members)
    flush(stdout)
    id = node_count.i += 1  #assign ids in preorder
    if count(members .> NEGLIGIBLE) <= min_members_per_branch
        return GBDTNode(id, void_label)
    end
    if d == 0
        return GBDTNode(id, best_label(y_truth, members))
    end

    #gbes
    println("before gbes")
    @time gbes_result = optimize(p, grammar, typ, (node,grammar)->loss(node, grammar, X, y_truth, 
        members, eval_module, exprs); verbose=verbose)
    println("after gbes")
    flush(stdout)

    if gbes_result.expr in exprs
        return GBDTNode(id, best_label(y_truth, members))
    end

    y_fuzz = partitionfuzzy(X, members, gbes_result.expr, eval_module)
    members_true, members_false = members_by_fuzz(members, y_fuzz)

    n_true, n_false = count(members_true .> NEGLIGIBLE), count(members_false .> NEGLIGIBLE)

    if is_nearly_homogenous(y_truth, members) && n_true > min_members_per_branch && n_false > min_members_per_branch
        return GBDTNode(id, best_label(y_truth, members))
    end

    if n_true <= min_members_per_branch && n_false <= min_members_per_branch
        return GBDTNode(id, best_label(y_truth, members))
    end

    #don't create split if split doesn't result in two valid groups 
    #if length(members_true) <= min_members_per_branch || length(members_false) <= min_members_per_branch
    #    return GBDTNode(id, mode(y_truth[members]))
    #end


    #don't create split if both sides of the split have the same dominant label
    #if prevent_same_label && (mode(y_truth[members_true]) == mode(y_truth[members_false]))
    #    return GBDTNode(id, mode(y_truth[members]))
    #end

    push!(exprs, gbes_result.expr)

    child_true = _split(node_count, grammar, typ, p, X, y_truth, members_true, d-1, 
        loss, eval_module, void_label, copy(exprs);
        min_members_per_branch=min_members_per_branch,
        prevent_same_label=prevent_same_label, 
        verbose=verbose)
    child_false = _split(node_count, grammar, typ, p, X, y_truth, members_false, d-1, 
        loss, eval_module, void_label, copy(exprs);
        min_members_per_branch=min_members_per_branch,
        prevent_same_label=prevent_same_label, 
        verbose=verbose)

    return GBDTNode(id, best_label(y_truth, members), gbes_result, [child_true, child_false])
end

"""
    partition(X::AbstractVector{T}, members::AbstractVector{Int}, expr, eval_module::Module) where T

Returns a Boolean vector of length members containing the results of evaluating expr on each member.  Expressions are evaluated in eval_module.
"""
function partition(X::AbstractVector{T}, members::AbstractVector{Int}, expr, eval_module::Module) where T
    y_bool = Vector{Bool}(undef, length(members))
    for i in eachindex(members)
        @eval eval_module x = $(X[members[i]])
        y_bool[i] = Core.eval(eval_module, expr) #use x in expression
    end
    y_bool
end
function partitionfuzzy(X::AbstractVector{T}, members::AbstractVector{Float64}, expr, eval_module::Module) where T
    y_fuzz = zeros(length(members))
    @eval eval_module x = Array{Any}(undef, Threads.nthreads())
    @Threads.threads for i in findall(x->x>0.0, members)
        if members[i] < NEGLIGIBLE
            y_fuzz[i] = 0.5
            continue # speedup
        end
#        @eval eval_module x = $(X[i])
        @eval eval_module x[Threads.threadid()] = $(X[i])
        y_fuzz[i] = Core.eval(eval_module, expr) #use x in expression #TODO This takes a long time
    end
    y_fuzz
end

"""
    members_by_bool(members::AbstractVector{Int}, y_bool::AbstractVector{Bool})

Returns a tuple containing the results of splitting members by the Boolean values in y_bool.
"""
function members_by_bool(members::AbstractVector{Int}, y_bool::AbstractVector{Bool})
    @assert length(y_bool) == length(members)
    return members[findall(y_bool)], members[findall(!,y_bool)]
end
function members_by_fuzz(members::AbstractVector{Float64}, y_fuzz::AbstractVector{Float64})
    @assert length(y_fuzz) == length(members)
    return members .* y_fuzz, members .* (1 .- y_fuzz)
end

"""
    gini(v1::AbstractVector{T}, v2::AbstractVector{T}) where T

Returns the gini impurity of v1 and v2 weighted by number of elements.
"""
function gini(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
    N1, N2 = length(v1), length(v2)
    return (N1*gini(v1) + N2*gini(v2)) / (N1+N2)
end
"""
    gini(v::AbstractVector{T}) where T

Returns the Gini impurity of v.  Returns 0.0 if empty.
"""
function gini(v::AbstractVector{T}) where T
    isempty(v) && return 0.0
    return 1.0 - sum(abs2, proportions(v))
end

"""
    gini(v1::AbstractVector{T}, v2::AbstractVector{T}) where T

Returns the gini impurity of v1 and v2 weighted by number of elements.
"""
function gini_afsoon(y_truth::AbstractVector{Int},
                     members1::AbstractVector{Float64}, members2::AbstractVector{Float64})
    N1, N2 = sum(members1), sum(members2)
    return (N1*gini_afsoon(y_truth, members1) + N2*gini_afsoon(y_truth, members2)) / (N1+N2)
end
"""
    gini(v::AbstractVector{T}) where T

Returns the Gini impurity of v.  Returns 0.0 if empty.
"""
function gini_afsoon(y_truth::AbstractVector{Int}, members::AbstractVector{Float64})
    s = sum(members)
    if s < NEGLIGIBLE
        return 0.0
    end
    label_to_score = Dict(y => score(findall(x->x==y, y_truth), members) for y in unique(y_truth))
    v = values(label_to_score)
    return 1.0 - sum(abs2, v ./ s)
end


"""
    Base.length(model::GBDT)

Returns the number of vertices in the GBDT. 
"""
Base.length(model::GBDT) = length(model.tree)
"""
    Base.length(root::GBDTNode)

Returns the number of vertices in the tree rooted at root.
"""
function Base.length(root::GBDTNode)
    retval = 1
    for c in root.children
        retval += length(c)
    end
    return retval
end

Base.show(io::IO, model::GBDT) = Base.show(io::IO, model.tree)
Base.show(io::IO, tree::GBDTNode) = print_tree(io, tree)

function print_rules(node::GBDTNode, void_label::Int, prefix::String)
    if isleaf(node)
        if node.label != void_label
            println(prefix)
        end
        return
    end

    print_rules(node.children[1], void_label, string(prefix, " & ", node.gbes_result.expr))
    print_rules(node.children[2], void_label, string(prefix, " & !", node.gbes_result.expr))
end

function score(indexes::AbstractVector{Int}, members::AbstractVector{Float64})
    return sum(members[indexes])
end

function best_label(y_truth::AbstractVector{Int}, members::AbstractVector{Float64})
    label_to_score = Dict(y => score(findall(x->x==y, y_truth), members) for y in unique(y_truth))
    return findmax(label_to_score)[2]
end

"""
    Base.display(model::GBDT; edgelabels::Bool=false)

Returns a TikzGraphs plot of the tree.  Turn off edgelabels for cleaner plot.  Left branch is true, right branch is false.
"""
function Base.display(model::GBDT; kwargs...)
    display(model.tree, model.catdisc; kwargs...)
end
"""
    Base.display(root::GBDTNode, catdisc::Uniont{Nothing,CategoricalDiscretizer}=nothing;
                     edgelabels::Bool=false)

Returns a TikzGraphs plot of the tree.  Turn off edgelabels for cleaner plot.  Left branch is true, right branch is false.
If catdisc is supplied, use it to decode the labels.
"""
function Base.display(root::GBDTNode, catdisc::Union{Nothing,CategoricalDiscretizer}=nothing;
                     edgelabels::Bool=false)
    n_nodes = length(root)
    g = DiGraph(n_nodes)
    text_labels, edge_labels = Vector{String}(undef,n_nodes), Dict{Tuple{Int,Int},String}() 
    for node in PreOrderDFS(root)
        if node.gbes_result != nothing
            r = node.gbes_result
            text_labels[node.id] = string("$(node.id): $(verbatim(string(r.expr)))")
        else
            label = catdisc != nothing ?  decode(catdisc, node.label) : node.label
            text_labels[node.id] = string("$(node.id): label=$(label)")
        end
        for (i, ch) in enumerate(node.children)
            add_edge!(g, node.id, ch.id)
            edge_labels[(node.id, ch.id)] = i==1 ? "True" : "False"
        end
    end
    if edgelabels
        return TikzGraphs.plot(g, text_labels; edge_labels=edge_labels)
    else
        return TikzGraphs.plot(g, text_labels)
    end
end
#Stay in text mode, escape some latex characters
function verbatim(s::String)
    s = replace(s, "_"=>"\\_")
end

"""
    classify(model::GBDT, X::AbstractVector{T}, members::AbstractVector{Int}, 
                     eval_module::Module=Main) where T

Predict classification label of each member using GBDT model.  Evaluate expressions in eval_module.
"""
function classify(model::GBDT, X::AbstractVector{T}, 
    members::AbstractVector{Int}=collect(1:length(X)), 
    eval_module::Module=Main) where T

    classify(model.tree, X, members, eval_module; catdisc=model.catdisc)
end
function classifyfuzzy(model::GBDT, X::AbstractVector{T},
    members::AbstractVector{Int}=collect(1:length(X)),
    eval_module::Module=Main) where T

    classifyfuzzy(model.tree, X, members, eval_module; catdisc=model.catdisc)
end

"""
    classify(node::GBDTNode, X::AbstractVector{T}, members::AbstractVector{Int}, eval_module::Module=Main;
                     catdisc::Union{Nothing,CategoricalDiscretizer}=nothing) where T

Predict classification label of each member using GBDT tree.  Evaluate expressions in eval_module.  If catdisc is available, use discretizer to decode labels.
"""
function classify(node::GBDTNode, X::AbstractVector{T},
    members::AbstractVector{Int}=collect(1:length(X)),
    eval_module::Module=Main;
    catdisc::Union{Nothing,CategoricalDiscretizer}=nothing) where T

    y_pred = Vector{Int}(undef,length(members))
    for i in eachindex(members)
        @eval eval_module x=$(X[i])
        y_pred[i] = _classify(node, eval_module) 
    end
    if catdisc == nothing
        return y_pred
    else
        return decode(catdisc, y_pred)
    end
end
function _classify(node::GBDTNode, eval_module::Module)
    isleaf(node) && return node.label

    ex = get_expr(node.gbes_result)
    ch =  Core.eval(eval_module, ex) ? node.children[1] : node.children[2] 
    return _classify(ch, eval_module) 
end

function classifyfuzzy(node::GBDTNode, X::AbstractVector{T},
    members::AbstractVector{Int}=collect(1:length(X)),
    eval_module::Module=Main;
    catdisc::Union{Nothing,CategoricalDiscretizer}=nothing) where T

    y_pred = Vector{Dict{Int64, Float64}}(undef,length(members))
    for i in eachindex(members)
        @eval eval_module x=$(X[i])
        labels = Dict{Int64, Float64}()
        _classifyfuzzy(node, eval_module, 1.0, labels)
        y_pred[i] = labels
    end
    if catdisc == nothing
        return y_pred
    else
        return decode(catdisc, y_pred)
    end
end
function _classifyfuzzy(node::GBDTNode, eval_module::Module, score::Float64, labels::Dict{Int64, Float64})
    if score == 0.0
        return
    end
    if isleaf(node)
        c = get(labels, node.label, -1)
        if c == -1
            labels[node.label] = score
        else
            labels[node.label] = c + score
        end
        return
    end

    ex = get_expr(node.gbes_result)
    val =  Core.eval(eval_module, ex)
    _classifyfuzzy(node.children[1], eval_module, score * val, labels)
    _classifyfuzzy(node.children[2], eval_module, score * (1.0 - val), labels)
end


"""
    node_members(model::GBDT, X::AbstractVector{T}, members::AbstractVector{Int}, 
                      eval_module::Module=Main) where T

Returns the members of each node in the tree.
"""
function node_members(model::GBDT, X::AbstractVector{T}, members::AbstractVector{Int}, 
                      eval_module::Module=Main) where T
    node_members(model.tree, X, members, eval_module)
end
"""
    node_members(node::gbdtnode, x::abstractvector{t}, members::abstractvector{int}, 
                      eval_module::module=main) where T

returns the members of each node in the tree.
"""
function node_members(node::GBDTNode, X::AbstractVector{T}, members::AbstractVector{Int}, 
                      eval_module::Module=Main) where T
    mvec = Vector{Vector{Int}}(undef,length(node))
    _node_members!(mvec, node, X, members, eval_module)
    mvec
end
function _node_members!(mvec::Vector{Vector{Int}}, node::GBDTNode, X::AbstractVector{T}, 
                        members::AbstractVector{Int}, eval_module::Module) where T
    mvec[node.id] = deepcopy(members)
    isleaf(node) && return

    ex = get_expr(node.gbes_result)
    y_bool = partition(X, members, ex, eval_module)
    members_true, members_false = members_by_bool(members, y_bool)
    _node_members!(mvec, node.children[1], X, members_true, eval_module)
    _node_members!(mvec, node.children[2], X, members_false, eval_module)
end

"""
    Base.getindex(model::GBDT, id::Int)

returns node with id 
"""
function Base.getindex(model::GBDT, id::Int)
    for node in PreOrderDFS(model.tree)
        if node.id == id
            return node 
        end
    end
    error("node id not found")
end

"""
    children_id(node::GBDTNode) 

returns a vector that contains the node ids of the children of node
"""
children_id(node::GBDTNode) = Int[c.id for c in children(node)]

"""
   get_expr(node::GBDTNode)

returns the expression of the node 
"""
ExprOptimization.get_expr(node::GBDTNode) = get_expr(node.gbes_result)

end # module
