using MLJ,MLJModelInterface, MLJTuning
using Random, LinearAlgebra, StatsBase, Parameters
using KernelFunctions: Kernel, kernelmatrix, ExponentialKernel
import MLJBase

const MMI = MLJModelInterface

@with_kw mutable struct KRRModel <: MMI.Deterministic
    mu::Float64 = 1.0
    kernel::Kernel = ExponentialKernel()
end


mutable struct Leverage <: MLJ.ResamplingStrategy
    s::Int64
    alpha::Float64
    probs::Array{Float64}
    folds::Int32
end

mutable struct AllData <: MLJ.ResamplingStrategy 
    N::Int
end


@with_kw mutable struct LeverageSampler{T} <: MMI.Deterministic
    type::T = UniformSampling()
    alpha::Float64 = 1.0
    s::Int = 1
    rng = 1
    kernel::Kernel = ExponentialKernel()
end
# LeverageSampler(; type="Leverage", alpha=1.0,s=1,rng=nothing,kernel=ExponentialKernel()) = LeverageSampler(type,alpha,s,rng)



function Leverage(type,alpha,s,K,folds)
    lscores = get_lscores(type,K,1,alpha)
    probs = lscores ./ sum(lscores) 
    fitresult = 1 ./ sqrt.(probs*s)
    return Leverage(s,alpha,probs,folds)
end

@with_kw mutable struct LKRRModel <: DeterministicComposite
    KRR::KRRModel = KRRModel()
    LS::LeverageSampler = LeverageSampler()
end
LKRRModel(; KRR=KRRModel(), LS=LeverageSampler()) = LKRRModel(KRR,LS)

@with_kw mutable struct TunedLKRRModel <: MMI.Deterministic
    tuner::MLJTuning.DeterministicTunedModel
    tuned::Bool
    tuned_model::LKRRModel
end


function MMI.fit(model::TunedLKRRModel, verbosity, X, y)
    if model.tuned == false
        verbosity > 0 ? println("fit tuner") : nothing
        mach = machine(model.tuner,X,y)
        MLJBase.fit!(mach,force=true)
        model.tuned_model = mach.fitresult.model 
        lkrr = machine(mach.fitresult.model,X,y)
        MLJBase.fit!(lkrr)
        model.tuned = true
        return (lkrr.fitresult,nothing,nothing)
    else
        verbosity > 0 ? println("fit LKRR in tuner") : nothing
        lkrr = machine(model.tuned_model,X,y)
        MLJBase.fit!(lkrr)
        return (lkrr.fitresult,nothing,nothing)
    end
end

function MMI.update(model::TunedLKRRModel, verbosity::Int, old_fitresult, old_cache, X, y)
    verbosity > 0 ? println("update TunedLKRR") : nothing
    model.tuned = false
    return MMI.fit(model,verbosity,X,y)
end
#
function MMI.predict(model::TunedLKRRModel, fitresult, Xnew)
    return MMI.predict(model.tuned_model,fitresult,Xnew)
end

function MMI.fit(model::LKRRModel, verbosity, X, y)
    verbosity > 0 ? println("fit composite") : nothing
    model.LS.rng = model.LS.rng + 1
    ys = source(y)
    Xs = source(X)
    ls = machine(model.LS,Xs,ys)
    MLJ.fit!(ls, verbosity=verbosity)
    # select samples and rows from y,X and produce weights
    yt = transform(ls,ys)
    Kt = transform(ls,Xs)
    wt = KKMC.transform(ls,source(length(y)))
    krr = machine(model.KRR, Kt, yt, wt)
    zhat = MMI.predict(krr,Xs)
    yhat = inverse_transform(ls,zhat)
    mach = machine(Deterministic(), source(Xs()), source(ys()); predict=yhat)
    return!(mach, model, verbosity)
end

function MMI.fit(m::KRRModel,verbosity::Int,X,y)
    verbosity > 0 ? println("fit KRR") : nothing
    K = kernelmatrix(m.kernel,X,obsdim=1)
    fitresult = (X,(K+m.mu*I) \ y)
    cache = nothing
    report = nothing
    return (fitresult,cache,report)
end

function MMI.fit(m::KRRModel,verbosity::Int,X,y,w)
    verbosity > 0 ? println("fit weighted KRR") : nothing
    K = kernelmatrix(m.kernel,X,obsdim=1)
    y = y .* w
    W = diagm(w)
    K = W*K*W
    coef = (K+m.mu*I) \ y
    fitresult = (X,W*coef)
    cache = nothing
    report = nothing
    return (fitresult,cache,report)
end

function MMI.predict(m::KRRModel, fitresult, Xnew) 
    Xtrain, coef = fitresult
    s = length(coef)
    X = vcat(Xtrain,Xnew)
    K = kernelmatrix(m.kernel,X,obsdim=1)[s+1:end,1:s]
    return K*coef
end

function MMI.fit(LS::LeverageSampler, verbosity::Int, X, y)
    verbosity > 0 ? println("fit LS") : nothing
    K = kernelmatrix(LS.kernel,X,obsdim=1)
    lscores = get_lscores(LS.type,K,1,LS.alpha)
    probs = lscores ./ sum(lscores) 
    rng = MersenneTwister(LS.rng)
    idxs = StatsBase.sample(rng,collect(1:length(y)), Weights(probs),LS.s, replace=false) |> sort
    w = 1 ./ sqrt.(probs*LS.s)
    return ((idx=idxs,weights=w),nothing,nothing)
end

function MMI.clean!(m::LeverageSampler) 
    return 1
end


function MLJBase.train_test_pairs(LS::Leverage, rows)
    rows = 1:length(LS.probs)
    setlist = Tuple{Array{Int64,1},Array{Int64,1}}[]
    for i=1:LS.folds
        train = StatsBase.sample(rows, Weights(LS.probs[rows]),LS.s, replace=false)
        test = setdiff(rows,train)
        push!(setlist, (train,test))
    end
    return setlist 
end

function MLJBase.train_test_pairs(AD::AllData, rows)
    return [(shuffle(collect(1:AD.N)),shuffle(collect(1:AD.N)))]
end

MLJ.transform(LS::LeverageSampler, fitresult, x) = x[fitresult.idx,:]
# MLJ.transform(LS::LeverageSampler, fitresult, X::Array{Float64,2}) = X[fitresult.idx,:]


MLJ.transform(LS::LeverageSampler, fitresult, N::Int) = fitresult.weights[fitresult.idx]

function MLJ.inverse_transform(LS::LeverageSampler, fitresult, x)
    test_idx = setdiff(1:length(x),fitresult.idx)
    return table((idx=test_idx,val=x[test_idx]))
end


# function MLJ.inverse_transform(LS::LeverageSampler{GreedyLeverageSampling}, fitresult, x)
    # test_idx = setdiff(1:length(x),fitresult.idx)
    # return table((idx=test_idx,val=x[test_idx]))
# end
function MLJ.transform(LS::LeverageSampler{GreedyLeverageSampling}, fitresult, x)
    idx = sortperm(fitresult.weights)[1:LS.s] |> sort
    return x[idx,:]
end

MLJ.transform(LS::LeverageSampler{GreedyLeverageSampling}, fitresult, N::Int) = ones(length(fitresult.idx))

function tuple_rms(yhat::NamedTuple, ground::Array{Float64,1}) 
    MLJ.rms(yhat[2],ground[yhat[1]])
end

MMI.metadata_pkg.(
                  (KRRModel),
                  name       = "KKMC",
                  uuid       = "",
                  url        = "",
                  julia      = true,
                  license    = "MIT",
                  is_wrapper = false)

MMI.metadata_model(KRRModel,
                   input   = MMI.Table(MMI.Continuous),  # what input data is supported?
                   target  = AbstractVector{MMI.Continuous},           # for a supervised model, what target?
                   output  = MMI.Table(MMI.Continuous),  # for an unsupervised, what output?
                   weights = true,                                                  # does the model support sample weights?
                   descr   = "A short description of your model",
                   path    = "KKMC.KRRModel"
                  )
# struct CVKernel <: ResamplingStrategy

# end
# function train_test_pairs(strategy::CVKernel, rows)
# train
# end

