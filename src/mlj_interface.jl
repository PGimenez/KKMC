using MLJ,MLJModelInterface
using Random, LinearAlgebra, StatsBase, Parameters
import MLJBase

const MMI = MLJModelInterface

@mlj_model mutable struct KRRModel <: MMI.Deterministic
    mu::Float64 = 1.0
    kernel::String = "linear"
end


mutable struct Leverage <: MLJ.ResamplingStrategy
    s::Int64
    alpha::Float64
    probs::Array{Float64}
    folds::Int32
end

@with_kw mutable struct LeverageSampler <: MMI.Deterministic
    type::String
    alpha::Float64
    s::Int
end
LeverageSampler(; type="Leverage", alpha=1.0) = LeverageSampler(type,alpha)


@with_kw mutable struct LeverageWeighter <: MMI.Deterministic
    type::String
    alpha::Float64
    s::Int
end

function Leverage(type,alpha,s,K,folds)
    lscores = get_lscores(type,K,1,alpha)
    probs = lscores ./ sum(lscores) 
    fitresult = 1/ sqrt.(probs*s)
    return Leverage(s,alpha,probs,folds)
end

@with_kw mutable struct LKRRModel <: DeterministicComposite
    KRR::KRRModel = KRRModel()
    LS::LeverageSampler = LeverageSampler()
end
LKRRModel(; KRR=KRRModel(), LS=LeverageSampler()) = LKRRModel(KRR,LS)


function MMI.fit(model::LKRRModel, verbosity, X, y)
    ys = source(y)
    Xs = source(X)
    ls = machine(model.LS,Xs,ys)
    MLJ.fit!(ls)
    lw_model = LeverageWeighter(model.LS.type,model.LS.alpha,model.LS.s)
    lw = machine(lw_model,Xs,ys)
    MLJ.fit!(lw)
    yt = transform(ls,ys)
    Kt = transform(ls,Xs)
    K_predict = transform(lw,Xs)
    krr = machine(model.KRR, Kt, yt)
    zhat = MMI.predict(krr,K_predict)
    yhat = inverse_transform(ls,zhat)
    MLJ.fit!(yhat)
    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    return!(mach, model, verbosity)
end

weighted_kernel(X,W) = hcat(X[:,1],W*X[:,2:end]*W)
weighted_kernel(X) = X

function MMI.fit(m::KRRModel,verbosity::Int,X,y)
    idx = y[1]
    K = X[:,2:end]
    K = K[:,idx]
    y = y[2]
    fitresult = (idx,inv(K+m.mu*I)*y)
    cache = nothing
    report = nothing
    return (fitresult,cache,report)
end


function MMI.predict(m::KRRModel, fitresult, xnew) 
    samples, coef = fitresult
    idx = setdiff(Int.(xnew[:,1]),samples)
    K = xnew[:,2:end]
    K = K[idx,samples]
    return table((idx=idx,val=K*coef))
end

function MMI.fit(LS::LeverageSampler, verbosity::Int, X, y)
    K = X[:,2:end]
    lscores = get_lscores(LS.type,K,1,LS.alpha)
    probs = lscores ./ sum(lscores) 
    weights = 1/ sqrt.(probs*LS.s)
    train = StatsBase.sample(y[1], Weights(probs),LS.s, replace=false)
    return ((weights[:],train),nothing,nothing)
end

function MMI.fit(LW::LeverageWeighter, verbosity::Int, X, y)
    LS = LeverageSampler(LW.type,LW.alpha,LW.s)
    fitresult,cache,report = MMI.fit(LS,verbosity,X,y)
    return (fitresult[1],cache,report)
end
# function MMI.clean!(m::KRRModel) 
    # return 1
# end
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

function MLJ.transform(LS::LeverageSampler, fitresult, x::NamedTuple)
    idx = fitresult[2]
    w = fitresult[1][idx]
    y = x[2]
    return (idx,w.*y[idx])
end

function MLJ.inverse_transform(LS::LeverageSampler, fitresult, x::NamedTuple)
    idx = x[1]
    yt = x[2]
    w = 1 ./ fitresult[1][idx]
    return table((idx=idx,val=w.*yt))
end

function MLJ.transform(LS::LeverageSampler, fitresult, K::Array{Float64,2})
    idx = fitresult[2]
    w = fitresult[1]
    K = K[idx,2:end]
    Wl = diagm(w[idx])
    Wr = diagm(w)
    return hcat(idx,Wl*K*Wr)
end

function MLJ.transform(LW::LeverageWeighter, fitresult, x::NamedTuple)
    return fitresult .* x[2]
end
function MLJ.transform(LW::LeverageWeighter, fitresult, K::Array{Float64,2})
    idx = K[:,1]
    K = K[:,2:end]
    W = diagm(fitresult)
    return hcat(idx, W*K*W)
end

tuple_rms(yhat::NamedTuple, ground::NamedTuple) = MLJ.rms(yhat[2],ground[2][yhat[1]])

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

