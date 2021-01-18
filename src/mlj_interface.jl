using MLJ,MLJModelInterface
using Random, LinearAlgebra, StatsBase, Parameters
import MLJBase

const MMI = MLJModelInterface

@mlj_model mutable struct KRRModel <: MMI.Deterministic
    mu::Float64 = 1.0
    kernel::String = "linear"
    # kernel_matrix::Array{Float64,2} = zeros(2,2)
end


mutable struct Leverage <: MLJ.ResamplingStrategy
    s::Int64
    alpha::Float64
    probs::Array{Float64}
    folds::Int32
end
# Keyword Constructor
# Leverage(; s=1, alpha=1.0, probs=[1.0], folds=1, lr=LeverageReweight()) = Leverage(s,alpha,probs,folds,lr)

@with_kw mutable struct LeverageReweight <: Static
    alpha::Float64
    weights::Array{Float64}
end

function Leverage(type,alpha,s,K,folds)
    lscores = get_lscores(type,K,1,alpha)
    probs = lscores ./ sum(lscores) 
    fitresult = 1/ sqrt.(probs*s)
    return Leverage(s,alpha,probs,folds)
end

function LeverageReweight(type,alpha,s,K)
    lscores = get_lscores(type,K,1,alpha)
    probs = lscores ./ sum(lscores) 
    fitresult = 1/ sqrt.(probs*s)
    return LeverageReweight(alpha,fitresult)
end

@with_kw mutable struct LKRRModel <: DeterministicComposite
    KRR::KRRModel = KRRModel()
    LR::LeverageReweight = LeverageReweight()
end
# keyword constructor
LKRRModel(; KRR=KRRModel(), LR=LeverageReweight()) = LKRRModel(KRR,LR)

function MMI.fit(model::LKRRModel, verbosity, X, y)
    ys = source(y)
    Xs = source(X)
    lr = machine(model.LR)
    # MLJ.fit!(lr)
    yt = transform(lr,ys)
    Kt = transform(lr,Xs)
    krr = machine(model.KRR, Kt, yt)
    zhat = MMI.predict(krr,Kt)
    yhat = inverse_transform(lr,zhat)
    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    return!(mach, model, verbosity)
end


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
    idx = Int.(xnew[:,1])
    K = xnew[:,2:end]
    K = K[:,samples]
    return table((idx=idx,val=K*coef))
end

# function MMI.clean!(m::KRRModel) 
    # return 1
# end
function MMI.clean!(m::LeverageReweight) 
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

function MLJ.transform(LR::LeverageReweight, fitresult, x::NamedTuple)
    fitresult = LR.weights
    idx = x[1]
    y = x[2]
    W = diagm(fitresult[idx])
    return (idx,W*y)
end

function MLJ.inverse_transform(LR::LeverageReweight, fitresult, x::NamedTuple)
    fitresult = LR.weights[:]
    idx = x[1]
    # y = x[2]
    W = diagm(1 ./ LR.weights[idx])
    y = W*x[2]
    return table((idx=idx,val=y))
end

function MLJ.transform(LR::LeverageReweight, fitresult, K::Array{Float64,2})
    fitresult = LR.weights
    idx = Int.(K[:,1])
    K = K[:,2:end]
    Wl = diagm(fitresult[idx])
    Wr = diagm(fitresult[:])
    return hcat(idx,Wl*K*Wr)
end

tuple_rms(y1::NamedTuple, y2::NamedTuple) = MLJ.rms(y1[2],y2[2])

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

