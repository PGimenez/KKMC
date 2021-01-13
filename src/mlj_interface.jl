using MLJ,MLJModelInterface
using Random, LinearAlgebra, StatsBase
import MLJBase

const MMI = MLJModelInterface

MMI.@mlj_model mutable struct KRRModel <: MLJModelInterface.Deterministic
    mu::Float64 = 1::(_ > 0)
    kernel::String = "linear"
    kernel_matrix::Array{Float64,2} = zeros(1,1)
end

function MMI.fit(m::KRRModel,verbosity::Int,x,y,w=1)
    # w = 1 ./ sqrt.(w*length(y))
    x = x[1]
    K = m.kernel_matrix[x,x]
    K = reshape(K,(length(x),length(x)))
    # W = diagm(w)
    # K = W*K*W
    # y = W*y
    # fitresult = (x,W*inv(K+m.mu*I)*y)
    fitresult = (x,inv(K+m.mu*I)*y)
    cache = nothing
    report = nothing
    return (fitresult,cache,report)
end

function MMI.predict(m::KRRModel, fitresult, xnew) 
    x = xnew[1]
    samples, coef = fitresult
    K = m.kernel_matrix[x,samples]
    return K*coef
end

# function MMI.clean!(m::KRRModel) 
    # return 1
# end

struct Leverage <: MLJ.ResamplingStrategy
    s::Int64
    alpha::Float64
    probs::Array{Float64}
    folds::Int32
end

# Keyword Constructor
Leverage(; s=1, alpha=1, probs=1, folds=1) = Leverage(s,alpha,probs,folds)


function MLJBase.train_test_pairs(LS::Leverage, rows)
    # train, test = partition(rows, LS.fraction_train,
                          # shuffle=LS.shuffle, rng=holdout.rng)
    setlist = Tuple{Array{Int64,1},Array{Int64,1}}[]
    for i=1:LS.folds
        train = StatsBase.sample(rows, Weights(LS.probs[rows]),LS.s, replace=false)
        test = setdiff(rows,train)
        push!(setlist, (train,test))
    end
    return setlist 
end

mutable struct LeverageReweight <: MMI.Unsupervised
    type::String
    alpha::Float64
    s::Int32
    K::Array{Float64,2}
end

function MMI.fit(LR::LeverageReweight,verbosity::Int,K)
    lscores = get_lscores(LR.type,LR.K,1,LR.alpha)
    probs = lscores ./ sum(lscores) 
    fitresult = 1/ sqrt.(probs*LR.s)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MLJ.transform(LR::LeverageReweight, fitresult, x::NamedTuple)
    idx = x[1]
    y = x[2]
    W = diagm(fitresult[idx])
    return W*y
end

function MLJ.transform(LR::LeverageReweight, fitresult, K::Array{Float64,2})
    W = diagm(fitresult[:])
    return W*K*W
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

