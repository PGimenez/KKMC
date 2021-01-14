using MLJ,MLJModelInterface
using Random, LinearAlgebra, StatsBase, Parameters
import MLJBase

const MMI = MLJModelInterface

@mlj_model mutable struct KRRModel <: MLJModelInterface.Deterministic
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
Leverage(; s=1, alpha=1.0, probs=[1.0], folds=1) = Leverage(s,alpha,probs,folds)

function MMI.fit(m::KRRModel,verbosity::Int,X,y,w=1)
    idx = y[1]
    K = X[:,idx]
    y = y[2]
    fitresult = (idx,inv(K+m.mu*I)*y)
    cache = nothing
    report = nothing
    return (fitresult,cache,report)
end

function MMI.predict(m::KRRModel, fitresult, xnew) 
    samples, coef = fitresult
    K = xnew[:,2:end]
    K = xnew[:,samples]
    return K*coef
end

# function MMI.clean!(m::KRRModel) 
    # return 1
# end



function MLJBase.train_test_pairs(LS::Leverage, rows)
    setlist = Tuple{Array{Int64,1},Array{Int64,1}}[]
    for i=1:LS.folds
        train = StatsBase.sample(rows, Weights(LS.probs[rows]),LS.s, replace=false)
        test = setdiff(rows,train)
        push!(setlist, (train,test))
    end
    LS.s = LS.s + 1
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
    return (idx,W*y)
end

function MLJ.transform(LR::LeverageReweight, fitresult, K::Array{Float64,2})
    idx = Int.(K[:,1])
    K = K[:,2:end]
    Wl = diagm(fitresult[idx])
    Wr = diagm(fitresult[:])
    return Wl*K*Wr
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

