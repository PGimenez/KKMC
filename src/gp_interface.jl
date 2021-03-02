
using MLJ,MLJModelInterface, MLJTuning
using Random, LinearAlgebra, StatsBase, Parameters
using AugmentedGaussianProcesses: GP, train!, predict_y
using KernelFunctions: Kernel, EyeKernel, SqExponentialKernel
import MLJBase

const MMI = MLJModelInterface

@with_kw mutable struct GaussianProcess <: MLJ.Deterministic
    kernel::Kernel = SqExponentialKernel()
    noise::Float64 = 1.0
    opt_noise::Bool = true
end

function MMI.fit(model::GaussianProcess, verbosity, X, y)
    gp_model = GP(X,y,model.kernel,noise=model.noise,opt_noise=model.opt_noise,optimiser=false)
    train!(gp_model)
    return (gp_model,nothing,nothing)
end

function MMI.predict(model::GaussianProcess, fitresult, Xnew)
    return  predict_y(fitresult,Xnew)
end
