
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

@with_kw mutable struct LGP <: DeterministicComposite
    GP::GaussianProcess = GausianProcess()
    LS::LeverageSampler = LeverageSampler()
end

function MMI.fit(model::GaussianProcess, verbosity, X, y)
    gp_model = GP(X,y,model.kernel,noise=model.noise,opt_noise=model.opt_noise,optimiser=false)
    train!(gp_model)
    return (gp_model,nothing,nothing)
end

function MMI.predict(model::GaussianProcess, fitresult, Xnew)
    return  predict_y(fitresult,Xnew)
end


function MMI.fit(model::LGP, verbosity, X, y)
    verbosity > 0 ? println("fit composite") : nothing
    model.LS.rng = model.LS.rng + 1
    ys = source(y)
    Xs = source(X)
    ls = machine(model.LS,Xs,ys)
    MLJ.fit!(ls, verbosity=verbosity)
    # select samples and rows from y,X
    yt = transform(ls,ys)
    Kt = transform(ls,Xs)
    gp = machine(model.GP, Kt, yt)
    zhat = MMI.predict(gp,Xs)
    yhat = inverse_transform(ls,zhat)
    mach = machine(Deterministic(), source(Xs()), source(ys()); predict=yhat)
    return!(mach, model, verbosity)
end

