using MLJModelInterface
using Random, LinearAlgebra
import MLJBase

const MMI = MLJModelInterface

MMI.@mlj_model mutable struct KRRModel <: MLJModelInterface.Deterministic
    mu::Float64 = 1::(_ > 0)
    kernel::String = "linear"
    kernel_matrix::Array{Float64,2} = zeros(1,1)
end

function MMI.fit(m::KRRModel,verbosity::Int,x,y)
    x = x[:]
    K = m.kernel_matrix[x,x]
    fitresult = (x,inv(K+m.mu*I)*y)
    cache = nothing
    report = nothing
    return (fitresult,cache,report)
end

function MMI.predict(::KRRModel, fitresult, xnew) 
    x = xnew[:]
    samples, coef = fitresult
    K = m.kernel_matrix[x,samples]
    return K*coef
end

function MMI.clean!(m::KRRModel)
    return 1
end

# struct CVKernel <: ResamplingStrategy

# end
# function train_test_pairs(strategy::CVKernel, rows)
    # train
# end

