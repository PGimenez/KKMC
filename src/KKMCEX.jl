module KKMCEX
using SparseArrays, StatsBase, LinearAlgebra, Parameters
export KRR, VectorData, fit!, SamplingMatrix, VectorSampler, KronVectorSampler


abstract type Sampler end
export Sampler

mutable struct VectorSampler <: Sampler
    S::Any
    prob::Array{Float64}
    weighted::Bool
    function VectorSampler(s::Int64, prob::Array{Float64,1}, weighted::Bool)
        S = SamplingMatrix(s,prob,weighted)
        return new(S,prob,weighted)
    end
end
export VectorSampler

struct KronVectorSampler <: Sampler
    S::Any
    prob::Array{Array{Float64,1},1}
    weighted::Bool
    function KronVectorSampler(s_blocks::Array{Int64,1}, prob::Array{Array{Float64,1},1}, weighted::Bool)
        S = []
        for (i,s) in enumerate(s_blocks)
            push!(S,SamplingMatrix(s,prob[i],weighted))
        end
        return new(S,prob,weighted)
    end
end
export KronVectorSampler

abstract type KernelRegressor end
export KernelRegressor

@with_kw mutable struct KRR <: KernelRegressor
    mu::Float64 = 1e-6
    s::Int64 = 1
    d::Array{Float64} = [1.0]
    S::Any = 1
    train_error::Float64 = 1e3
end
export KRR
KRR(mu::Float64,s::Int64) = KRR(mu,s,zeros(s),spzeros(1),0.0)
KRR(mu::Float64,s::Int64,sampler::VectorSampler) = KRR(mu,s,zeros(s),sampler.S,0.0)

@with_kw  mutable struct TSKRR <: KernelRegressor
    mu::Float64 = 1e-6
    s::Int64 = 1
    D::Array{Float64,2} = randn(2,2)
    S::Any = 1
    train_error::Float64 = 1e3
end
export TSKRR
TSKRR(mu::Float64,s::Int64) = TSKRR(mu,s,zeros(s),KronVectorSampler([1],[1.0],true),spzeros(1),0.0)
TSKRR(mu::Float64,s::Int64,sampler::KronVectorSampler) = KRR(mu,s,zeros(s,s),sampler.S,0.0)

@with_kw  mutable struct RTSKRR <: KernelRegressor
    mu::Float64 = 1e-6
    s::Int64 = 1
    D::Array{Float64,2} = randn(2,2)
    S::Any = 1
    train_error::Float64 = 1e3
end
export RTSKRR
RTSKRR(mu::Float64,s::Int64) = RTSKRR(mu,s,zeros(s,s),spzeros(1),0.0)
RTSKRR(mu::Float64,s::Int64,sampler::VectorSampler) = RTSKRR(mu,s,zeros(s,s),sampler.S,0.0)

mutable struct VectorData
    N::Int64
    L::Int64
    f::Array{Float64,1}
    Kw::Array{Float64,2}
    Kh::Array{Float64,2}
end


function SamplingMatrix(s::Int64, prob::Array{Float64,1}, weighted::Bool)
    prob = abs.(prob)
    idx = StatsBase.sample(1:length(prob), Weights(prob),s, replace=false)
    idx = sort(idx[1:s])
    S = SparseArrays.spzeros(length(idx),length(prob))
    if weighted == true
        sample_weights = 1 ./ sqrt.(prob * length(idx))
    else
        sample_weights = ones(length(prob))
    end
    for i in 1:length(idx)
        S[i,idx[i]] = sample_weights[idx[i]]
    end
    return S
end
export SamplingMatrix



function fit!(model::KRR,data::VectorData,m::Array{Float64,1})
    K = build_ksamp(model.S,data)
    m = model.S*m
    S_w = Matrix(Diagonal(sum(model.S,dims=2)[:]))
    Ks = Hermitian(S_w*K*S_w)
    model.d = cholesky!(model.mu*I+Ks)\m
    model.train_error = sum((m - Ks*model.d).^2)
end
export fit!

function predict!(model::KRR, data::VectorData)
    F_est = data.Kw*reshape(model.S'*model.d,(data.N,data.L))*data.Kh
    return sum((data.f - vec(F_est)).^2)
end
export predict!


function build_kron_matrix(matrices::Array{Any})
    X_kron = 1
    for X in matrices
        X_kron = kron(X, X_kron)
    end
    return X_kron
end
export build_kron_matrix

function fit!(model::TSKRR, data::VectorData,m::Array{Float64,1})
    KS = model
    M = reshape(m,(data.N,data.L))
    M = Matrix(KS.S[1]*M*KS.S[2]')
    Kw = Matrix(KS.S[1]*data.Kw*KS.S[1]')
    Kh = Matrix(KS.S[2]*data.Kh*KS.S[2]')
    Dw = cholesky!(model.mu*I+Kw)\M
    Dh = cholesky!(model.mu*I+Kh)\Dw'
    model.S = [KS.S[1],KS.S[2]]
    model.D = Dh'
    model.train_error = sum((M - Kw*Dh'*Kh).^2)
end
export fit!

function recursive_KRR(mu::Float64,N::Int64,L::Int64,K::Array{Float64,2},Om::Array{Float64,2},M::Array{Float64,2})
    D = zeros(N,L)
    K_inv = inv(K+mu*I)
    for j = 1:L
        # reg =  Om[:,j] +  1e35*(1 .-Om[:,j])
        # reg = diagm(0 => reg)
        # # A = Om[:,j]' .* K_inv .*Om[:,j]
        # A = inv(K+reg)
        # D[:,j] = A*M[:,j]
        samples = Om[:,j]
        S = SamplingMatrix(Int(sum(samples)), samples, false)
        K_s = Matrix(S*K*S')
        idx = findall(samples .== 1)
        D[:,j] = S'*inv(K_s+ mu*I)*M[idx,j]
    end
    return D
end
function recursive_KRR(mu::Float64,N::Int64,L::Int64,K::Array{Float64,2},M::Array{Float64,2})
    K_inv = inv(K+mu*I)
    return K_inv*M
end
function fit!(model::RTSKRR,data::VectorData,m::Array{Float64,1})
    step = data.N
    D = zeros(data.N,data.L)
    M = reshape(m,(data.N,data.L))
    Om = reshape(sum(model.S,dims=1),(data.N,data.L))
    Dw = recursive_KRR(model.mu,data.N,data.L,data.Kw,Om,M)
    # Dh = recursive_KRR(model.mu,data.L,data.N,data.Kh,Matrix(Om'),Matrix(Dw'))
    # Dh = recursive_KRR(model.mu,data.L,data.N,data.Kh,ones(data.L,data.N),Matrix((data.Kw*Dw)'))
    Dh = recursive_KRR(model.mu,data.L,data.N,data.Kh,Matrix((data.Kw*Dw)'))
    # Dh = recursive_KRR(model.mu,data.L,data.N,data.Kh,ones(data.L,data.N),Matrix((Kw*Dw)'))
    # Dw = data.Kw*(cholesky!(model.mu*I+data.Kw)\M)
    # Dh = data.Kh*(cholesky!(model.mu*I+data.Kh)\Dw')
    model.D = Dh'
    idx_train = findall(Om .!= 0)
    model.train_error = sum(abs.(M - model.D*data.Kh)[idx_train])
    # model.train_error = mean(abs.(M - model.D)[idx_train])
end
export fit!
function predict!(model::RTSKRR, data::VectorData)
    # F_est = data.Kw*model.D*data.Kh
    F_est = model.D*data.Kh
    return sum(abs.(data.f - vec(F_est)))
end
export predict!


function predict!(model::TSKRR, data::VectorData)
    S = kron(model.S[2],model.S[1])
    F_est = data.Kw*model.S[1]'*model.D*model.S[2]'*data.Kh
    return sum((data.f - vec(F_est)).^2)
end
export predict!

function build_ksamp(S,data::VectorData)
    N = data.N
    L = data.L
    Kw = data.Kw
    Kh = data.Kh
    ns = size(S,1)
    Ks = zeros(ns,ns)
    nums = collect(1:ns)
    # sample_positions = findall(sum(S,dims=1)[1,:].!=0)
    sample_positions = zeros(ns)
    for i = nums
        j = 1
        while S[i,j] == 0;  j=j+1; end
        sample_positions[i] = j
    end
    for s = nums
        idx_sample = sample_positions[s]
        idx_Kw_sample = convert(Int,(idx_sample-1)%N)+1
        idx_Kh_sample = convert(Int,ceil(idx_sample/N))
        for i = nums
            idx = sample_positions[i]
            idx_Kw = convert(Int,(idx-1)%N)+1
            idx_Kh = convert(Int,ceil(idx/N))
            Kw_val = Kw[idx_Kw_sample,idx_Kw]
            Kh_val = Kh[idx_Kh_sample,idx_Kh]
            Ks[s,i] =  Kw_val*Kh_val
        end
    end
    return Ks
end
end
