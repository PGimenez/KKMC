module KernelMF
using LinearAlgebra, StatsBase, Parameters
export MF, KMF, RGKMF, fit!, predict!
function frob(X)
    return sum(X.^2)
end

abstract type MF end

mutable struct MatrixData
    N::Int64
    L::Int64
    F::Array{Float64,2}
    Kw::Array{Float64,2}
    Kh::Array{Float64,2}
end
export MatrixData

@with_kw  mutable struct KMF <: MF
    mu::Float64 = 1e-6
    N::Int64 = 1
    L::Int64 = 1
    p::Int64 = 1
    W::Array{Float64,2} = randn(2,2)
    H::Array{Float64,2} = randn(2,2)
    train_error::Float64 = 1e3
    tol::Float64 = 1e-6
    maxiter::Int64 = 5e3
end
export KMF
KMF(mu::Float64,N::Int64,L::Int64,p::Int64) = KMF(mu,N,L,p,randn(N,p),randn(L,p),1.0,1e-6,5e3)

@with_kw  mutable struct RGKMF <: MF
    sigma::Float64 = 1e-6
    N::Int64 = 1
    L::Int64 = 1
    p::Int64 = 1
    W::Array{Float64,2} = randn(2,2)
    H::Array{Float64,2} = randn(2,2)
    train_error::Float64 = 1e3
    tol::Float64 = 1e-6
    maxiter::Int64 = 5e3
end
export RGKMF
RGKMF(sigma::Float64,N::Int64,L::Int64,p::Int64) = RGKMF(sigma,N,L,p,randn(N,p),randn(L,p),1.0,1e-6,5e3)

# function fit!(model::KMF,M::Array{Float64,2},Kw::Array{Float64,2},Kh::Array{Float64,2})
function fit!(model::KMF,data::Any,M::Array{Float64,2})
    Kw = data.Kw
    Kh = data.Kh
    Kw_i = pinv(Kw+0.0001*I) # need to make Kw invertible
    Kh_i = pinv(Kh+0.0001*I)
    idx_train = findall(M .!= 0)
    Om = zeros(size(M,1),size(M,2))
    Om[idx_train] = ones(length(idx_train))
    W = model.W
    H = model.H
    count = 1
    err = 100
    ratio = 1
    mu = model.mu
    while (ratio > model.tol) && (count < 4000)
    # while (ratio > model.tol)
        err_old = err
        for i = collect(1:model.N)
            idx = findall(Om[i,:] .== 1)
            W[i,:] = real((-mu*Kw_i[i,:]'*W + mu*Kw_i[i,i]*W[i,:]' + M[i,:]'*H)*pinv(H[idx,:]'*H[idx,:] + mu*Kw_i[i,i]*I))
        end
        for j = collect(1:model.L)
            idx = findall(Om[:,j] .== 1)
            H[j,:] = real((-mu*Kh_i[j,:]'*H + mu*Kh_i[j,j]*H[j,:]' + M[:,j]'*W)*pinv(W[idx,:]'*W[idx,:] + mu*Kh_i[j,j]*I))
        end
        err = mean(abs.(W*H'-M))
        ratio = abs(err_old - err)/err_old
        count= count+1
    end
    X = W*H'
    train_samples = findall(M .!= 0)
    train_error = mean(abs.(X[idx_train]-M[idx_train]))
    model.W = W
    model.H = H
    model.train_error = train_error
end
export fit!

function fit!(model::RGKMF,data::MatrixData,M::Array{Float64,2})
    idx_train = findall(M .!= 0)
    Om = zeros(size(M,1),size(M,2))
    Om[idx_train] = ones(length(idx_train))
    W = model.W
    H = model.H
    Kw = data.Kw
    Kh = data.Kh
    err = 100
    ratio = 1
    M_residual = M
    count = 1
    for i = 1:model.p
        w = randn(model.N)
        h = randn(model.L)
        ratio = 1
        count = 1
        while (ratio > model.tol) && (count < 400)
            # while (ratio > model.tol)
            err_old = err
            m_sum_w = M_residual*h
            w, Cw = find_moments(h, Om, m_sum_w, Kw, model.sigma)
            m_sum_h = M_residual'*w
            h, Ch = find_moments(w, Om', m_sum_h, Kh, model.sigma)
            err = mean(abs.(w*h'-M_residual))
            ratio = abs(err_old - err)/err_old
            count = count+1
        end
        M_residual = (M_residual - w*h').*Om #comprovar TODO
        W[:,i] = w
        H[:,i] = h
    end
    X = W*H'
    train_error = mean(abs.(X[idx_train]-M[idx_train]))
    model.W = W
    model.H = H
    model.train_error = train_error
end
export fit!

function find_moments(w, Om, m_sum, Kh, sigma)
    # D = (inv(diagm(0 => vec(Array(Om*abs.(w.^2))))))
    # gm = D*Kh*inv(Kh + sigma*D)*m_sum
    D = (diagm(0 => vec(Array(Om*abs.(w.^2)))))
    gm = Kh*inv(D*Kh + sigma*I)*m_sum
    C = D*( Kh - Kh*inv(Kh + sigma*I)*Kh)*D
    return gm, C
end

function predict!(model::MF,F::Array{Float64,2})
    return mean(abs.(model.W*model.H' .- F))
end
end
