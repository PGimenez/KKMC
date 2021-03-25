module LeverageScores

using StatsBase
using SparseArrays
using Distributed
using SharedArrays, ReusePatterns
import LinearAlgebra.eigen
import Base.sortperm

export get_lscores, PassiveSampling, UniformSampling, UnweightedUniformSampling, LeverageSampling, UnweightedLeverageSampling, GreedyLeverageSampling

abstract type PassiveSampling end

@quasiabstract struct UniformSampling <: PassiveSampling end
@quasiabstract struct UnweightedUniformSampling <: UniformSampling end

@quasiabstract struct LeverageSampling <: PassiveSampling end
@quasiabstract struct UnweightedLeverageSampling <: LeverageSampling end

struct GreedyLeverageSampling <: PassiveSampling end
    

function leverage_scores(K,mu)
    return diag(K*pinv(K+mu*I))
end
function leverage_kron(Kr,Kc,mu)
    N = size(Kr,1)
    L = size(Kc,1)
    er,Qr = (eigen(Kr)...,)
    ec,Qc = (eigen(Kc)...,)
    # sort eigenvalues
    idx_sort_r = sortperm(er)
    idx_sort_c = sortperm(ec)
    er = er[idx_sort_r]; Qr = Qr[:,idx_sort_r]
    ec = ec[idx_sort_c]; Qc = Qc[:,idx_sort_c]
    # scores = SharedArray{Float64}(N*L)
    scores = zeros(N*L)
    for i in 1:N
        for j in 1:L
            idx = i + (j-1)*N
            scores[idx] = eigen_kron(Qr[i,:],Qc[j,:],er,ec,mu)
        end
    end
    return scores
end
function eigen_kron(qr,qc,er,ec,mu)
    N = length(qr)
    L = length(qc)
    r = 0.0
    for j in 1:L
    @sync @distributed for i in 1:N
            e = ec[j]*er[i]
            r = r + (qc[j]*qr[i])^2*(e/(e+mu))
        end
    end
    return r
end


function leverage_kron_inv(Kr,Kc,alpha)
    N = size(Kr,1)
    L = size(Kc,1)
    l_matr = Kr*inv(Kr+alpha*I)
    l_matc = Kc*inv(Kc+alpha*I)
    l_r = diag(l_matr)
    l_c = diag(l_matc)
    return kron(l_c,l_r)
end

function leverage_kron_samp(Kr,Kc,nu)
    N = size(Kr,1)
    L = size(Kc,1)
    factor = 3
    samples = convert(Int,round(N/factor))
    prob = ones(N)/N*samples
    # s = sample_vector(prob,samples)
    idx = StatsBase.sample(1:length(prob), Weights(prob),samples, replace=false)
    idx = sort(idx[1:samples])
    Kr_s = Kr[:,idx]
    Kr_ss = Kr[idx,idx]
    l_matr = (Kr-Kr_s*inv(Kr_ss+nu*I)*Kr_s')
    samples = convert(Int,round(L/factor))
    prob =  ones(L)/L*samples
    # s = sample_vector(prob,samples)
    idx = StatsBase.sample(1:length(prob), Weights(prob),samples, replace=false)
    idx = sort(idx[1:samples])
    Kc_s = Kr[:,idx]
    Kc_ss = Kr[idx,idx]
    l_matc = (Kc-Kc_s*inv(Kc_ss+nu*I)*Kc_s')
    l_r = diag(l_matr)
    l_c = diag(l_matc)
    return kron(l_c,l_r)
end

function leverage_data(Y,alpha)
    U,d,V = svd(Y)
    U = U[:,1:5]
    V = V[:,1:5]
    D = diagm(d ./ (d+alpha))
    l_r = sum(abs.(U).^2,dims=2)
    l_c = sum(abs.(V).^2,dims=2)
    return kron(l_c,l_r)
end

function get_lscores(method,Kr,Kc,mu)
    # @show method isa UniformSampling, typeof(method)

    if method isa UniformSampling
        return ones(size(Kr,1)*size(Kc,1))
        # elseif  method == "Exact"
        # return leverage(K_kron,mu)
    elseif  method isa Union{LeverageSampling,GreedyLeverageSampling}
        return leverage_kron(Kr,Kc,mu)
    # elseif  method == "Inverse"
        # return leverage_kron_inv(Kr,Kc,mu)
    # elseif  method == "Sampled"
        # return leverage_kron_samp(Kr,Kc,mu)
    end
end

function get_lscores_grid(method,Kr,Kc,mu)
    if method == "Uniform"
        return 0.1*ones(size(Kr,1)+size(Kc,1))
    elseif  method == "Leverage"
        return vcat(leverage_scores(Kr,mu),leverage_scores(Kc,mu))
    end
end

# function get_probs(l,l_kron,l_kron_inv)
    # prob = l/sum(l)*samples_avg
    # prob_kron = l_kron/sum(l_kron)*samples_avg
    # prob_kron_inv = l_kron_inv/sum(l_kron_inv)*samples_avg
    # return prob,prob_kron,prob_kron_inv
# end

function get_lscores_alphas(algconf_list,Kr,Kc,alpha_vec)
    leverage_methods = [alg.sampling for alg in algconf_list]
    L_mat = SharedArray{Float64}(length(leverage_methods),length(alpha_vec),size(Kr,1)*size(Kc,1))
    N = size(Kr,1)
    L = size(Kc,1)
    @sync @distributed for (m,method) in collect(enumerate(algconf_list))
        @sync @distributed for (i,alpha) in collect(enumerate(alpha_vec))
            if method.grid == true
                # L_mat[m,i,1:N] = get_lscores(method.sampling,Kr,1,alpha)
                # L_mat[m,i,N+1:N+L] = get_lscores(method.sampling,Kc,1,alpha)
                L_mat[m,i,1:N+L] = get_lscores_grid(method.sampling,Kr,Kc,alpha)
            else
                L_mat[m,i,:] = get_lscores(method.sampling,Kr,Kc,alpha)
            end
        end
    end
    return L_mat
end

end
