using PyCall
using LightGraphs
using StatsBase, Statistics, SparseArrays, LinearAlgebra, JLD2, Random, MLJBase, Dates
using DelimitedFiles
using KernelFunctions: ExponentialKernel, LinearKernel
import RDatasets, KernelFunctions

DataKernels = Dict("housing"   => KernelFunctions.transform(ExponentialKernel(),0.1),
                   "mushrooms" => LinearKernel(c=1e-8),
                   "stocks"    => KernelFunctions.transform(ExponentialKernel(),0.1),
                   "labor"     => KernelFunctions.transform(ExponentialKernel(),0.1),
                   "wine"      => KernelFunctions.transform(ExponentialKernel(),2))

function make_PSD(K)
    if size(K,1) == 1; return K; end;
    e,Q = eigen(K)
    e = abs.(e)
    Q = abs.(Q)
    K = Q*diagm(0 => abs.(e))*Q'
    return Matrix(Hermitian(K))
end

function gen_synth_kernels(N,L)
    X = rand(N,N) .- 0.5
    X[1,:] = X[1,:] .+ 0.1
    X[10,:] = X[5,:] .+ 0.2
    X = X .- mean(X,dims=2)
    X = X .+ collect(1:N)/N
    Kw = X*X'
    X = rand(L,L) .- 0.5
    X[1,:] = X[1,:] .+ 0.1
    X[5,:] = X[5,:] .+ 0.2
    X[10,:] = X[5,:] .+ 0.2
    X = X .- mean(X,dims=2)
    X = X .+ exp.(-collect(L:-1:1))
    Kh = X*X'
    return Kw,Kh
end

function synth_rank_matrix(N,L,rank)
    Kw = randn(N,N)
    Kw = Kw*Kw'/maximum(Kw)^2
    Kh = randn(L,L)
    Kh = Kh*Kh'/maximum(Kh)^2
    W = sqrt(Kw)*randn(N,rank)
    H = sqrt(Kh)*randn(L,rank)
    return W*H', Kw, Kh
end


function synth_matrix(N,L)
    # rnk = convert(Int,round(N/20))
    # Kw,Kh = gen_synth_kernels(N,L)
    # F = Kw*randn(N,rnk)*randn(rnk,L)*Kh
    # return F, Kw, Kh
    return synth_rank_matrix(N,L,Int(round(N/20)))
end

function synth_matrix_cov(N,L)
    rnk = convert(Int,round(N/20))
    Kw,Kh = gen_synth_kernels(N,L)
    F = Kw*randn(N,rnk)*randn(rnk,L)*Kh
    F = Kw*Kh
    # Kw = F'*F/N
    # Kh = F*F'/L
   return F, Kw, Kh
end

function stock_market(N)
    X = MLJBase.@load_smarket
    # size 1250x7
    X = X[1]
    times = [Dates.value(x)*1.0 for x in X[1]]
    f = X.Today
    X_feat = hcat(times, X.Lag1, X.Lag2, X.Lag3, X.Lag4, X.Lag5, X.Volume)
    X_feat = X_feat ./ var(X_feat,dims=1)
    idx = randperm(length(f))[1:N]
    K = gaussian_kernel(X_feat[idx,:],10)
    return f[idx], X_feat[idx,:], K, [1]
end

function labor(N)
    X = RDatasets.dataset("Ecdat","LaborSupply")
    # size 5320x52
    f = X.LNWG
    X_feat = hcat(X.Disab, X.ID, X.Kids, X.LNHR, X.Year)
    X_feat = X_feat ./ var(X_feat,dims=1)
    idx = randperm(length(f))[1:N]
    K = gaussian_kernel(X_feat[idx,:],10)
    # K = (1/N)*X_feat[idx,:]*X_feat[idx,:]'
    return f[idx],X_feat[idx,:], K, [1]
end

function housing_matrix(N)
    X = readdlm("data/housing.data")
    #size 506x14
    idx = randperm(size(X,1))[1:N]
    X = X[idx,:]
    #data matrix
    F =  Array{Float64,2}(undef,N,1)
    F[:,:] = X[:,end]
    #feature matrix
    X = X[1:N,1:end-1]
    #normalize features
    X = X ./ var(X,dims=1)
    # X = X .- mean(X,dims=1)
    # W = sqrt(pinv(X*X'))
    # X = W*X
    Kw = gaussian_kernel(X,1)
    Kh =  Array{Float64,2}(undef,1,10)
    Kh[1,1] = 1
    return F, X, Kw, Kh
end

function drug_matrix(N,L; delta = 0)
    Kw = readdlm("data/drug_target/drug-drug_similarities_2D.txt")
    Kh = readdlm("data/drug_target/target-target_similarities_WS.txt")
    F = readdlm("data/drug_target/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt")
    if N == -1; N = size(Kw)[1]; end
    if L == -1; L = size(Kh)[1]; end
    Kw = Kw[1:N,1:N]
    Kh = Kh[1:L,1:L]
    er = (eigen(Kw)...,)[1]
    ec = (eigen(Kh)...,)[1]
    F = F[1:N,1:L]
    if delta > 0; F = F ./ (F .+maximum(F))*delta; end
    return F, Kw ./ maximum(er), Kh ./ maximum(ec)
end

function wine_data(N)
    # X1 = readdlm("data/wine/winequality-red.csv",';')
    X2 = readdlm("data/wine/winequality-white.csv",';')
    # size: 4898x12
    # X = vcat(X1,X2)
    X = X2
    idx = randperm(size(X,1))[1:N]
    F = X[idx,end]
    X = X[idx,1:end-1]
    W = pinv(X*X')
    # X = W*X
    X = X ./ var(X,dims=1)
    Kw = gaussian_kernel(X,1)
    Kh =  Array{Float64,2}(undef,1,1)
    Kh[:] = [1]
    return F, X,Kw, Kh
end

function temp_matrix(N,L)
        JLD2.@load "data/temp_data.jld" F A
        F = F[1:N,1:L]
        A = A[1:N,1:N]
        A = ceil.((A'+A)/100)
        Nnodes = size(A,1)
        g = DiGraph(size(A,1))
        add_edge! = LightGraphs.add_edge!
        for i = 1:size(A,1)
            for j = 1:size(A,2)
                if A[i,j] == 1
                    add_edge!(g,i,j)
                end
            end
        end
        amat=Matrix(adjacency_matrix(g))
        dist_matrix = zeros(size(A))
        for i = 1:size(A,1)
            dist_matrix[i,:] = dijkstra_shortest_paths(g,i).dists'
        end
        dist_matrix[(LinearIndices(A))[findall(dist_matrix.==Inf)]] .= 10000000
        dist_matrix[findall(dist_matrix.==Inf)] .= 10000000
        P = exp.(-(Nnodes^2*dist_matrix/sum(dist_matrix)))
        Ar = P
        vec = zeros(L)
        vec[1:6] .= 1
        vec[end-6:end] .= 1
        Ac = zeros(L,L)
        for i = 1:L
            Ac[i,:] = circshift(vec,(i,))
        end
        return Ar[1:N,1:N], Ac[1:L,1:L], F[1:N,1:L]
end

function laplacian(A)
    return diagm(A*ones(size(A,1))) - A
end

function laplacian_kernel(X, pow)
    if pow == -1
        return inv(X+I)
    elseif pow == -2
        return inv(X)
    end
    ev,Q = (eigen(X)...,)
    idx = findmin(ev)[2]
    ev[idx] = 0
    return Q*diagm(1 ./ exp(pow*ev/2))*inv(Q)
end

function temp_data(N,L)
    Ar, Ac, M = temp_matrix(N,L)
    Kw = laplacian_kernel(laplacian(Ar),-1)
    Kh = laplacian_kernel(laplacian(Ac),-1)
    Kw = (Kw+Kw')/2
    Kh = (Kh+Kh')/2
    return M, Kw, Kh
end


function load_mushrooms(path)
    features = 0
    labels = 0
        c = h5open(path,"r") do file
           features = read(file, "features")
           labels = read(file, "labels")
        end
    adj = abs.(labels.+labels')-1
    return features'*1,labels,adj
end

function mushroom_matrix(N, L)
    JLD2.@load "data/mushrooms.jld" Pr labels Y
    N_labels = size(Pr,1)
    #size 5643
    # N_labels = 1000
    # prob = ones(N)*(N_labels/N)
    # rvec = rand(length(prob))
    # idx = findall(prob-rvec .> 0)
    idx = randperm(N_labels)[1:N]
    Pr = Pr[idx,:]
    labels = labels[idx]
    F = Y[idx,idx]
    Kw = cor(Pr,Pr,dims=2)
    # Kw = Pr*Pr'
    # Kw = gaussian_kernel(Pr,10)
    if L == 1
        Kh =  Array{Float64,2}(undef,1,1)
        Kh[:] = [1]
        F = zeros(N,1)
        F[:] = labels
    else
        Kh = Kw
    end
return F,Pr,Kw,Kh
end

function data_matrices(name,N,L; rank=1)
    if name == "temp"
        F,Kw,Kh = temp_data(N,L)
    elseif (name == "temp_cov") || (name == "temp_cov_prev")
        F,Kw,Kh =  temp_covkernel(name,N,L)
    elseif (name == "temp_gauss")
        F,Kw,Kh =  temp_gausskernel(N,L,1e6)
    elseif (name == "synth")
        F,Kw,Kh = synth_matrix(N,L)
    elseif (name == "synth_rank")
        F,Kw,Kh = synth_rank_matrix(N,L,rank)
    elseif (name == "synth_cov")
        F,Kw,Kh = synth_matrix_cov(N,L)
    elseif (name == "drugs")
        F,Kw,Kh = drug_matrix(N,L)
    elseif (name == "drugs_norm")
        F,Kw,Kh = drug_matrix(N,L,delta=0.5)
    elseif (name == "mushrooms")
        F,X,Kw,Kh = mushroom_matrix(N,L)
    elseif (name == "housing")
        F,X,Kw,Kh = housing_matrix(N)
    elseif (name == "wine")
        F,X,Kw,Kh = wine_data(N)
    elseif (name == "stocks")
        F,X,Kw,Kh = stock_market(N)
    elseif (name == "labor")
        F,X,Kw,Kh = labor(N)
    else
        error("Wrong data name")
    end
    return F,X,make_PSD(Kw),make_PSD(Kh)
end

function temp_covkernel(name,N,L)
    JLD2.@load "data/temp_data.jld" F F_prev A
    if name == "temp_cov"
        F_c = F
    elseif name == "temp_cov_prev"
        F_c = F_prev
    end
    F = F[1:N,1:L]
    F_c = F_c[1:N,1:L]
    Kw = cov(F_c')
    Kh = cov(F_c)
    return F, Kw, Kh
end

function temp_gausskernel(N,L,sigma)

    JLD2.@load "data/temp_data.jld" F F_prev A
    F = F[1:N,1:L]
    F_prev = F_prev[1:N,1:L]
    Kw = gaussian_kernel(F_prev,sigma)
    Kh = gaussian_kernel(F_prev',sigma)
    return F, Kw, Kh
end

function gaussian_kernel(X,sigma)
    N = size(X,1)
    K = zeros(N,N)
    for j = 1:N
        for i = 1:N
            K[i,j] = exp(-(norm(X[i,:] - X[j,:])/sigma))
        end
    end
return K


end
