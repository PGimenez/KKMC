
using Test, AugmentedGaussianProcesses, KernelFunctions, LinearAlgebra, Statistics, Random, MLJ, MLJBase, KKMC
Random.seed!(1)

name="housing"
N = 500
s=221
r=5
y,X,K,Kh = KKMC.data_matrices(name,N,1,rank=1)
y = y[:]
k = KernelFunctions.transform(ExponentialKernel(),1)
Kf = kernelmatrix(k,X,obsdim=1)

@testset "Kernel matrix" begin
    @test sum(abs.(Kf .- KKMC.gaussian_kernel(X,1))) < 0.1
    @test sum(abs.(Kf .- K)) < 0.1
end

@testset "KRR model" begin
    krr_model = KRRModel(1e-5,k)
    krr = machine(krr_model,X,y)
    MLJ.fit!(krr,verbosity=-1)
    MLJ.predict(krr)
    @test abs(mean(MLJ.predict(krr) .- y))  < 1

    t = 10
    MLJ.fit!(krr,rows=1:N-t,verbosity=-1)
    MLJ.predict(krr,rows=N-t:N)
    @test mean(abs.(MLJ.predict(krr,rows=N-t:N) .- y[N-t:N]))  < 20
end


@testset "Learning network LKRR" begin
    t = 450
    ys = source(y)
    Xs = source(X)
    ls_model = LeverageSampler(LeverageSampling(),1,t,2,k)
    ls = machine(ls_model,Xs,ys)
    MLJ.fit!(ls,verbosity=-1)

    yt = KKMC.transform(ls,ys)
    Xt = KKMC.transform(ls,Xs)
    wt = KKMC.transform(ls,source(0))

    krr_model = KRRModel(1e-8,k)
    krr = machine(krr_model, Xt, yt, wt)
    MLJ.fit!(krr,verbosity=-1)
    zhat = MLJ.predict(krr,Xs)
    @test mean(abs.(zhat()[1:N-t,:] .- y[1:N-t])) < 0.2

    # test inverse transform selecting test samples excluding training ones
    yhat = KKMC.inverse_transform(ls,zhat)
    rms_network = tuple_rms(yhat(),y)
    @test abs(rms_network - 5.7) < 0.1

    # test LKRRModel 
    ls_model.rng = 1
    lkrr_model = LKRRModel(krr_model, ls_model)
    lkrr = machine(lkrr_model, X, y)
    MLJ.fit!(lkrr,force=true,verbosity=-1)
    MLJ.predict(lkrr)
    @test abs(tuple_rms(MLJ.predict(lkrr),y) - rms_network) == 0
end

@testset "Normal KRR vs LKRR with uniform weights vs GP" begin
    r = 10
    s = 50
    krr_model = KRRModel(1e-5,k)
    krr = machine(krr_model,X,y)
    holdout = Holdout(fraction_train = s/N, shuffle=true)
    strat = [MLJBase.train_test_pairs(holdout,1:N)[1] for x in 1:r]
    result_krr = evaluate!(krr, resampling=strat,  measure=rms, verbosity=-1,check_measure=false)

    ls_model = LeverageSampler(KKMC.UniformSampling(),1.0,s,1,k)
    lkrr_model = LKRRModel(krr_model,ls_model)
    lkrr = machine(lkrr_model, X, y)
    AD = AllData(N)
    result_lkrr = evaluate!(lkrr, resampling=AD, repeats=r, measure=tuple_rms, verbosity=-1,check_measure=false)
    @test abs(result_krr.measurement[1] - result_lkrr.measurement[1]) < 1

    gp_model = GaussianProcess(k, 1e-5, false)
    gp = machine(gp_model,X,y)
    result_gp = evaluate!(gp, resampling=strat, measure=rms, verbosity=-1,check_measure=false)
    @test abs(result_gp.measurement[1] - result_lkrr.measurement[1]) < 1
end
