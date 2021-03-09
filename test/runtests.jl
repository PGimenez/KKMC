
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
    Random.seed!(1920)
    @test sum(abs.(Kf .- KKMC.gaussian_kernel(X,1))) < 0.1
    @test sum(abs.(Kf .- K)) < 0.1
end

@testset "KRR model" begin
    Random.seed!(1920)
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
    Random.seed!(1920)
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
    @test mean(abs.(zhat()[1:N-t,:] .- y[1:N-t])) < 0.3

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

    # Leverage sampling
    ls_model = LeverageSampler(LeverageSampling(),1,t,2,k)
    lkrr_model = LKRRModel(krr_model, ls_model)
    lkrr = machine(lkrr_model, X, y)
    MLJ.fit!(lkrr,force=true,verbosity=-1)
    MLJ.predict(lkrr)
    @test abs(tuple_rms(MLJ.predict(lkrr),y) - 3.24) < 0.1

    # Greedy leverage sampling
    ls_model = LeverageSampler(GreedyLeverageSampling(),1,t,2,k)
    lkrr_model = LKRRModel(krr_model, ls_model)
    lkrr = machine(lkrr_model, X, y)
    MLJ.fit!(lkrr,force=true,verbosity=-1)
    MLJ.predict(lkrr)
    @test abs(tuple_rms(MLJ.predict(lkrr),y) - 0.92) < 0.1
end

@testset "Normal KRR vs LKRR with uniform weights vs GP" begin
    Random.seed!(1920)
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
    @test abs(result_krr.measurement[1] - result_lkrr.measurement[1]) < 1.5

    gp_model = GaussianProcess(k, 1e-5, false)
    gp = machine(gp_model,X,y)
    result_gp = evaluate!(gp, resampling=strat, measure=rms, verbosity=-1,check_measure=false)
    @test abs(result_gp.measurement[1] - result_lkrr.measurement[1]) < 1.5
end


# @testset "Learning network LGP" begin
    Random.seed!(1920)
    t = 450
    ys = source(y)
    Xs = source(X)
    ls_model = LeverageSampler(LeverageSampling(),1,t,2,k)
    ls = machine(ls_model,Xs,ys)
    MLJ.fit!(ls,verbosity=-1)

    yt = KKMC.transform(ls,ys)
    Xt = KKMC.transform(ls,Xs)

    gp_model = GaussianProcess(k, 1e-5, true)
    gp = machine(gp_model, Xt, yt)
    MLJ.fit!(gp,verbosity=-1)
    zhat = MLJ.predict(gp,Xs)
    @test mean(abs.(zhat()[1:N-t,:] .- y[1:N-t])) < 0.3

    # test inverse transform selecting test samples excluding training ones
    yhat = KKMC.inverse_transform(ls,zhat)
    rms_network = tuple_rms(yhat(),y)
    @test abs(rms_network - 5.7) < 0.1

    # test LKRRModel 
    ls_model.rng = 1
    lgp_model = LGP(gp_model, ls_model)
    lgp = machine(lgp_model, X, y)
    MLJ.fit!(lgp,force=true,verbosity=-1)
    MLJ.predict(lgp)
    @test abs(tuple_rms(MLJ.predict(lgp),y) - rms_network) == 0

    # Leverage sampling
    # ls_model = LeverageSampler(LeverageSampling(),1,t,2,k)
    # lgp_model = LGP(gp_model, ls_model)
    # lgp = machine(lgp_model, X, y)
    # MLJ.fit!(lgp,force=true,verbosity=-1)
    # MLJ.predict(lgp)
    # @test abs(tuple_rms(MLJ.predict(lgp),y) - 3.24) < 0.1

    # # Greedy leverage sampling
    # ls_model = LeverageSampler(GreedyLeverageSampling(),1,t,2,k)
    # lgp_model = LGP(gp_model, ls_model)
    # lgp = machine(lgp_model, X, y)
    # MLJ.fit!(lgp,force=true,verbosity=-1)
    # MLJ.predict(lgp)
    # @test abs(tuple_rms(MLJ.predict(lgp),y) - 0.92) < 0.1
# end

# config_test = SimConfig( config_name = "test", data_types = ["housing"], size = (301,1), samples = Int.(collect(1:30:150)), passive = true, weighted = true, grid = false, mu_range = [1e-6,1e1,10], alpha_range = [1e-8,1e1,10], hyper_points = 50, rea = 10, tune_rea = 1, SNR = 1e15,
# )

# KKRR_unif_config = LKRRAlgConfig( name = "Uniform", sampling = UniformSampling(), grid=false, constructor = :(KRR))
# KKRR_lev_config = LKRRAlgConfig( name = "Leverage", sampling = LeverageSampling(), grid=false, constructor = :(KRR))
# alg_list = [KRR_config,KKRR_unif_config,KKRR_lev_config,KKRR_greedy_config,GP_config]
# fit_models(config_test,alg_list)
