using Parameters, Plots, JLD2
using KernelFunctions: Kernel
import KernelFunctions

@with_kw struct SimConfig
    config_name::String
    data_types::Array{String,1}
    size::Tuple{Int64,Int64}
    samples::Array{Int,1}
    grid::Bool
    passive::Bool
    weighted::Bool
    mu_range::Array{Float64,1}
    alpha_range::Array{Float64,1}
    hyper_points::Int64
    rea::Int64
    tune_rea::Int64
    SNR::Int64
end

abstract type AlgConfig end

@with_kw mutable struct KRRAlgConfig <: AlgConfig
    name::String
end

@with_kw mutable struct LKRRAlgConfig <: AlgConfig
    name::String
    constructor::Any
    sampling::PassiveSampling
    grid::Bool
    options::Dict = Dict()
    tol::Float64 = 1e-6
end

@with_kw mutable struct GPAlgConfig <: AlgConfig
    name::String
    constructor::Any
    sampling::PassiveSampling
    kernel::Kernel
    noise::Float64
    opt_noise::Bool
    tol::Float64 = 1e-6
end

function self_tuning_lkrr(simconf::SimConfig, algconf::LKRRAlgConfig, name)
    krr_model = KRRModel(mu=1e-8, kernel=DataKernels[name])
    ls_model = LeverageSampler(algconf.sampling,1,1,1,DataKernels[name])
    lkrr_model = LKRRModel(krr_model,ls_model)

    N = simconf.size[1]*simconf.size[2]
    strat = [(collect(1:N),collect(1:N)) for i in 1:simconf.tune_rea]
    AD = AllData(N)
    r_mu = range(lkrr_model, :(KRR.mu), lower=simconf.mu_range[1], upper=simconf.mu_range[2], scale=:log10);
    r_alpha = range(lkrr_model, :(LS.alpha), lower=simconf.alpha_range[1], upper=simconf.alpha_range[2], scale=:log10);
    tune_rea = simconf.tune_rea
    if algconf.sampling isa GreedyLeverageSampling; tune_rea = 1; end
    param_ranges = [r_mu, r_alpha]
    if algconf.sampling isa UniformSampling; param_ranges = r_mu; end
    self_tuning_regressor = TunedModel(model=lkrr_model, tuning=MLJ.LatinHypercube(gens=2, popsize=120), n=simconf.hyper_points, resampling=AD, repeats=tune_rea, range=param_ranges, measure=tuple_rms);
    return self_tuning_regressor
end

function run_simulation(cfg, algconf_list)
    result_curves_conf = Array{Array{NamedTuple}}(undef,length(algconf_list))
    for (n,name) in enumerate(cfg.data_types)
        println(name)
        result_curves = Array{NamedTuple}(undef,length(algconf_list),1)
        @time for (m,algconf) in enumerate(algconf_list)
            @time result_curves[m] = run_alg(algconf,cfg,name)
        end
        result_curves_conf[n] = result_curves
    end
    return result_curves_conf
end
export run_simulation

function run_alg(algconf::LKRRAlgConfig,cfg,name)
    N = convert(Int64,cfg.size[1])
    F,X,K,Kh = data_matrices(name,N,1,rank=1)
    N = length(F)
    f = F[:]
    AD = AllData(N)
    test_err = zeros(length(cfg.samples))
    for (i,s) in enumerate(cfg.samples)
        self_tuned_model = self_tuning_lkrr(cfg,algconf,name)
        # self_tuned_wrapper = TunedLKRRModel(self_tuned_model,false,LKRRModel())
        # r_s = range(self_tuned_wrapper, :(tuner.model.LS.s), values=cfg.samples);
        self_tuned_model.model.LS.s = s
        tuned_lkrr = machine(self_tuned_model,X,f)
        MLJ.fit!(tuned_lkrr,verbosity=-1)
        plot_param_search(algconf,cfg,name,[tuned_lkrr])
        lkrr_model = report(tuned_lkrr).best_history_entry.model
        lkrr = machine(lkrr_model,X,f)
        rea = cfg.rea
        if algconf.sampling isa GreedyLeverageSampling; rea = 1; end
        result = evaluate!(lkrr, resampling=AD, repeats=rea, measure=tuple_rms, verbosity=-1,check_measure=false)
        test_err[i] = result.measurement[1] 
    end
    return (parameter_values = cfg.samples, measurements = test_err)
end
    # if algconf.sampling isa GreedyLeverageSampling; rea = 1; end
    # return MLJ.learning_curve(tuned_lkrr; range=r_s, resolution=10, resampling=AD, repeats=rea, measure=tuple_rms, verbosity=-1)
    # for (i,s) in enumerate(cfg.samples)
        # holdout = Holdout(fraction_train = s/N, shuffle=true)
        # strat = [(randperm(N)[1:s],1:N) for x in 1:cfg.tune_rea]
        # self_tuning_model = TunedModel(model=krr_model, tuning=MLJ.LatinHypercube(gens=2, popsize=120), n=cfg.hyper_points, resampling=strat, range=r_mu, measure=rms);
        # self_tuning = machine(self_tuning_model,X,f)
        # MLJ.fit!(self_tuning,verbosity=-1)
        # krr_model = report(self_tuning).best_history_entry.model
        # krr = machine(krr_model,X,f)
        # strat = [(randperm(N)[1:s],1:N) for x in 1:cfg.rea]
        # result = evaluate!(krr, resampling=strat, measure=rms, verbosity=-1,check_measure=false)
        # test_err[i] = result.measurement[1] 
    # end
# end
function plot_param_search(algconf::LKRRAlgConfig,cfg,name,fitted_models)
    return 1
end

function plot_param_search(algconf::LKRRAlgConfig,cfg,name,fitted_models)
        # plot(0,0,xlabel="s",ylabel="RMS")
        if algconf.sampling isa UniformSampling; return 0; end
        figpath = "plots/debug/$(cfg.config_name)/"
        mkpath(figpath)
        for mach in fitted_models
            plot(mach)
            savefig("$figpath/fitted_$(algconf.name)_$(name)_$(cfg.size[1])_$(mach.fitresult.model.LS.s).pdf")
            closeall()
        end
end

function run_alg(algconf::GPAlgConfig,cfg,name)
    N = convert(Int64,cfg.size[1])
    f,X,K,Kh = data_matrices(name,N,1,rank=1)
    N = length(f)
    f = f[:]
    gp_model = GaussianProcess(DataKernels[name], algconf.noise, algconf.opt_noise)
    gp = machine(gp_model,X,f)
    test_err = zeros(length(cfg.samples))
    for (i,s) in enumerate(cfg.samples)
        holdout = Holdout(fraction_train = s/N, shuffle=true)
        strat = [MLJBase.train_test_pairs(holdout,1:N)[1] for r in 1:cfg.rea]
        result = evaluate!(gp, resampling=strat, measure=rms, verbosity=-1,check_measure=false)
        test_err[i] = result.measurement[1] 
    end
    return (parameter_values = cfg.samples, measurements = test_err)
end

function run_alg(algconf::KRRAlgConfig,cfg,name)
    N = convert(Int64,cfg.size[1])
    f,X,K,Kh = data_matrices(name,N,1,rank=1)
    N = length(f)
    # K = KernelFunctions.kernelmatrix(DataKernels[name],X,obsdim=1)
    # K = hcat(collect(1:N),K)
    # F = table((idx=collect(1:N),val=f[:]))
    f = f[:]
    krr_model = KRRModel(1e-8,DataKernels[name])
    krr = machine(krr_model,K,f)
    test_err = zeros(length(cfg.samples))
    r_mu = range(krr_model, :mu, lower=cfg.mu_range[1], upper=cfg.mu_range[2], scale=:log10);
    for (i,s) in enumerate(cfg.samples)
        holdout = Holdout(fraction_train = s/N, shuffle=true)
        strat = [(randperm(N)[1:s],1:N) for x in 1:cfg.tune_rea]
        self_tuning_model = TunedModel(model=krr_model, tuning=MLJ.LatinHypercube(gens=2, popsize=120), n=cfg.hyper_points, resampling=strat, range=r_mu, measure=rms);
        self_tuning = machine(self_tuning_model,X,f)
        MLJ.fit!(self_tuning,verbosity=-1)
        krr_model = report(self_tuning).best_history_entry.model
        krr = machine(krr_model,X,f)
        strat = [(randperm(N)[1:s],1:N) for x in 1:cfg.rea]
        result = evaluate!(krr, resampling=strat, measure=rms, verbosity=-1,check_measure=false)
        test_err[i] = result.measurement[1] 
    end
    return (parameter_values = cfg.samples, measurements = test_err)
end

function plot_curves(cfg,algconf_list,result_curves)
    for (n,name) in enumerate(cfg.data_types)
        plot(0,0,xlabel="s",ylabel="RMS")
        figpath = "plots/$(cfg.config_name)/$name/"
        mkpath(figpath)
        mkpath("plots/latest")
        for (m,algconf) in enumerate(algconf_list)
            @show result_curves[n][m].measurements
            plot!(result_curves[n][m].parameter_values, result_curves[n][m].measurements, yscale=:log10, label=algconf.name)
        end
        # title(name)
        savefig("$figpath/error.pdf")
        savefig("plots/latest/$(cfg.config_name)-$name-$(cfg.size[1])-error.pdf")
        @save "$figpath/result_curves-$(cfg.config_name)-$name-$(cfg.size[1])-.jld2" result_curves
        closeall()
    end
end

function  run_simulation_list(cfg_list,alg_list)
    for cfg in cfg_list
        result_curves_conf = run_simulation(cfg,alg_list)
        plot_curves(cfg,alg_list,result_curves_conf)
    end
end

export plot_curves
