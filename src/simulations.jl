using Parameters, Plots, JLD2

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
    train_rea::Int64
    SNR::Int64
end

abstract type AlgConfig end
@with_kw mutable struct KRRAlgConfig <: AlgConfig
    name::String
    constructor::Any
    sampling::PassiveSampling
    grid::Bool
    options::Dict = Dict()
    tol::Float64 = 1e-6
end

function self_tuning_lkrr(simconf::SimConfig, algconf::KRRAlgConfig)
    krr_model = KRRModel(mu=1e-8, kernel="")
    ls_model = LeverageSampler(algconf.sampling,1,1,1)
    lkrr_model = LKRRModel(krr_model,ls_model)

    N = simconf.size[1]*simconf.size[2]
    strat = [(collect(1:N),collect(1:N)) for i in 1:simconf.train_rea]
    AD = AllData(N)
    r_mu = range(lkrr_model, :(KRR.mu), lower=simconf.mu_range[1], upper=simconf.mu_range[2], scale=:log10);
    r_alpha = range(lkrr_model, :(LS.alpha), lower=simconf.alpha_range[1], upper=simconf.alpha_range[2], scale=:log10);
    train_rea = simconf.train_rea
    if algconf.sampling isa GreedyLeverageSampling; train_rea = 1; end
    self_tuning_regressor = TunedModel(model=lkrr_model, tuning=MLJ.RandomSearch(), n=simconf.hyper_points, resampling=AD, repeats=train_rea, range=[r_mu, r_alpha], measure=tuple_rms);
    return self_tuning_regressor
end

function run_simulation(cfg, algconf_list)
    result_curves_conf = Array{Array{NamedTuple}}(undef,length(algconf_list))
    for (n,name) in enumerate(cfg.data_types)
        N = convert(Int64,cfg.size[1])
        # L = convert(Int64,matrix_size[2])
        F,Kw,Kh = data_matrices(name,N,1,rank=1)
        K = Kw
        F = table((idx=collect(1:N),val=F[:]))
        K = hcat(collect(1:N),K)
        AD = AllData(N)
        result_curves = Array{NamedTuple}(undef,length(algconf_list),1)
        for (m,model_conf) in enumerate(algconf_list)
            self_tuned_model = self_tuning_lkrr(cfg,model_conf)
            r_s = range(self_tuned_model, :(model.LS.s), values=cfg.samples);
            self_tuned_wrapper = TunedLKRRModel(self_tuned_model,false,LKRRModel())
            strat = [(collect(1:N),collect(1:N)) for i in 1:cfg.rea]
            r_s = range(self_tuned_wrapper, :(tuner.model.LS.s), values=cfg.samples);
            tuned_lkrr = machine(self_tuned_wrapper,K,F)
            rea = cfg.rea
            if model_conf.sampling isa GreedyLeverageSampling; rea = 1; end
            result_curves[m] = MLJ.learning_curve(tuned_lkrr; range=r_s, resolution=10, resampling=AD, repeats=rea, measure=tuple_rms, verbosity=0)
        end
        result_curves_conf[n] = result_curves
    end
    return result_curves_conf
end
export run_simulation

function plot_curves(cfg,algconf_list,result_curves)
    plot(0,0,xlabel="s",ylabel="RMS")
    for (n,name) in enumerate(cfg.data_types)
        figpath = "plots/$(cfg.config_name)/$name/"
        mkpath(figpath)
        mkpath("plots/latest")
        for (m,model_conf) in enumerate(algconf_list)
            plot!(result_curves[n][m].parameter_values, result_curves[n][m].measurements, yscale=:log10, label=model_conf.name)
        end
        # title(name)
        savefig("$figpath/error.pdf")
        savefig("plots/latest/$(cfg.config_name)-$name-error.pdf")
    end
    closeall()
    @save "$figpath/result_curves-$(cfg.config_name)-$name.jld2" result_curves
end

function  run_simulation_list(cfg_list,alg_list)
    for cfg in cfg_list
        result_curves_conf = run_simulation(cfg,alg_list)
        plot_curves(cfg,alg_list,result_curves_conf)
    end
end

export plot_curves
