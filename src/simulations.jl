using Parameters, Plots, PGFPlotsX, JLD2, MLJTuning
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

@with_kw mutable struct LKRRAlgConfig{T} <: AlgConfig
    name::String
    constructor::Any
    sampling::T = PassiveSampling()
    grid::Bool
    options::Dict = Dict()
end

@with_kw mutable struct GPAlgConfig <: AlgConfig
    name::String
    constructor::Any
    kernel::Kernel
    noise::Float64
    opt_noise::Bool
end

@with_kw mutable struct LGPAlgConfig{T} <: AlgConfig
    name::String
    constructor::Any
    sampling::T = PassiveSampling()
    kernel::Kernel
    noise::Float64
    opt_noise::Bool
end


function run_simulation(cfg, algconf_list)
    result_curves_conf = Array{Array{NamedTuple}}(undef,length(algconf_list))
    for (n,name) in enumerate(cfg.data_types)
        println(name)
        result_curves = Array{NamedTuple}(undef,length(algconf_list),1)
        @time for (m,algconf) in enumerate(algconf_list)
            println(algconf.name)
            @time result_curves[m] = run_alg(algconf,cfg,name)
        end
        result_curves_conf[n] = result_curves
    end
    return result_curves_conf
end
export run_simulation

function  run_simulation_list(cfg_list,alg_list)
    for cfg in cfg_list
        result_curves_conf = run_simulation(cfg,alg_list)
        save_curves(cfg,alg_list,result_curves_conf)
        plot_curves(cfg,alg_list)
    end
end

function fit_models(cfg, algconf_list)
    for (n,name) in enumerate(cfg.data_types)
        println(name)
        binpath = "bin/$(cfg.config_name)/$name/"
        mkpath(binpath)
        @time for (m,algconf) in enumerate(algconf_list)
            @time fitted_models = param_search(algconf,cfg,name)
            @save "$binpath/fitted_$(algconf.name)_$(name)_$(cfg.size[1]).jld" fitted_models
            plot_param_search(algconf,cfg,name,fitted_models)
        end
    end
    return 1
end
export run_simulation

function run_alg(algconf,cfg,name)
    N = convert(Int64,cfg.size[1])
    F,X,K,Kh = data_matrices(name,N,1,rank=1)
    N = length(F)
    f = F[:]
    test_err = zeros(length(cfg.samples))
    for (i,s) in enumerate(cfg.samples)
        self_tuned_model = self_tuning_model(algconf,cfg,name,N,s)
        opt_model = tune_model(algconf,cfg,name,self_tuned_model,X,f,s)
        mach = machine(opt_model,X,f)
        sampler, reps = get_resampling(algconf,N,s,cfg.rea)
        error_measure = get_measure(algconf)
        result = evaluate!(mach, resampling=sampler, repeats=reps, measure=error_measure, verbosity=-1,check_measure=false)
        test_err[i] = result.measurement[1] 
    end
    return (parameter_values = cfg.samples, measurements = test_err)
end

tune_model(algconf,cfg,name,self_tuned_model,X,f,s) = self_tuned_model

function tune_model(algconf::Union{KRRAlgConfig,LKRRAlgConfig,LGPAlgConfig},cfg,name,self_tuned_model,X,f,s)
        tuned_mach = machine(self_tuned_model,X,f)
        MLJ.fit!(tuned_mach,verbosity=-1)
        save_param_search(algconf,cfg,name,tuned_mach,s)
        return  report(tuned_mach).best_history_entry.model
end


get_resampling(algconf::Union{LKRRAlgConfig,LGPAlgConfig},N,s,rea) = (AllData(N), rea)
get_resampling(algconf::Union{LKRRAlgConfig{GreedyLeverageSampling},LGPAlgConfig{GreedyLeverageSampling}},N,s,rea) = (AllData(N), 1)
get_resampling(algconf::Union{KRRAlgConfig,GPAlgConfig},N,s,rea) = (holdout_strategy(N,s,rea), 1)
get_measure(algconf::Union{LKRRAlgConfig,LGPAlgConfig}) = tuple_rms
get_measure(algconf::Union{KRRAlgConfig,GPAlgConfig}) = rms

function holdout_strategy(N,s,rea)
        holdout = Holdout(fraction_train = s/N, shuffle=true)
        strat = [MLJBase.train_test_pairs(holdout,1:N)[1] for r in 1:rea]
        return strat
end

function self_tuning_model(algconf::KRRAlgConfig,cfg,name,N,s)
    krr_model = KRRModel(1e-8,DataKernels[name])
    r_mu = range(krr_model, :mu, lower=cfg.mu_range[1], upper=cfg.mu_range[2], scale=:log10);
    holdout = Holdout(fraction_train = s/N, shuffle=true)
    strat = holdout_strategy(N,s,cfg.tune_rea)
    return TunedModel(model=krr_model, tuning=MLJ.LatinHypercube(gens=2, popsize=120), n=Int(round(0.5*cfg.hyper_points)), resampling=strat, range=r_mu, measure=rms);
end

function self_tuning_model(algconf::LKRRAlgConfig,cfg,name,N,s)
    krr_model = KRRModel(mu=1e-8, kernel=DataKernels[name])
    ls_model = LeverageSampler(algconf.sampling,1,s,1,DataKernels[name])
    lkrr_model = LKRRModel(krr_model,ls_model)

    strat = [(collect(1:N),collect(1:N)) for i in 1:cfg.tune_rea]
    AD = AllData(N)
    r_mu = range(lkrr_model, :(KRR.mu), lower=cfg.mu_range[1], upper=cfg.mu_range[2], scale=:log10);
    r_alpha = range(lkrr_model, :(LS.alpha), lower=cfg.alpha_range[1], upper=cfg.alpha_range[2], scale=:log10);
    tune_rea = cfg.tune_rea
    hpoints = cfg.hyper_points
    if algconf.sampling isa GreedyLeverageSampling; tune_rea = 1; end
    param_ranges = [r_mu, r_alpha]
    if algconf.sampling isa UniformSampling; param_ranges = r_mu; hpoints = Int(round(0.5*cfg.hyper_points)); end
    self_tuning_regressor = TunedModel(model=lkrr_model, tuning=MLJ.LatinHypercube(gens=2, popsize=120), n=hpoints, resampling=AD, repeats=tune_rea, range=param_ranges, measure=tuple_rms);
    return self_tuning_regressor
end

function self_tuning_model(algconf::GPAlgConfig,cfg,name,N,s)
    return GaussianProcess(DataKernels[name], algconf.noise, algconf.opt_noise)
end

function self_tuning_model(algconf::LGPAlgConfig,cfg,name,N,s)
    gp_model = GaussianProcess(DataKernels[name], algconf.noise, algconf.opt_noise)
    AD = AllData(N)
    ls_model = LeverageSampler(algconf.sampling,1,s,1,DataKernels[name])
    lgp_model = LGP(gp_model, ls_model)
    r_alpha = range(lgp_model, :(LS.alpha), lower=cfg.alpha_range[1], upper=cfg.alpha_range[2], scale=:log10);
    tune_rea = cfg.tune_rea
    rea = cfg.rea
    hpoints = Int(round(0.5*cfg.hyper_points))
    if algconf.sampling isa UniformSampling; hpoints = 2; end # uniform sampling is the same for all values of alpha
    if algconf.sampling isa GreedyLeverageSampling; tune_rea = 1; end
    return TunedModel(model=lgp_model, tuning=MLJ.LatinHypercube(gens=2, popsize=120), n=hpoints, resampling=AD, repeats=tune_rea, range=r_alpha, measure=tuple_rms);
end

function save_curves(cfg,algconf_list,result_curves)
    for (n,name) in enumerate(cfg.data_types)
        figpath = "plots/$(cfg.config_name)/$name"
        mkpath(figpath)
        @JLD2.save "$figpath/result_curves-$(cfg.config_name)-$name-$(cfg.size[1]).jld2" result_curves
    end
end

function plot_curves(cfg,algconf_list)
    colors=["red","green","blue","yellow","cyan","magenta","olive","orange","black"]
    # markers=["*","diamond*","asterisk"]
    lines=[:solid,:solid,:solid,:solid,:solid,:dash,:dash,:dash,:dash]
    for (n,name) in enumerate(cfg.data_types)
        figpath = "plots/$(cfg.config_name)/$name"
        @JLD2.load "$figpath/result_curves-$(cfg.config_name)-$name-$(cfg.size[1]).jld2" result_curves
        plot(0,0,xlabel="s",ylabel="RMS")
        figpath = "plots/$(cfg.config_name)/$name/"
        mkpath(figpath)
        mkpath("plots/latest")
        for (m,algconf) in enumerate(algconf_list)
            plot!(result_curves[n][m].parameter_values, result_curves[n][m].measurements, yscale=:log10, label=algconf.name,markershape=:auto, color=colors[m],markerstrokealpha=1,markerstrokewidth=0,linestyle=lines[m],legend=:topright,background="gray94")
        end
        # title(name)
        savefig("$figpath/error.pdf")
        savefig("plots/latest/$(cfg.config_name)-$name-$(cfg.size[1])-error.pdf")
        # closeall()
    end
end

function plot_param_search(algconf,cfg,name)
    return 1
end


function save_param_search(algconf::KRRAlgConfig,cfg,name,fitted_model,s)
        # if algconf.sampling isa UniformSampling; return 0; end
        figpath = "plots/debug/$(cfg.config_name)/"
        mkpath(figpath)
        # @save "$figpath/machine-fitted_$(algconf.name)_$(name)_$(cfg.size[1])_$(fitted_model.fitresult.model.LS.s).jld2" fitted_model cfg algconf name
        MLJ.save("$figpath/machine-fitted_$(algconf.name)_$(name)_$(cfg.size[1])_$(s).jlso", fitted_model) 
end

function save_param_search(algconf::Union{LKRRAlgConfig,LGPAlgConfig},cfg,name,fitted_model,s)
        # if algconf.sampling isa UniformSampling; return 0; end
        figpath = "plots/debug/$(cfg.config_name)/"
        mkpath(figpath)
        # @save "$figpath/machine-fitted_$(algconf.name)_$(name)_$(cfg.size[1])_$(fitted_model.fitresult.model.LS.s).jld2" fitted_model cfg algconf name
        MLJ.save("$figpath/machine-fitted_$(algconf.name)_$(name)_$(cfg.size[1])_$(fitted_model.fitresult.model.LS.s).jlso", fitted_model) 
end

function plot_param_search(algconf::LKRRAlgConfig,cfg,name)
        # plot(0,0,xlabel="s",ylabel="RMS")
        if algconf.sampling isa UniformSampling; return 0; end
        figpath = "plots/debug/$(cfg.config_name)/"
        mkpath(figpath)
        for s in cfg.samples
            # @JLD2.load "$figpath/machine-fitted_$(algconf.name)_$(name)_$(cfg.size[1])_$s.jld2" fitted_model cfg algconf name
            fitted_model = machine("$figpath/machine-fitted_$(algconf.name)_$(name)_$(cfg.size[1])_$s.jlso")
            plot(fitted_model)
            savefig("$figpath/fitted_$(algconf.name)_$(name)_$(cfg.size[1])_$s.pdf")
            # closeall()
        end
end


function scatter_param_search(algconf,cfg,name)
    return 1
end

function scatter_param_search(algconf_list,cfg,name)
    plot(0,0,xlabel="s",ylabel="RMS")
    figpath = "plots/debug/$(cfg.config_name)/"
    colors=["red","green","blue","yellow", "cyan","magenta","olive","orange","black"]
    markers=[:circle, :rect,  :diamond, :hexagon, :cross,  :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon]
    mkpath(figpath)
    p1 = 0
    p2 = 1
    for s in cfg.samples
        l = @layout [a;b]
        p1 = scatter()
        p2 = scatter()
        for (m,alg) in enumerate(algconf_list)
            if alg isa Union{KRRAlgConfig,LKRRAlgConfig}
                # @JLD2.load "$figpath/machine-fitted_$(alg.name)_$(name)_$(cfg.size[1])_$s.jld2" fitted_model cfg algconf name
                fitted_model = machine("$figpath/machine-fitted_$(alg.name)_$(name)_$(cfg.size[1])_$s.jlso")
                r = report(fitted_model).plotting
                z = r.measurements
                x = r.parameter_values[:,1]
                scatter!(p1,x,z,xscale=:log10,yscale=:log10, label=alg.name,markershape=markers[m], color=colors[m],markerstrokealpha=1,markerstrokewidth=0,legend=:topright,background="gray94")
                xaxis!("\\mu")
                if in(:sampling,fieldnames(typeof(alg))) && alg.sampling isa Union{LeverageSampling,GreedyLeverageSampling}
                    y = r.parameter_values[:,2]
                    scatter!(p2,y,z,xscale=:log10,yscale=:log10, label=alg.name,markershape=markers[m], color=colors[m],markerstrokealpha=1,markerstrokewidth=0,legend=:topright,background="gray94")
                xaxis!("\\alpha")
                yaxis!("RMSE")
                end
            end
            if alg isa LGPAlgConfig && !(alg.sampling isa UniformSampling)
                # @JLD2.load "$figpath/machine-fitted_$(alg.name)_$(name)_$(cfg.size[1])_$s.jld2" fitted_model cfg algconf name
                fitted_model = machine("$figpath/machine-fitted_$(alg.name)_$(name)_$(cfg.size[1])_$s.jlso")
                r = report(fitted_model).plotting
                z = r.measurements
                x = r.parameter_values[:,1]
                scatter!(p2,x,z,xscale=:log10,yscale=:log10, label=alg.name,markershape=markers[m], color=colors[m],markerstrokealpha=1,markerstrokewidth=0,legend=:topright,background="gray94")
            end
                xaxis!("\\alpha")
                yaxis!("RMSE")
        end
        plot(p1,p2)
        savefig("$figpath/scatter_fitted_$(name)_$(cfg.size[1])_$s.pdf")
        # closeall()
    end
end


function paper_plots(cfg_list,algconf_list)
    # pgfplotsx()
    for cfg in cfg_list
         plot_curves(cfg,algconf_list)
            scatter_param_search(algconf_list, cfg, cfg.data_types[1])
        for alg in algconf_list
             plot_param_search(alg, cfg, cfg.data_types[1])
        end
    end
end

export plot_curves


@recipe function f(mach::MLJBase.Machine{<:MLJTuning.EitherTunedModel})
    r = report(mach).plotting
    z = r.measurements
    x = r.parameter_values[:,1]
    y = r.parameter_values[:,2]

    r.parameter_scales[1] == :none &&
        (x = string.(x))

    r.parameter_scales[2] == :none &&
        (y = string.(y))

    xsc, ysc = r.parameter_scales

    xguide --> "\\mu"
    yguide --> "\\alpha"
    xscale --> (xsc in [:custom, :linear] ? :identity : xsc)
    yscale --> (ysc in [:custom, :linear] ? :identity : ysc)

    st = get(plotattributes, :seriestype, :scatter)

    if st ∈ (:surface, :heatmap, :contour, :contourf, :wireframe)
        ux = unique(x)
        uy = unique(y)
        m = reshape(z, (length(ux), length(uy)))'
        ux, uy, m
    else
        label --> ""
        seriestype := st
        ms = get(plotattributes, :markersize, 4)
        markersize := MLJTuning._getmarkersize(ms, z)
        markershape --> :circle
        markerstrokealpha --> 0
        marker_z --> z
        x, y
    end
end
