using Parameters

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
    sampling::String
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
    self_tuning_regressor = TunedModel(model=lkrr_model, tuning=MLJ.RandomSearch(), n=simconf.train_rea, resampling=AD, repeats=simconf.train_rea, range=[r_mu, r_alpha], measure=tuple_rms);
    return self_tuning_regressor
end

# function run_simulation(config_list, algconf_list)
    # for config in config_list
        # for name in config.data_types
            # for matrix_size in config.sizes
                # println(matrix_size)
                # N = convert(Int64,matrix_size[1])
                # L = convert(Int64,matrix_size[2])
                # F,Kw,Kh = data_matrices(name,N,L,rank=1)
                # normF = sum(F.^2)
                # figpath = "plots/$(config.config_name)/$name/$matrix_size/"
                # @load "$figpath/ho_results.jld2" opt_param
                # alg_names = [algconf.name for algconf in algconf_list]
                # error_methods = Dict(name =>  Dict("mu" => opt_param[name][1], "alpha" => opt_param[name][2], "error" => Matrix, "leverage" => zeros(N*L), "probability" =>zeros(N*L)) for name in alg_names)
                # error_samples = SharedArray{Float64}(length(config.samples))
                # for (m,model_conf) in enumerate(algconf_list)
                    # model = eval(model_conf.constructor)()
                    # Random.seed!(1920)
                    # lscores_mat = zeros(length(F),length(config.samples))
                    # probs_mat = zeros(length(F),length(config.samples))
                    # for (j,s) in enumerate(config.samples)
                        # print("$s, ")
                        # s = convert(Int64,round(s*N*L))
                        # mu = opt_param[model_conf.name][j,1]
                        # alpha = opt_param[model_conf.name][j,2]
                        # lscores = get_lscores(model_conf.sampling,Kw,Kh,alpha)
                        # probs= get_probs(data,lscores,model_conf.grid)
                        # lscores_mat[:,j] = lscores
                        # # probs_mat[:,j] = probs
                        # model = eval(model_conf.constructor)()
                        # samples = (model_conf.grid == true) ? [Int(round(sqrt(s))),Int(round(sqrt(s)))] : s
                        # error_rea = SharedArray{Float64}(config.rea)
                        # @sync @distributed for r = 1:config.rea
                            # Random.seed!(convert(Int32,j*r))
                            # matrix_sampler = get_sampler(model,probs,samples,config.weighted)
                            # M = sample_matrix(data,matrix_sampler,config.SNR*1.0)
                            # set_model_params!(mu,M,matrix_sampler,data,model_conf,model)
                            # fitmodel!(model,data,M)
                            # error_rea[r] = predict!(model,data)
                        # end
                        # error_samples[j] = mean(error_rea)
                    # end
                    # error_methods[model_conf.name]["error"] = Array(error_samples/normF)
                    # error_methods[model_conf.name]["leverage"] = lscores_mat
                    # # error_methods[model_conf.name]["probability"] = probs_mat
                # end
            # Plots.heatmap(data.F)
            # Plots.savefig("$figpath/Y_$matrix_size.pdf")
            # samples = config.samples
            # @save "$figpath/results.jld2" error_methods alg_names algconf_list config samples matrix_size
            # plot_results("$figpath")
            # flush(stdout)
            # end
        # end
    # end
# end
# export run_simulation
