#https://github.com/PetrKryslUCSD/FinEtools.jl/blob/master/src/FinEtools.jl
module KKMC

include("KKMCEX.jl")
include("KernelMF.jl")
include("leverage_scores.jl")
include("gen_data.jl")

using .LeverageScores

include("simulations.jl")
include("mlj_interface.jl")
include("gp_interface.jl")


export KRR, VectorData, fit!, predict!, SamplingMatrix, VectorSampler, KronVectorSampler, sampler, KernelRegressor, TSKRR, RTSKRR, build_kron_matrix

export MF, KMF, RGKMF, fit!, predict!, MatrixData

export KRRModel

export PassiveSampling, UniformSampling, UnweightedUniformSampling, LeverageSampling, UnweightedLeverageSampling, GreedyLeverageSampling
export get_lscores, Leverage, AllData, train_test_pairs, LeverageWeighter, LeverageSampler, transform, LKRRModel, tuple_rms

export  SimConfig, KRRAlgConfig, LKRRAlgConfig, GPAlgConfig, LGPAlgConfig, plot_curves, paper_plots, run_simulation_list, fit_models
export TunedLKRRModel, self_tuning_lkrr

export GaussianProcess, LGP
end # module
