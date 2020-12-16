#https://github.com/PetrKryslUCSD/FinEtools.jl/blob/master/src/FinEtools.jl
module KKMC

include("KKMCEX.jl")
include("KernelMF.jl")

using .KKMCEX
using .KernelMF

function MatrixToVectorData(data::MatrixData)
    return VectorData(data.N,data.L,vec(data.F),data.Kw,data.Kh)
end

fit!(model::KMF,data::Any,M::Array{Float64,2}) = KernelMF.fit!(model::KMF,data::Any,M::Array{Float64,2})
fit!(model::KRR,data::VectorData,m::Array{Float64,1}) = KKMCEX.fit!(model::KRR,data,m)
predict!(model::MF,data::MatrixData) = KernelMF.predict!(model::MF,data.F)
predict!(model::KRR,data::VectorData) = KKMCEX.predict!(model::KRR,data)

export KRR, VectorData, fit!, predict!, SamplingMatrix, VectorSampler, KronVectorSampler, sampler, KernelRegressor, TSKRR, RTSKRR, build_kron_matrix

export MF, KMF, RGKMF, fit!, predict!, MatrixData

export MatrixToVectorData
end # module
