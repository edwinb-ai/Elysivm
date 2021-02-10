module LeastSquaresSVM

using LinearAlgebra
using KernelFunctions
using Krylov
using Tullio
using StatsBase
import MLJModelInterface

const MMI = MLJModelInterface
const BL = LinearAlgebra.BLAS

export SVM, LSSVC, LSSVR, KernelRBF, svmtrain, svmtrain_mc, svmpredict, LSSVClassifier,
LSSVRegressor, FixedSizeSVR, FixedSizeRegressor

include("types.jl")
include("utils.jl")
include("tools.jl")
include("training.jl")
include("fixed_training.jl")
include("mlj_interface.jl")
# Some precompilation for the Tullio tools
include("precompile.jl")
_precompile_tullio()

end
