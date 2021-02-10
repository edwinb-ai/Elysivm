##
## Types
##

MMI.@mlj_model mutable struct LSSVClassifier <: MMI.Deterministic
    kernel::Symbol = :rbf::(_ in (:rbf, :linear, :poly))
    γ::Float64 = 1.0::(_ > 0.0)
    σ::Float64 = 1.0::(_ > 0.0)
    degree::Int = 0::(_ >= 0)
end

MMI.@mlj_model mutable struct LSSVRegressor <: MMI.Deterministic
    kernel::Symbol = :rbf::(_ in (:rbf, :linear, :poly))
    γ::Float64 = 1.0::(_ > 0.0)
    σ::Float64 = 1.0::(_ > 0.0)
    degree::Int = 0::(_ >= 0)
end

MMI.@mlj_model mutable struct FixedSizeRegressor <: MMI.Deterministic
    kernel::Symbol = :rbf::(_ in (:rbf, :linear, :poly))
    γ::Float64 = 1.0::(_ > 0.0)
    σ::Float64 = 1.0::(_ > 0.0)
    degree::Int = 0::(_ >= 0)
    subsample::Int = 10::(_ >= 0)
    iters::Int = 50_000::(_ >= 0)
end

##
## Fitting functions
##

function MMI.fit(model::LSSVClassifier, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X; transpose=true) # notice the transpose

    a_target_element = y[1]
    num_classes = length(MMI.classes(a_target_element))

    decode  = MMI.decoder(a_target_element) # for predict method
    y_plain = convert(Array{eltype(Xmatrix)}, MMI.int(y))

    cache = nothing

    if num_classes == 2 # binary classification
        new_y = broadcast(x -> x == 2.0 ? -1.0 : 1.0, y_plain)

        svm = LSSVC(;kernel=model.kernel, γ=model.γ, σ=model.σ)
        fitted = svmtrain(svm, Xmatrix, new_y)

        fitresult = (deepcopy(svm), fitted, decode)

        report = (kernel = model.kernel, γ = model.γ, σ = model.σ)
    else # multiclass classification
        svm = LSSVC(; kernel=model.kernel, γ=model.γ, σ=model.σ)
        fitted = svmtrain_mc(svm, Xmatrix, y_plain, num_classes)
        fitresult = (fitted, decode)
        report = (kernel = model.kernel, γ = model.γ, σ = model.σ)
    end

    return (fitresult, cache, report)
end

function MMI.fit(model::LSSVRegressor, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X; transpose=true) # notice the transpose

    cache = nothing

    svr = LSSVR(; kernel=model.kernel, γ=model.γ, σ=model.σ)
    fitted = svmtrain(svr, Xmatrix, y)

    fitresult = (deepcopy(svr), fitted)

    report = (kernel = model.kernel, γ = model.γ, σ = model.σ)

    return (fitresult, cache, report)
end

function MMI.fit(model::FixedSizeRegressor, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X; transpose=true) # notice the transpose

    cache = nothing

    svr = FixedSizeSVR(; kernel=model.kernel,
    γ=model.γ,
    σ=model.σ,
    subsample=model.subsample,
    iters=model.iters
    )
    fitted = svmtrain(svr, Xmatrix, y)
    fitresult = (deepcopy(svr), fitted)

    report = (kernel = model.kernel,
        γ = model.γ,
        σ = model.σ,
        size = model.subsample,
        iters = model.iters)

    return (fitresult, cache, report)
end

##
## Prediction functions
##

function MMI.predict(model::LSSVClassifier, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew; transpose=true) # notice the transpose
    n_fits = length(fitresult) # number of elements from the fit step

    if n_fits == 3 # binary classification
        (svm, fitted, decode) = fitresult
        results = svmpredict(svm, fitted, Xmatrix)
        broadcast!(x -> x == -1.0 ? 2.0 : 1.0, results, results)
        y = convert(Array{UInt64}, results)
        predictions = decode(y)
    else
        (fitted, decode) = fitresult
        results = svmpredict_mc(fitted, Xmatrix)
        y = convert(Array{UInt64}, results)
        predictions = decode(y)
    end

    return predictions
end

function MMI.predict(model::LSSVRegressor, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew; transpose=true) # notice the transpose
    (svr, fitted) = fitresult
    results = svmpredict(svr, fitted, Xmatrix)

    return results
end

function MMI.predict(model::FixedSizeRegressor, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew; transpose=true) # notice the transpose
    (svr, fitted) = fitresult
    results = svmpredict(svr, fitted, Xmatrix)

    return results
end

##
## Metadata
##

# shared metadata
const LSSVM = (LSSVClassifier, LSSVRegressor, FixedSizeRegressor)

MMI.metadata_pkg.(LSSVM,
    name="LeastSquaresSVM",
    uuid="6bfd0e71-701c-47cd-9c90-5bf8fe84640d",
    url="https://github.com/edwinb-ai/LeastSquaresSVM",
    julia=true,
    license="MIT",
    is_wrapper=false)

MMI.metadata_model(LSSVClassifier,
    input=MMI.Table(MMI.Continuous),
    target=AbstractVector{MMI.Finite},
    weights=false,
    descr="A Least Squares Support Vector Classifier implementation.",
    path="LeastSquaresSVM.LSSVClassifier"
)

MMI.metadata_model(LSSVRegressor,
    input=MMI.Table(MMI.Continuous),
    target=AbstractVector{MMI.Continuous},
    weights=false,
    descr="A Least Squares Support Vector Regressor implementation.",
    path="LeastSquaresSVM.LSSVRegressor"
)

MMI.metadata_model(FixedSizeRegressor,
    input=MMI.Table(MMI.Continuous),
    target=AbstractVector{MMI.Continuous},
    weights=false,
    descr="A Fixed Size Least Squares Support Vector Regressor implementation.",
    path="LeastSquaresSVM.FixedSizeRegressor"
)
