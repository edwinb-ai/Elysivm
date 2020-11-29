"""
    svmtrain(svm::LSSVC, x::AbstractMatrix, y::AbstractVector) -> Tuple

Solves a Least Squares Support Vector Classification problem using the Conjugate Gradient method. In particular, it uses the Lanczos process due to the fact that the matrices are symmetric.

# Arguments
- `svm::LSSVC`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `x::AbstractMatrix`: The data matrix with the features. It is expected that this array is already standardized, i.e. the mean for each feature is zero and its standard deviation is one.
- `y::AbstractVector`: A vector that contains the classes. It is expected that there are only two classes, -1 and 1.

# Returns
- `Tuple`: A tuple containing `x`, `y` and the following two elements:
    - `b`: Contains the bias for the decision function.
    - `α`: Contains the weights for the decision function.
"""
function svmtrain(svm::LSSVC, x::AbstractMatrix, y::AbstractVector)
    n = size(y, 1)
    # Initialize the necessary matrices
    Ω = build_omega(x, y; sigma=svm.σ, kernel=svm.kernel)
    H = Ω + I / svm.γ

    # * Start solving the subproblems
    # First, solve for eta
    (η, stats) = cg_lanczos(H, y)
    # Then, solve for nu
    (ν, stats) = cg_lanczos(H, ones(n))

    # We then compute s
    s = dot(y, η)

    # Finally, we solve the problem for alpha and b
    b = dot(η, ones(n)) / s
    α = ν .- (η * b)

    return (x, y, α, b)
end

"""
    svmpredict(svm::LSSVC, fits, xnew::AbstractMatrix) -> AbstractArray

Uses the information obtained from `svmtrain` such as the bias and weights to construct a decision function and predict new class values.

# Arguments
- `svm::LSSVC`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `fits`: It can be any container data structure but it must have four elements: `x`, the data matrix; `y`, the labels vector; `α`, the weights; and `b`, the bias.
- `xnew::AbstractMatrix`: The data matrix that contains the new instances to be predicted.

# Returns
- `Array`: The labels corresponding to the prediction to each of the instances in `xnew`.
"""
function svmpredict(svm::LSSVC, fits, xnew::AbstractMatrix)
    x, y, α, b = fits
    # Compute the asymmetric kernel matrix in one go
    if svm.kernel == "rbf"
        kern_mat = KernelRBF(x, xnew, svm.σ)
    end
    result = sum(@. kern_mat * y * α; dims=1) .+ b
    # We need to remove the trailing dimension
    result = reshape(result, size(result, 2))

    return sign.(result)
end

"""
    svmtrain(svm::LSSVR, x::AbstractMatrix, y::AbstractVector) -> Tuple

Solves a Least Squares Support Vector Regression problem using the Conjugate Gradient method. In particular, it uses the Lanczos process due to the fact that the matrices are symmetric.

# Arguments
- `svm::LSSVR`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `x::AbstractMatrix`: The data matrix with the features. It is expected that this array is already standardized, i.e. the mean for each feature is zero and its standard deviation is one.
- `y::AbstractVector`: A vector that contains the classes. It is expected that there are only two classes, -1 and 1.

# Returns
- `Tuple`: A tuple containing `x` and the following two elements:
    - `b`: Contains the bias for the decision function.
    - `α`: Contains the weights for the decision function.
"""
function svmtrain(svm::LSSVR, x::AbstractMatrix, y::AbstractVector)
    n = size(y, 1)
    # Here, the omega matrix is just the kernel matrix
    if svm.kernel == "rbf"
        Ω = KernelRBF(x, svm.σ)
    end
    H = Ω + I / svm.γ

    # * Start solving the subproblems
    # First, solve for eta
    (η, stats) = cg_lanczos(H, ones(n))
    # Then, solve for nu
    (ν, stats) = cg_lanczos(H, y)

    # We then compute s
    s = dot(ones(n), η)

    # Finally, we solve the problem for alpha and b
    b = dot(η, y) / s
    α = ν .- (η * b)

    return (x, α, b)
end

"""
    svmpredict(svm::LSSVR, fits, xnew::AbstractMatrix) -> AbstractArray

Uses the information obtained from `svmtrain` such as the bias and weights to construct a decision function and predict the new values of the function.

# Arguments
- `svm::LSSVR`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `fits`: It can be any container data structure but it must have four elements: `x`, the data matrix; `y`, the labels vector; `α`, the weights; and `b`, the bias.
- `xnew::AbstractMatrix`: The data matrix that contains the new instances to be predicted.

# Returns
- `Array`: The labels corresponding to the prediction to each of the instances in `xnew`.
"""
function svmpredict(svm::LSSVR, fits, xnew::AbstractMatrix)
    x, α, b = fits
    # Compute the asymmetric kernel matrix in one go
    if svm.kernel == "rbf"
        kern_mat = KernelRBF(x, xnew, svm.σ)
    end
    result = sum(kern_mat .* α; dims=1) .+ b
    # We need to remove the trailing dimension
    result = reshape(result, size(result, 2))

    return result
end

@doc raw"""
    build_omega(x::AbstractMatrix, y::AbstractVector, sigma::Float64;
        kernel::String="rbf") -> AbstractMatrix

It builds a matrix, known as the "omega matrix", that contains the following information

``\Omega_{kl} = y_{k} y_{l} K(x_{k}, x_{l})``

with ``k,l=1,\dots,N``, and ``N`` being the length of `x`. In other words, the number of training instances.

This matrix contains information about the mapping to a new space using the kernel. It is exclusively used in the training step of the learning procedure.

# Arguments
- `x::AbstractMatrix`: The data matrix with the training instances.
- `y::AbstractVector`: The labels for each of the instances in `x`.

# Keywords
- `kernel::String="rbf"`: The kernel to be used. For now, only the RBF kernel is implemented.
- `sigma::Float64`: The hyperparameter for the RBF kernel.

# Returns
- `Ω`: The omega matrix computed as shown above.
"""
function build_omega(
    x::AbstractMatrix,
    y::AbstractVector;
    sigma::Float64=1.0,
    kernel::String="rbf",
)
    if kernel == "rbf"
        # Compute using own RBF type
        kern_mat = KernelRBF(x, sigma)
    end

    # Compute omega matrix
    Ω = (y .* y') .* kern_mat

    return Ω
end
