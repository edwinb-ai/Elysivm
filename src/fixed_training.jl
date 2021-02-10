function _sampleindex(X::AbstractMatrix, r::Real)
    0 < r <= 1 || throw(ArgumentError("Sample rate `r` must be in range (0,1]"))
    n = size(X, 2)
    m = ceil(Int, n * r)
    S = StatsBase.sample(1:n, m; replace=false, ordered=true)

    return S
end

function _sample_matrix(k::Kernel, X::AbstractMatrix, S::Vector{<:Integer})
    X_obs = view(X, :, S)
    C = kernelmatrix(k, X_obs, X)
    Cs = C[:, S]

    return (C, Cs)
end

function _renyi_entropy(X::AbstractMatrix, N::Integer, M::Integer)
    ones_m = ones(M)
    integral = BL.gemv('N', 1.0, X, ones_m)
    entropy = BL.dot(M, ones_m, 1, integral, 1)
    entropy /= N^2

    return -log(entropy)
end

function _nystroem_renyi(k::Kernel, X::AbstractMatrix, n, m; iters=50_000)
    @assert m < n

    best = -Inf
    old = 0
    r = m / n
    Cs = Matrix{eltype(X)}(undef, m, m)
    idxs = Vector{Int}(undef, m)

    for _ = 1:iters
        idxs = _sampleindex(X, r)
        C, Cs = _sample_matrix(k, X, idxs)
        old = _renyi_entropy(Cs, n, m)
        if old > best
            best = old
        end
    end

    return (best, Cs, idxs)
end

function svmtrain(svm::FixedSizeSVR, x::AbstractMatrix, y::AbstractVector)
    # Declare the size to work with
    n = size(y, 1)
    m = svm.subsample

    # Use this information to create the Nyström approximation
    kwargs = _kwargs2dict(svm)
    k = _choose_kernel(; kwargs...)
    best, Cs, idxs = _nystroem_renyi(k, x, n, m; iters=svm.iters)

    # We do a spectral decomposition
    fact = eigen(Cs)

    # We now augment the kernel matrix with ones to include the
    # bias term
    kern_mat_aug = hcat(fact.vectors, ones(m))

    # Create the square matrix A^T A
    sq_mat = kern_mat_aug' * kern_mat_aug

    # We need the correct size for the vector
    b = kern_mat_aug' * y[idxs]

    # We now solve the ridge regression problem
    result, stats = cgls(sq_mat, b; λ=1 / svm.γ)
    @assert check_if_solved(stats) == true

    # Extract the weights and the bias found
    wi_s = result[1:end - 1]
    bias = result[end]

    # Compute the weights for the decision function
    alphas = m .* fact.vectors
    @. alphas /= sqrt(fact.values)
    result = prod_reduction(alphas, wi_s)

    return (x, result, bias, idxs)
end

function svmpredict(svm::FixedSizeSVR, fits, xnew::AbstractMatrix)
    x, alphas, bias, idxs = fits
    kwargs = _kwargs2dict(svm)
    k = _choose_kernel(; kwargs...)
    kern_mat = kernelmatrix(k, xnew, view(x, :, idxs)) |> transpose
    alphas = dropdims(alphas; dims=1)
    result = prod_reduction(kern_mat, alphas) .+ bias

    # We need to remove the trailing dimension
    result = dropdims(result; dims=1)

    return result
end
