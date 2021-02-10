"""
    svmtrain(svm::LSSVC, x::AbstractMatrix, y::AbstractVector) -> Tuple

Solves a Least Squares Support Vector **classification** problem using the Conjugate
Gradient method. In particular, it uses the Lanczos process due to the fact that the
matrices are symmetric.

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
    # Specify the keyword arguments
    kwargs = _kwargs2dict(svm)
    # We build the kernel matrix and the omega matrix
    kern_mat = _build_kernel_matrix(x; kwargs...)
    y_external = extern_prod(y, y)
    # Mutiply the external product in-place to save memory
    pairwise_mul!(kern_mat, y_external)
    H = kern_mat + I / svm.γ

    # * Start solving the subproblems
    # First, solve for eta
    η, stats = cg_lanczos(H, y)
    @assert check_if_solved(stats) == true
    # Then, solve for nu
    ν, stats = cg_lanczos(H, ones(n))
    @assert check_if_solved(stats) == true

    # We then compute s
    s = BL.dot(n, y, 1, η, 1)

    # Finally, we solve the problem for alpha and b
    b = BL.dot(n, η, 1, ones(n), 1)
    b /= s
    rmul!(η, b)
    α = pairwise_diff(ν, η)

    return (x, y, α, b)
end

"""
    svmpredict(svm::LSSVC, fits, xnew::AbstractMatrix) -> AbstractArray

Uses the information obtained from `svmtrain` such as the bias and weights to construct a decision function and predict new class values. For the _classification_ problem only.

# Arguments
- `svm::LSSVC`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `fits`: It can be any container data structure but it must have four elements: `x`, the data matrix; `y`, the labels vector; `α`, the weights; and `b`, the bias.
- `xnew::AbstractMatrix`: The data matrix that contains the new instances to be predicted.

# Returns
- `Array`: The labels corresponding to the prediction to each of the instances in `xnew`.
"""
function svmpredict(svm::LSSVC, fits, xnew::AbstractMatrix)
    x, y, α, b = fits
    kwargs = _kwargs2dict(svm)
    @assert size(x, 1) == size(xnew, 1)
    # Build the kernel matrix with new observations
    kern_mat = _build_kernel_matrix(x, xnew; kwargs...)
    # Compute the decision function
    result = prod_reduction(kern_mat, y, α) .+ b
    result = dropdims(result; dims=2) # Squeeze singleton dimension

    return sign.(result)
end

function svmtrain_mc(svm::LSSVC, x, y, nclass)
    num_classifiers = nclass * (nclass - 1) / 2
    # Create a collection to store the learned parameters
    class_parameters = Vector{Tuple}(undef, Int(num_classifiers))
    # Create a collection to store the class pairs
    class_pairs = similar(class_parameters)
    c_idx = 1 # For keeping track of the classifiers

    for idx = 1:nclass - 1
        # Get the elements and indices for the first class
        a_class, a_idxs = _find_and_copy(idx, y)
        a_class .= 1.0 # The first class is encoded as 1.0
        for jdx = (idx + 1):nclass
            # Get the elements and indices for the second class
            b_class, b_idxs = _find_and_copy(jdx, y)
            b_class .= -1.0 # The second class is encoded as -1.0

            # Train a binary classification problem with the first and second classes
            all_indxs = vcat(a_idxs, b_idxs) # Join all indices
            all_classes = vcat(a_class, b_class) # Join both classes

            # Always copy the model and use views for the samples
            samples = view(x, :, all_indxs)
            # Solve the binary classification problem between classes idx and jdx
            new_svm = deepcopy(svm)
            fits = svmtrain(new_svm, samples, all_classes)

            # We save the parameters of that classifier
            class_parameters[c_idx] = fits
            # We save the class pairs for decoding them later
            class_pairs[c_idx] = (idx, jdx)
            c_idx += 1
        end
    end

    return (deepcopy(svm), class_parameters, class_pairs)
end

function svmpredict_mc(fits, Xnew::AbstractMatrix)
    # Extract the model, parameters and class codes
    svm, params, pairs = fits
    nclass = length(pairs) # The third element are the class pairs
    # This will hold all the predictions for the classifiers
    pooled_predictions = Matrix{eltype(Xnew)}(undef, nclass, size(Xnew, 2))

    for (idx, p, s) in zip(1:nclass, params, pairs)
        # For prediction, we just need the fitted parameters
        prediction = svmpredict(svm, p, Xnew)

        # Now that we have predictions, decode them using the class pairs
        broadcast!(x -> x == 1.0 ? s[1] : s[2], prediction, prediction)

        # Save the decoded predictions to our pooling collection
        pooled_predictions[idx, :] = prediction
    end

    # Apply the voting scheme
    results = _predictions_by_votes(pooled_predictions)

    return results
end

"""
    svmtrain(svm::LSSVR, x::AbstractMatrix, y::AbstractVector) -> Tuple

Solves a Least Squares Support Vector **regression** problem using the Conjugate Gradient
method. In particular, it uses the Lanczos process due to the fact that the matrices are
symmetric.

# Arguments
- `svm::LSSVR`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `x::AbstractMatrix`: The data matrix with the features. It is expected that this array is already standardized, i.e. the mean for each feature is zero and its standard deviation is one.
- `y::AbstractVector`: A vector that contains the continuous value of the function estimation. The elements can be any subtype of `Real`.

# Returns
- `Tuple`: A tuple containing `x` and the following two elements:
- `b`: Contains the bias for the decision function.
- `α`: Contains the weights for the decision function.
"""
function svmtrain(svm::LSSVR, x::AbstractMatrix, y::AbstractVector)
    n = size(y, 1)
    # Specify the keyword arguments
    kwargs = _kwargs2dict(svm)
    # We build the kernel matrix and the omega matrix
    kern_mat = _build_kernel_matrix(x; kwargs...)
    H = kern_mat + I / svm.γ

    # * Start solving the subproblems
    # First, solve for eta
    η, stats = cg_lanczos(H, ones(n))
    @assert check_if_solved(stats) == true
    # Then, solve for nu
    ν, stats = cg_lanczos(H, y)
    @assert check_if_solved(stats) == true

    # We then compute s
    s = BL.dot(n, ones(n), 1, η, 1)
    # Finally, we solve the problem for alpha and b
    b = BL.dot(n, η, 1, y, 1)
    b /= s
    rmul!(η, b)
    α = pairwise_diff(ν, η)

    return (x, α, b)
end

"""
    svmpredict(svm::LSSVR, fits, xnew::AbstractMatrix) -> AbstractArray

Uses the information obtained from `svmtrain` such as the bias and weights to construct a
decision function and predict the new values of the function. For the _regression_
problem only.

# Arguments
- `svm::LSSVR`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `fits`: It can be any container data structure but it must have four elements: `x`, the data matrix; `y`, the labels vector; `α`, the weights; and `b`, the bias.
- `xnew::AbstractMatrix`: The data matrix that contains the new instances to be predicted.

# Returns
- `Array`: The labels corresponding to the prediction to each of the instances in `xnew`.
"""
function svmpredict(svm::LSSVR, fits, xnew::AbstractMatrix)
    x, α, b = fits
    @assert size(x, 1) == size(xnew, 1)

    kwargs = _kwargs2dict(svm)
    kern_mat = _build_kernel_matrix(x, xnew; kwargs...)
    result = prod_reduction(kern_mat, α) .+ b

    # We need to remove the trailing dimension
    result = dropdims(result; dims=2)

    return result
end
