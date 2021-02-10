"""
    This revert function reverses the internal implementation of the
`ScaleTransform` operation from `KernelFunctions.jl`. This is to preserve the
previous convention of this package.
"""
_revert(x) = sqrt(2.0 * x)

"""
    _build_kernel_matrix(x; kwargs...) -> AbstractMatrix
    _build_kernel_matrix(x, y; kwargs...) -> AbstractMatrix

It builds a kernel matrix using a user specified kernel. When just one matrix is passed,
`x`, the resulting matrix has the same size as `x`.
When both `x` and `y` are passed, the resulting matrix has the size
(`size(x, 2)`, `size(y, 2)`).

The matrix is built using the `kernelmatrix!` functions from `KernelFunctions.jl`.

# Arguments
- `x::AbstractMatrix`: The data matrix with the training instances.
- `y::AbstractMatrix`: Another data matrix, normally used for the prediction step in the classification implementation.

# Keywords
- `kernel::Symbol=:rbf`: The kernel to be used. The available options are `:rbf` for a RBF type kernel; `:linear`, for a classic linear kernel; and `:poly` for a polynomial kernel. The degree of the polynomial is handled with the `degree` keyword argument.
- `sigma::Float64`: The hyperparameter for the RBF kernel.
- `degree::Float64`: The degree of the polynomial kernel if "poly" is chosen.

# Returns
- `kern_mat`: The kernel matrix.
"""
function _build_kernel_matrix(x; kwargs...)
    # Extract the matrix for the keyword arguments
    kernel = kwargs[:kernel]

    if kernel == :rbf
        # Create the kernel with the corresponding scale
        t = ScaleTransform(_revert(kwargs[:sigma]))
        κ = transform(SqExponentialKernel(), t)
        kern_mat = kernelmatrix(κ, x)
    elseif kernel == :linear
        κ = LinearKernel()
        kern_mat = kernelmatrix(κ, x)
    elseif kernel == :poly
        # Create the kernel with the corresponding degree
        κ = PolynomialKernel(; degree=kwargs[:degree], c=0.0)
        kern_mat = kernelmatrix(κ, x)
    end

    return kern_mat
end

function _build_kernel_matrix(x, y; kwargs...)
    # Check that the first dimension is the same
    @assert size(x, 1) == size(y, 1)

    # Extract the matrix for the keyword arguments
    kernel = kwargs[:kernel]

    if kernel == :rbf
        # Create the kernel with the corresponding scale
        t = ScaleTransform(_revert(kwargs[:sigma]))
        κ = transform(SqExponentialKernel(), t)
        kern_mat = kernelmatrix(κ, x, y)
    elseif kernel == :linear
        κ = LinearKernel()
        kern_mat = kernelmatrix(κ, x, y)
    elseif kernel == :poly
        # Create the kernel with the corresponding degree
        κ = PolynomialKernel(; degree=kwargs[:degree], c=0.0)
        kern_mat = kernelmatrix(κ, x, y)
    end

    return kern_mat
end

function _choose_kernel(; kwargs...)
    # Extract the matrix for the keyword arguments
    kernel = kwargs[:kernel]

    if kernel == :rbf
        # Create the kernel with the corresponding scale
        t = ScaleTransform(_revert(kwargs[:sigma]))
        κ = transform(SqExponentialKernel(), t)
    elseif kernel == :linear
        κ = LinearKernel()
    elseif kernel == :poly
        # Create the kernel with the corresponding degree
        κ = PolynomialKernel(; degree=kwargs[:degree], c=0.0)
    end

    return κ
end

"""
    Takes a Support Vector Machine type and converts some of its attributes to a
dictionary that makes it easier to handle as keyword arguments.
"""
_kwargs2dict(svm::Union{LSSVC,LSSVR}) =
    Dict(:kernel => svm.kernel, :sigma => svm.σ, :degree => svm.degree)

_kwargs2dict(svm::FixedSizeSVR) = Dict(
    :kernel => svm.kernel,
    :sigma => svm.σ,
    :degree => svm.degree,
    :subsample => svm.subsample,
    :iters => svm.iters
)

"""
    Function to find all the instances in an array `y` that are
equal to some value `k`. It returns a copy of the array and the indices where the
condition is met.
"""
function _find_and_copy(k, y)
    indices = findall(isequal(k), y)

    return (copy(y[indices]), indices)
end

"""
    This function takes in a matrix `x` and does the following logic:
- Obtains the unique elements from the matrix.
- Counts how many occurrences of each element happen in the array.
- Using `argmax`, the indices where this condition is met are extracted.
- Finally, we only need the first dimension index, so we extract it as such.

Essentially, this is a voting scheme for the multiclass classification problem.
In the case of a tie, the smallest index is always chosen, i.e. 1. This is not the best
strategy, but it is after the following paper:

Chih-Wei Hsu and Chih-Jen Lin (2002) ‘A comparison of methods for multiclass support
vector machines’, IEEE Transactions on Neural Networks, 13(2), pp. 416.
doi: 10.1109/72.991427.

Where it says the following quote:

> [...] Then we predict is in the class with the largest vote. The
> voting approach described above is also called the “Max Wins”
> strategy. In case that two classes have identical votes, though it
> may not be a good strategy, now we simply select the one with
> the smaller index. [...]
"""
function _predictions_by_votes(x)
    unique_elements = unique(x) |> sort
    predictions = zeros(size(x, 2))
    counts = map(z -> count(==(z), x, dims=1), unique_elements)
    largest_values = argmax(vcat(counts...), dims=1)

    for (i, l) in enumerate(largest_values)
        predictions[i] = l[1]
    end

    return predictions
end

"""
    To test whether a solver from Krylov.jl solved a given problem successfully.
"""
check_if_solved(stat) = stat.solved || @warn "A solution was not found!"
