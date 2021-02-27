using CSV, Tables, Statistics, BenchmarkTools

"""
    read_data(csv_file::String, T::Type{<:AbstractFloat})::Matrix{T}
    
Read the voltage-current data in `csv_file` into a matrix, where each column is a (V, I) point.
`T` is the data type we want.
"""
function read_data(csv_file::String, T::Type{<:AbstractFloat})::Matrix{T}
    data = CSV.File(csv_file) |> Tables.matrix
    # we want each column as a (V, I) point
    return convert.(T, data')
end

"""
    measure_RMSE_stats(f::Function, n_points; n_trials=30)

Measure the runtime of function `f`. `n_points` is the number of data points (used in computing RMSE).
"""
function measure_RMSE_stats(f::Function, n_points; n_trials=30)
    rmse_list = zeros(n_trials)
    for i = 1:n_trials
        res = f()
        rmse_list[i] = sqrt(res.sse / n_points)
    end
    return Dict(
        :min => minimum(rmse_list),
        :mean => mean(rmse_list),
        :max => maximum(rmse_list),
        :std => std(rmse_list)
    )
end


function benchmark(f::Function; n_trials=30)
    @benchmark f() evals = 1 samples = n_trials seconds = 1000
end