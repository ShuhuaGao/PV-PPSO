module PSO

using Printf, CUDA
using PV

export initialize, evolve!, pso

"""
    initialize(x_lb::AbstractVector{T}, x_ub::AbstractVector{T}, 
        d::Integer, np::Integer) where T <: AbstractFloat

Initialize a population of `np` particles with random positions `X` and velocities `V`. 
`d` is the dimension (i.e., number of variables). `x_lb` and `x_ub` are lower bound and upper bound 
of variables.

Two matrices `X` and `V` of size `d × np` are returned, each column of which denotes a position 
or velocity.
"""
function initialize(x_lb::AbstractVector{T}, x_ub::AbstractVector{T}, 
        d::Integer, np::Integer) where T <: AbstractFloat
    @assert d >= 1 && np >= 1 && length(x_lb) == d && length(x_ub) == d
    dx = x_ub .- x_lb
    X = rand(T, d, np) .* dx .+ x_lb
    V = rand(T, d, np) .* 2 .* dx .- dx
    return X, V
end

@inline function _clamp!(x::AbstractVector, l::AbstractVector, u::AbstractVector)
    @inbounds for i in eachindex(x)
        x[i] = clamp(x[i], l[i], u[i])
    end
    nothing
end

@inline function bounce_back!(x, v, l, u)
    T = eltype(x)
    @inbounds for i in eachindex(x)
        t = x[i] + v[i]
        if t < l[i]
            t = l[i] + rand(T) * (x[i] - l[i])
        elseif t > u[i]
            t = x[i] + rand(T) * (u[i] - x[i])
        end
        x[i] = t
    end
end


"""
    evolve!(f::Function, X::Matrix{T}, V::Matrix{T}, x_lb::AbstractVector{T}, 
        x_ub::AbstractVector{T}, niter::Integer=100; w, c1, c2, log=false) where T <: AbstractFloat

Evolve particles with initial positions `X` and velocities `V` for at most `niter` iterations 
to minimize the function `f`. `x_lb` and `x_ub` are lower bound and upper bound of variables.

`w, c1, c2` are parameters of PSO. If `log` is `true`, then progress information is printed in each 
iteration.

The best particle's position, its fitness, and the best fitness history for `niter` iterations are 
returned. 

This method is designed to work on CPU to validate the correctness of PSO.
"""
function evolve!(f::Function, X::Matrix{T}, V::Matrix{T}, x_lb::AbstractVector{T}, 
        x_ub::AbstractVector{T}, niter::Integer=100;  
        w, c1, c2, log=false) where T <: AbstractFloat
    @assert size(X) == size(V)
    @assert size(X, 1) == length(x_lb) == length(x_ub)
    @assert w_min <= w_max
    np = size(X, 2)
    fitness = zeros(T, np)
    pbest = copy(X)
    pbest_fitness = fill(T(Inf), np)
    gbest = X[:, 1]
    gbest_fitness = Inf
    gbest_fitness_hist = zeros(T, niter)
    
    if log
        @printf("%-10s%-10s\n", "iter", "sse")
    end

    ws = range(w_max, w_min; length=niter_w_decay)  # inertia weight decay

    w = w_max
    for iter = 1:niter
        # w = iter <= niter_w_decay ? ws[iter] : w_min
        # w *= 0.998
        w = max(w, w_min)
        # evaluate fitness and update pbest
        @inbounds for i = 1:np
            fitness[i] = f(@view X[:, i])
            if fitness[i] <= pbest_fitness[i]
                pbest_fitness[i] = fitness[i]
                pbest[:, i] .= @view X[:, i]
            end
        end
        # update gbest
        pbest_idx = argmin(pbest_fitness)
        @inbounds if pbest_fitness[pbest_idx] <= gbest_fitness
            gbest_fitness = pbest_fitness[pbest_idx] 
            gbest .= @view pbest[:, pbest_idx]
        end
        gbest_fitness_hist[iter] = gbest_fitness
        if log
            @printf("%-10d%-10.5e%10f\n", iter, gbest_fitness,w)
        end
        # terminate?
        if iter == niter
            break
        end
        # update velocity/position: parallel
        @inbounds @views for i = 1:np
            V[:, i] .= w .* V[:, i] .+ c1 .* rand() .* (pbest[:, i] .- X[:, i]) .+ 
                        c2 .* rand() .* (gbest .- X[:, i])
            # X[:, i] .+= V[:, i]
            # _clamp!(X[:, i], x_lb, x_ub)
            bounce_back!(X[:, i], V[:, i], x_lb, x_ub)
        end 
    end
    return gbest, gbest_fitness, gbest_fitness_hist
end


@inline function _copy!(dest::AbstractVector, src::AbstractVector)
    @inbounds for i = 1:length(src)
        dest[i] = src[i]
    end
end


"""
    evolve!(evaluator::Function, data::CuMatrix{T}, temperature::T,
        X::CuMatrix{T}, V::CuMatrix{T}, x_lb::CuVecOrMat{T}, 
        x_ub::CuVecOrMat{T}, niter::Integer=100; 
        w, c1, c2, log=false, threads::Integer=160,
        min_delta=0.0f0, patience=niter) where T <: AbstractFloat

This method is designed to work on GPU.
Refer to the CPU-based [`evolve!`](@ref) for details. Note the two parameters for early termination:
- `min_delta`: a threshold to decide whether the best fitness (i.e., fitness of *gbest*) has reduced
- `patience`: if the best fitness has not reduced for continuous `patience` iterations, terminate. 

A `NamedTuple` is returned with three fields
- `sse`: SSE of the best solution
- `solution`: best solution (*gbest*)
- `stopped_iter`: the iteration at which PSO is terminated
"""
function evolve!(evaluator::Function, data::CuMatrix{T}, temperature::T,
        X::CuMatrix{T}, V::CuMatrix{T}, x_lb::CuVecOrMat{T}, 
        x_ub::CuVecOrMat{T}, niter::Integer=100; 
        w, c1, c2, log=false, threads::Integer=160,
        min_delta=0.0f0, patience=niter) where T <: AbstractFloat
    np = size(X, 1)  
    nv = size(X, 2)  
    fitness = CUDA.zeros(T, np)
    pbest = copy(X)
    pbest_fitness = CUDA.fill(T(Inf), np)
    gbest = X[1, :]  # copied
    gbest_fitness = T(Inf)
    gbest_fitness_hist = CUDA.zeros(T, niter)
    R = CUDA.rand(T, 4 * np, niter)   # pre-generate random numbers 

    function evaluate!(iter::Integer)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if i > np
            return nothing
        end
        @inbounds fitness[i] = fit = evaluator(data, temperature, @view X[i, :])
        @inbounds if fit <= pbest_fitness[i]
            pbest_fitness[i] = fit
            @views _copy!(pbest[i, :], X[i, :])
        end
        nothing
    end

    function update_vel_pos!(iter)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if i > np
            return nothing
        end
        # update velocity and position
        @inbounds r1 = R[i, iter] 
        @inbounds r2 = R[i + np, iter] 
        @inbounds for j = 1:nv  # update each component individually
            V[i, j] = new_v = w * V[i, j] + c1 * r1 * (pbest[i, j] - X[i, j]) + c2 * r2 * (gbest[j] - X[i, j])
            old_x = X[i, j]
            t = old_x  + new_v
            l = x_lb[j]
            u = x_ub[j]
            if t < l
                t = l + R[i + 2 * np, iter] * (old_x - l)
            elseif t > u
                t = old_x + R[i + 3 * np, iter] * (u - old_x)
            end
            X[i, j] = t
        end
        nothing
    end

    # kernel config
    blocks = np ÷ threads + (np % threads > 0)
    if log
        @show threads, blocks
        @printf("%-10s%-10s\n", "iter", "sse")
    end
    
    wait = 0
    stopped_iter = 0

    for iter = 1:niter
        # fitness
        @cuda threads = threads blocks = blocks evaluate!(iter)
        # update gbest： parallel reduction
        pbest_idx = argmin(pbest_fitness) 
        min_pbest_fitness = pbest_fitness[pbest_idx]
        if min_pbest_fitness <= gbest_fitness
            if gbest_fitness - min_pbest_fitness >= min_delta
                wait = 0
            else
                wait += 1
            end
            gbest_fitness = min_pbest_fitness
            @inbounds _copy!(gbest, @view pbest[pbest_idx, :])
        end
        if log
            @printf("%-10d%-10.5e\n", iter, gbest_fitness)
        end
        # terminate?
        if wait >= patience
            stopped_iter = iter
            if log
                @printf("Early stopped at iteration %d\n", iter) 
            end
            break
        end
        if iter == niter
            stopped_iter = niter
            break
        end
        
        # update velocity/position: parallel
        @cuda threads = threads blocks = blocks update_vel_pos!(iter)
    end
    return (sse = gbest_fitness, solution = Array(gbest), stopped_iter = stopped_iter)
end


"""
    pso(data::AbstractMatrix{T}, temperature::T, model::Symbol, x_lb::AbstractVector{T}, 
        x_ub::AbstractVector{T}; niter::Integer=400, np::Integer=1600, 
        w::T, c1::T, c2::T, log=false, threads::Integer=160,
        min_delta::T=0.0, patience=niter + 1) where T <: AbstractFloat

Parallel PSO for PV parameter identification. Each column of `data` is a V-I point. `model` can be either 
`:SDM` or `:DDM`. `x_lb` and `x_ub` denote the lower bound and upper bound of each variable to 
confine the search range. 
- `niter`: number of iterations
- `np`: number of particles
- `w, c1, c2`: PSO velocity update parameters
- `log`: print progress information if `true`
- `threads`: number of threads per block in CUDA kernel
- `min_delta, patience`: early stopping. Similar to *keras* API: 
    https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
"""
function pso(data::AbstractMatrix{T}, temperature::T, model::Symbol, x_lb::AbstractVector{T}, 
        x_ub::AbstractVector{T}; niter::Integer=400, np::Integer=1600, 
        w::T, c1::T, c2::T, log=false, threads::Integer=160,
        min_delta::T=0.0, patience=niter + 1) where T <: AbstractFloat
    @assert model in (:SDM, :DDM)
    nv = model === :SDM ? 5 : 7
    @assert length(x_lb) == length(x_ub) == nv
    @assert size(data, 1) == 2
    @assert threads % 32 == 0

    # initialize: each row is a particle (for GPU)
    dx = reshape(x_ub .- x_lb, 1, :) |> CuArray
    x_lb = reshape(x_lb, 1, :) |> CuArray
    x_ub = reshape(x_ub, 1, :) |> CuArray
    X = CUDA.rand(T, np, nv) .* dx .+ x_lb
    V = CUDA.rand(T, np, nv) .* 2 .* dx .- dx

    # evolve particles
    if model === :SDM
        return evolve!(PV.sse_sdm, data |> CuArray, temperature, 
            X, V, x_lb |> CuArray, x_ub |> CuArray, niter;
            w, c1, c2, log, threads, min_delta, patience)
    else
        return evolve!(PV.sse_ddm, data |> CuArray, temperature, 
            X, V, x_lb |> CuArray, x_ub |> CuArray, niter;
            w, c1, c2, log, threads, min_delta, patience)
    end
end

end # module
