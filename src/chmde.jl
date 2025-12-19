module CHMDE
# Clustering Hybrid MDE from 10.1016/j.cor.2020.105165

export ClusteringHybredMDE

using SciMLBase
using OptimizationBase: OptimizationCache
using OptimizationBase
using Random
using LinearAlgebra
using QuasiMonteCarlo
using SpecialFunctions
using SparseArrays
using Statistics

include("common.jl")

struct ClusteringHybridMDE{S}
  F::Float64
  CR::Float64
  σ::Float64
  Np::Int
  local_solver::S
end

SciMLBase.has_init(opt::ClusteringHybridMDE) = true
SciMLBase.allowscallback(opt::ClusteringHybridMDE) = true
SciMLBase.allowsbounds(opt::ClusteringHybridMDE) = true
SciMLBase.requiresbounds(opt::ClusteringHybridMDE) = true

SciMLBase.requiresgradient(opt::ClusteringHybridMDE) = SciMLBase.requiresgradient(opt.local_solver)
SciMLBase.allowsfg(opt::ClusteringHybridMDE) = SciMLBase.allowsfg(opt.local_solver)

function ClusteringHybridMDE(local_solver;F=0.5,CR=0.,σ=5e-4,Np=2)
  @assert SciMLBase.allowsbounds(local_solver)
  ClusteringHybridMDE(F,CR,σ,Np,local_solver)
end


function SciMLBase.__init(prob::OptimizationProblem, opt::ClusteringHybridMDE;
                          maxiters::Number = 1000, callback = (args...) -> (false),
                          progress = false, 
                          local_prob = prob,
                          local_kwargs = (),
                          boundary_tol = 1e-8,
                          var_tol = 1e-16,
                          max_no_imp = 20,
                          max_no_pop_change = 10,
                          distance_tol = 1e-8,                         
                          initial_population = nothing, population_size=nothing,
                          kwargs...)



  if isnothing(population_size)
    if isnothing(initial_population) 
      population_size=3*length(prob.u0)
    else
      population_size=length(population)
    end
  end

  if isnothing(initial_population)
    initial_population = collect.(eachcol(QuasiMonteCarlo.sample(
      population_size-1,prob.lb,prob.ub,
      QuasiMonteCarlo.LatinHypercubeSample()
    )))
    push!(initial_population,prob.u0)
  end



  return OptimizationCache(prob, opt; maxiters, callback, progress,
                           population_size, initial_population,
                           local_prob,local_kwargs,
                           boundary_tol,var_tol,max_no_imp,max_no_pop_change,distance_tol,
                           kwargs...)
end

function gen_projection_matrix(nr,nc,s=3)
  I = Int[]
  J = Int[]
  V = Float64[]
  τ = 1/(2s)
  v = sqrt(s)
  for i in CartesianIndices((nr,nc))
    u = rand()
    if u < τ
      push!(I,i[1])
      push!(J,i[2])
      push!(V,v)
    elseif τ <= u < 2τ
      push!(I,i[1])
      push!(J,i[2])
      push!(V,-v)
    end
  end
  return SparseArrays.sparse!(I,J,V,nr,nc)
end

function population_distance(pop1,pop2)
  # Naive: there could be some reordering
  sum(norm(p1-p2) for (p1,p2) in zip(pop1,pop2))
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: ClusteringHybridMDE}
  uType = eltype(cache.reinit_cache.u0)
  CR = uType(cache.opt.CR)
  F = uType(cache.opt.F)
  σ = uType(cache.opt.σ)
  Np = cache.opt.Np

  # Why these types cannot be infered?
  maxiters::Int = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
  boundary_tol = cache.solver_args.boundary_tol
  var_tol = cache.solver_args.var_tol
  distance_tol = cache.solver_args.distance_tol
  max_no_imp = cache.solver_args.max_no_imp
  max_no_pop_change = cache.solver_args.max_no_pop_change

  population_size = cache.solver_args.population_size
  population = cache.solver_args.initial_population
  lb = cache.lb
  ub = cache.ub

  population_vals = Vector{uType}(undef,population_size)
  candidate_population = [similar(first(population)) for _ in 1:population_size]
  candidate_vals = Vector{uType}(undef,population_size)
  next_population = [similar(first(population)) for _ in 1:population_size]
  next_population_vals = fill(Inf,population_size)
  ls_vals = uType[]
  ls_points = similar(population,0)
  ϕ = Vector{Int8}(undef,population_size)

  best_val = typemax(uType)
  best_x = zero(first(population))

  Ndim = length(first(population))

  fevals = 0
  gevals = 0
  # Eval population
  for i in 1:population_size 
    population_vals[i] = cache.f(population[i],cache.p)
    fevals += 1
    if population_vals[i] < best_val
      best_val = population_vals[i]
      copyto!(best_x,population[i])
    end
  end

  local_prob = cache.opt.local_prob
  local_solver = cache.opt.local_solver
  local_kwargs = cache.solver_args.local_kwargs

  n_local = 0
  iterations = 0
  n_no_imp = 0
  n_no_pop_change = 0
  t0 = time()
  retcode = ReturnCode.MaxIters
  improved = false
  for iter in 1:maxiters

    # Generation
    proj_matrix = Ndim < 5 ? I : gen_projection_matrix(Np,Ndim)

    N = population_size + length(ls_vals)
    τ = pi^(-0.5) * (gamma(1+Ndim/2) * σ*log(N)/N)^(1/Ndim)

    for i in 1:population_size
      j = rand(1:population_size)
      popi = population[i]
      popj = population[j]
      cand = candidate_population[i]
      ϕ[i] = population_vals[i] > population_vals[j] ? 1 : -1
      for k in eachindex(cand)
        if rand() <= CR
          cand[k] = popi[k]
        else
          cand[k] = clamp(popi[k] + ϕ[i]*F*(popj[k]-popi[k]),lb[k]+boundary_tol,ub[k]-boundary_tol)
        end
      end

      candidate_vals[i] = cache.f(cand,cache.p)
      fevals += 1

      linkage = any(zip(population,population_vals)) do (p,v)
        norm(proj_matrix*p-proj_matrix*cand) < τ && v <= candidate_vals[i]
      end || any(zip(ls_points,ls_vals)) do (p,v)
        norm(proj_matrix*p-proj_matrix*cand) < τ && v <= candidate_vals[i]
      end 

      if !linkage # Local search
        push!(ls_points,copy(cand))
        push!(ls_vals,candidate_vals[i])
        n_local += 1

        local_sol = solve(remake(local_prob,u0=cand),
                          local_solver;local_kwargs...)

        fevals += local_sol.stats.fevals
        gevals += local_sol.stats.gevals

        copyto!(cand,local_sol.u)
        # The local_prob might not be an optimization problem or the loss
        # function might be different
        candidate_vals[i] = cache.f(cand,cache.p)

        push!(ls_points,copy(cand))
        push!(ls_vals,candidate_vals[i])
      end

      if candidate_vals[i] < best_val
        improved = true
        best_val = candidate_vals[i]
        copyto!(best_x,cand)
      end
    end

    # Selection
    for i in 1:population_size
      j = if ϕ[i] > 0
        i
      else
        argmin(abs(p-candidate_vals[i]) for p in population_vals)
      end
        if population_vals[j] > candidate_vals[i]
          copyto!(next_population[i],candidate_population[i])
          next_population_vals[i] = candidate_vals[i]
        else
          copyto!(next_population[i],population[i])
          next_population_vals[i] = population_vals[i]
        end
    end

    # Termination
    if !improved
      n_no_imp += 1
    else 
      n_no_imp = 0
    end

    if population_distance(population,next_population) < distance_tol
      n_no_pop_change += 1
    else
      n_no_pop_change = 0
    end

    # Synchronous update
    population,next_population = next_population,population
    population_vals,next_population_vals = next_population_vals,population_vals

    if n_no_imp > max_no_imp || 
      n_no_pop_change > max_no_pop_change ||
      var(next_population_vals) < var_tol
      retcode = ReturnCode.Success
      break
    end

    
    # Callback
    opt_state = OptimizationBase.OptimizationState(
      ; iter=iterations, u = best_x, objective = best_val, p = cache.p, 
      original = (;population,population_vals)
    )
    cb_call = cache.callback(opt_state, best_val)
    if !(cb_call isa Bool)
      error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
    elseif cb_call
      retcode = ReturnCode.Terminated
      stop = true
      break
    end
  end

  t1 = time()

  stats = OptimizationBase.OptimizationStats(; iterations ,
                                             time = t1-t0, fevals,gevals) 

  return SciMLBase.build_solution(cache, cache.opt, best_x, best_val;
                                  stats,retcode,original=(;population,population_vals))
end
end
