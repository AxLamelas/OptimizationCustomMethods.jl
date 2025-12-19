module SATR
# Bound constraints from 10.1137/0806023
# Trust region radius update from 10.1137/030602563

# Possible to use but disable by default
# Non-monotonicity from 10.1007/s00186-025-00904-4
# Noise tolerance from 10.48550/arXiv.2201.00973

export SelfAdaptiveTR

using SciMLBase
using OptimizationBase: OptimizationCache
using OptimizationBase
using Random
using LinearAlgebra

include("common.jl")

struct SelfAdaptiveTR 
  c1::Float64
  c2::Float64
  α1::Float64
  α2::Float64
  α3::Float64
  θmin::Float64
  r::Float64
  # Non-monotonicity
  ξ0::Float64
  # Noise tolerance
  ϵ::Float64
end

SciMLBase.has_init(opt::SelfAdaptiveTR) = true
SciMLBase.allowscallback(opt::SelfAdaptiveTR) = true
SciMLBase.requiresgradient(opt::SelfAdaptiveTR) = true
SciMLBase.allowsbounds(opt::SelfAdaptiveTR) = true
SciMLBase.allowsfg(opt::SelfAdaptiveTR) = true


function SelfAdaptiveTR(; 
                                c1 = 1e-4, c2 = 0.95,
                                α1 = 0.25, α2 = 3.5, α3 = 1.05, r = 0.9,
                                θmin = 0.95, ξ0 = 0.,ϵ=0.)
    SelfAdaptiveTR(c1,c2,α1,α2,α3,θmin,r,ξ0,ϵ)
end


function SciMLBase.__init(prob::OptimizationProblem, opt::SelfAdaptiveTR;
                          maxiters::Number = 1000, callback = (args...) -> (false),
                          progress = false, save_best = true,
                          initial_radius=nothing,
                          initial_hessian = Matrix{eltype(prob.u0)}(I,length(prob.u0),length(prob.u0)),
                          kwargs...)
    return OptimizationCache(prob, opt; maxiters, callback, progress,
                             save_best, initial_radius, initial_hessian ,kwargs...)
end

function update_hessian!(H,y,prev_val,prev_grad,p,val,grad)
    # Modified BFGS from 10.1007/s10589-008-9219-0
    ρ = 2*(prev_val-val) + dot(grad,p) + dot(prev_grad,p)
    n = max(ρ,0)/dot(p,p) 
    for i in eachindex(y)
        y[i] = grad[i] - prev_grad[i] + n * p[i]
    end
    c = dot(p,y)

    if isnan(c) || c <= 1e-8*norm(p)*norm(y) return end

    Bp = H * p
    mul!(H,y,y',1/c,1)
    mul!(H,Bp,Bp',-1/dot(p,Bp),1)
    
    return
end

###########################################
# Solution of TR subproblem from Optim.jl #
###########################################


# Check whether we are in the "hard case".
#
# Args:
#  H_eigv: The eigenvalues of H, low to high
#  qg: The inner product of the eigenvalues and the gradient in the same order
#
# Returns:
#  hard_case: Whether it is a candidate for the hard case
#  lambda_index: The index of the first lambda not equal to the smallest
#                eigenvalue, which is only correct if hard_case is true.
function check_hard_case_candidate(H_eigv, qg)
    @assert length(H_eigv) == length(qg)
    if H_eigv[1] >= 0
        # The hard case is only when the smallest eigenvalue is negative.
        return false, 1
    end
    hard_case = true
    lambda_index = 1
    hard_case_check_done = false
    while !hard_case_check_done
        if lambda_index > length(H_eigv)
            hard_case_check_done = true
        elseif abs(H_eigv[1] - H_eigv[lambda_index]) > 1e-10
            # The eigenvalues are reported in order.
            hard_case_check_done = true
        else
            if abs(qg[lambda_index]) > 1e-10
                hard_case_check_done = true
                hard_case = false
            end
            lambda_index += 1
        end
    end

    hard_case, lambda_index
end

# Equation 4.38 in N&W (2006)
function calc_p!(lambda::T, min_i, n, qg, H_eig, p) where {T}
    fill!(p, zero(T))
    for i = min_i:n
        p[:] -= qg[i] / (H_eig.values[i] + lambda) * H_eig.vectors[:, i]
    end
    return nothing
end

#==
Returns a tuple of initial safeguarding values for λ. Newton's method might not
work well without these safeguards when the Hessian is not positive definite.
==#
function initial_safeguards(H, gr, delta, lambda)
    # equations are on p. 560 of [MORESORENSEN]
    T = eltype(gr)
    λS = -Base.minimum(@view(H[diagind(H)])) # Base.minimum !== minimum
    # they state on the first page that ||⋅|| is the Euclidean norm
    gr_norm = norm(gr)
    Hnorm = opnorm(H, 1)
    λL = max(T(0), λS, gr_norm / delta - Hnorm)
    λU = gr_norm / delta + Hnorm
    # p. 558
    lambda = min(max(lambda, λL), λU)
    if lambda ≤ λS
        lambda = max(T(1) / 1000 * λU, sqrt(λL * λU))
    end
    lambda
end

# Choose a point in the trust region for the next step using
# the interative (nearly exact) method of section 4.3 of N&W (2006).
# This is appropriate for Hessians that you factorize quickly.
#
# Args:
#  gr: The gradient
#  H:  The Hessian
#  delta:  The trust region size, ||s|| <= delta
#  s: Memory allocated for the step size, updated in place
#  tolerance: The convergence tolerance for root finding
#  max_iters: The maximum number of root finding iterations
#
# Returns:
#  m - The numeric value of the quadratic minimization.
#  interior - A boolean indicating whether the solution was interior
#  lambda - The chosen regularizing quantity
#  hard_case - Whether or not it was a "hard case" as described by N&W (2006)
#  reached_solution - Whether or not a solution was reached (as opposed to
#      terminating early due to max_iters)
function solve_tr_subproblem!(gr,H, delta, s; tolerance = 1e-10, max_iters = 100)
    T = eltype(gr)
    n = length(gr)
    delta_sq = delta^2

    @assert n == length(s)
    @assert (n, n) == size(H)
    @assert max_iters >= 1

    # Note that currently the eigenvalues are only sorted if H is perfectly
    # symmetric.  (Julia issue #17093)
    Hsym = Symmetric(H)
    if any(!isfinite, Hsym)
        return T(-Inf) 
    end
    H_eig = eigen(Hsym)

    if !isempty(H_eig.values)
        min_H_ev, max_H_ev = H_eig.values[1], H_eig.values[n]
    else
        return T(-Inf)
    end
    H_ridged = copy(H)

    # Cache the inner products between the eigenvectors and the gradient.
    qg = H_eig.vectors' * gr

    # These values describe the outcome of the subproblem.  They will be
    # set below and returned at the end.
    interior = true
    hard_case = false
    reached_solution = true

    # Unconstrained solution
    if min_H_ev >= 1e-8
        calc_p!(zero(T), 1, n, qg, H_eig, s)
    end

    if min_H_ev >= 1e-8 && sum(abs2, s) <= delta_sq
        # No shrinkage is necessary: -(H \ gr) is the minimizer
        interior = true
        reached_solution = true
        lambda = zero(T)
    else
        interior = false

        # The hard case is when the gradient is orthogonal to all
        # eigenvectors associated with the lowest eigenvalue.
        hard_case_candidate, min_i = check_hard_case_candidate(H_eig.values, qg)

        # Solutions smaller than this lower bound on lambda are not allowed:
        # they don't ridge H enough to make H_ridge PSD.
        lambda_lb = nextfloat(-min_H_ev)
        lambda = lambda_lb

        hard_case = false
        if hard_case_candidate
            # The "hard case". lambda is taken to be -min_H_ev and we only need
            # to find a multiple of an orthogonal eigenvector that lands the
            # iterate on the boundary.

            # Formula 4.45 in N&W (2006)
            calc_p!(lambda, min_i, n, qg, H_eig, s)
            p_lambda2 = sum(abs2, s)
            if p_lambda2 > delta_sq
                # Then we can simply solve using root finding.
            else
                hard_case = true
                reached_solution = true

                tau = sqrt(delta_sq - p_lambda2)

                # I don't think it matters which eigenvector we pick so take
                # the first.
                calc_p!(lambda, min_i, n, qg, H_eig, s)
                s[:] = -s + tau * H_eig.vectors[:, 1]
            end
        end

        lambda = initial_safeguards(H, gr, delta, lambda)

        if !hard_case
            # Algorithim 4.3 of N&W (2006), with s insted of p_l for consistency
            # with Optim.jl

            reached_solution = false
            for iter = 1:max_iters
                lambda_previous = lambda

                for i = 1:n
                    H_ridged[i, i] = H[i, i] + lambda
                end

                F = cholesky(Hermitian(H_ridged), check = false)
                # Sometimes, lambda is not sufficiently large for the Cholesky factorization
                # to succeed. In that case, we set double lambda and continue to next iteration
                if !issuccess(F)
                    lambda *= 2
                    continue
                end

                R = F.U
                s[:] = -R \ (R' \ gr)
                q_l = R' \ s
                norm2_s = dot(s, s)
                denom = (delta * dot(q_l, q_l))
                lambda_update = if !iszero(denom)
                    norm2_s * (sqrt(norm2_s) - delta) / denom
                else
                    zero(delta)
                end
                lambda += lambda_update

                # Check that lambda is not less than lambda_lb, and if so, go
                # half the way to lambda_lb.
                if lambda < lambda_lb
                    lambda = 0.5 * (lambda_previous - lambda_lb) + lambda_lb
                end

                if abs(lambda - lambda_previous) < tolerance
                    reached_solution = true
                    break
                end
            end
        end
    end

    m = -(dot(gr, s) + 0.5 * dot(s, H , s))

    return m
end

###########################################


# function solve_tr_subproblem!(gr,H,Δ,p::AbstractVector{T}) where T
#   # Steihaug-Toint method https://doi.org/10.1137/0720042
#   fill!(p,zero(T))
#
#   ngr = norm(gr)
#   tol = min(1e-6,sqrt(ngr)) * ngr
#
#   r = -copy(gr)
#   d = copy(r)
#   Bd = similar(d)
#
#   nr = dot(r,r)
#   new_nr = zero(nr)
#
#   while true
#     mul!(Bd,H,d)
#     γ = dot(d,Bd)
#     if γ <= 0
#       nd2 = dot(d,d)
#       pd = dot(p,d)
#       σ = (-pd  + sqrt(max(zero(T),pd^2+nd2*(Δ^2-nd2*dot(p,p))))) / nd2 
#       p .+= σ * d
#       break
#     end
#     α = nr/γ
#     if norm(p+α*d) >= Δ
#       nd2 = dot(d,d)
#       pd = dot(p,d)
#       σ = (-pd  + sqrt(max(zero(T),pd^2+nd2*(Δ^2-nd2*dot(p,p))))) / nd2 
#       p .+= σ * d
#       break
#     end
#     p .+= α*d
#     r .-= α*Bd
#
#     if norm(r) / ngr < tol
#       break
#     end
#
#     new_nr = dot(r,r)
#     β = new_nr/nr
#     nr = new_nr
#     d .= r + β*d
#   end
#   return -(dot(gr,p) + 0.5 * dot(p,H,p))
# end
function step_back!(x,s,lb,ub,θ)
    τ,ind = findmin(
        iszero(s[i]) ? typemax(s[i]) : max((prevfloat(ub[i])-x[i])/s[i],(nextfloat(lb[i])-x[i])/s[i]) for i in eachindex(x))
    α = min(one(τ),θ*τ)
    for i in eachindex(s)
        s[i] *= α
    end
    return α < one(τ) ? ind : nothing
end

function update_scaling!(sqrt_abs_scale,dscale,x,gr,lb,ub)
    T = eltype(x)
    for i in eachindex(x)
        sqrt_abs_scale[i], dscale[i] = if gr[i] < 0 
            if isfinite(ub[i])
                sqrt(prevfloat(ub[i])-x[i]), -one(T)
            else
                one(T), zero(T)
            end
        else
            if isfinite(lb[i])
                sqrt(x[i] - nextfloat(lb[i])), one(T)
            else
                one(T), zero(T)
            end
        end
    end
    nothing
end

function solve_1d_tr_subproblem!(g,H,delta,s,shift)
    γ = - (dot(g,s) + 2*dot(s,H,shift)) / dot(s,H,s)
    α = min(γ,(delta-norm(shift)) / norm(s))
    for i in eachindex(s)
        s[i] *= α
    end
    return
end

function solve_1d_tr_subproblem!(g,H,delta,s)
    γ = - dot(g,s) / dot(s,H,s)
    α = min(γ,delta / norm(s))
    for i in eachindex(s)
        s[i] *= α
    end
    return
end

function solve_tr_subproblem!(gr::T,H::AbstractMatrix{W}, delta::W,
                              s::T,x::T,lb::T,ub::T,sqrt_abs_scale,dscale,θ) where {W,T<:AbstractVector{W}}

    Dinv = Diagonal(sqrt_abs_scale)

    # Scale gradient and Hessian
    gs = Dinv*gr
    Cs = Diagonal(gr .* dscale)
    Hs = Dinv*H*Dinv + Cs

    C = Diagonal(gr .* dscale ./ (sqrt_abs_scale .^ 2))
    
    solve_tr_subproblem!(gs,Hs,delta,s)
    # Undo the scaling
    for i in eachindex(s)
        s[i] *= sqrt_abs_scale[i]
    end

    # Candidate points
    ## Truncated 
    outside_ind = step_back!(x,s,lb,ub,θ)

    # Gradient
    grad_s = -gs
    solve_1d_tr_subproblem!(gs,Hs,delta,grad_s)
    for i in eachindex(grad_s)
        grad_s[i] *= sqrt_abs_scale[i]
    end
    step_back!(x,grad_s,lb,ub,θ)

    ## Reflection
    # cands = if !isnothing(outside_ind)
    #     reflected_s = s ./ sqrt_abs_scale
    #     reflected_s[outside_ind] *= -one(eltype(x))
    #
    #     solve_1d_tr_subproblem!(gs,Hs,delta,s ./ sqrt_abs_scale)
    #     for i in eachindex(reflected_s)
    #         reflected_s[i] *= sqrt_abs_scale[i]
    #     end
    #     (s,grad_s,reflected_s)
    # else
    #     (s,grad_s)
    # end
    cands = (s,grad_s)

    m = map(cands) do v
        b = 0.5 * dot(v,C,v)
        -(dot(gr,v) + 0.5 * dot(v,H,v) + b), b
    end

    i = argmax(m)
    copyto!(s,cands[i])

    return m[i]
end

function _check_stopping_tol(T::Type,solver_args)
  if isnothing(solver_args.abstol) && isnothing(solver_args.reltol)
    return T(1e-8), zero(T)
  end
  if isnothing(solver_args.abstol)
    return zero(T), T(solver_args.reltol)
  end
  if isnothing(solver_args.reltol)
    return T(solver_args.abstol),zero(T)
  end
  return T(solver_args.abstol), T(solver_args.reltol)
end

box = Ref{Any}()
function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SelfAdaptiveTR}
    uType = eltype(cache.reinit_cache.u0)
    c1 = uType(cache.opt.c1)
    c2 = uType(cache.opt.c2)
    α1 = uType(cache.opt.α1)
    α2 = uType(cache.opt.α2)
    α3 = uType(cache.opt.α3)
    r = uType(cache.opt.r)
    θmin = uType(cache.opt.θmin)
    ξ0 = uType(cache.opt.ξ0)
    ϵ = uType(cache.opt.ϵ)

    δ = 2/(1-c2) * ϵ

    # Why these types cannot be infered?
    maxiters::Int = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    abstol::uType,reltol::uType = _check_stopping_tol(uType,cache.solver_args)

    x = copy(cache.reinit_cache.u0)
    lb = if isnothing(cache.lb)
        fill(typemin(x),length(x))
    else
        cache.lb
    end
    ub = if isnothing(cache.ub)
        fill(typemax(x),length(x))
    else
        cache.ub
    end

    if !all(lb .< x .< ub)
        center = (ub+lb)/2
        s = x-center
        step_back!(center,s,lb,ub,θmin)
        x = center + s
        @warn "The initial guess is outside the given bounds. Clamping it..."  center s
    end

    gr = zero(x)
    cand = zero(x)
    cand_gr = zero(x)
    p = zero(x)
    y = zero(x)
    sqrt_abs_scale = zero(x)
    dscale = zero(x)
    best_val = typemax(uType)
    best_x = copy(x)

    val = eval_val_and_grad!(cache,gr,x)
    H = copy(cache.solver_args.initial_hessian)
    update_scaling!(sqrt_abs_scale,dscale,x,gr,lb,ub)

    if size(H) != (length(x),length(x))
        throw(error("Initial hessian does not have the correct dimensions!"))
    end

    cand_val = val
    D = val
    ξ = ξ0
    prevξ = zero(uType)
    Δ = if isnothing(cache.solver_args.initial_radius)
        0.1*sqrt(dot(gr,H,gr))
    else
        uType(cache.solver_args.initial_radius)
    end
    ρ = zero(uType)

    opt_state = OptimizationBase.OptimizationState(
        ; iter=0, u = x, objective = val, grad = gr, hess = H, original = (;Δ,D,ρ,lb,ub), p = cache.p)
    cb_call = cache.callback(opt_state, val)

    fevals = 1
    iterations = 0
    t0 = time()
    retcode = ReturnCode.MaxIters
    @inbounds for iter in 1:maxiters
        # Optimizality condition takes the scaling into account
        ng = norm(sqrt_abs_scale .^ 2 .* gr,Inf)
        if ng < abstol || ng < reltol * abs(val) 
            retcode = ReturnCode.Success
            break
        end
        iterations += 1

        θ = max(θmin,1-ng)

        n_stalls = 0
        trials = 0
        stop = false
        while true
            if isnan(Δ)
                retcode = ReturnCode.Failure
                stop = true
                break
            end

            box[] = (cache,gr,H,Δ,p,x,lb,ub,sqrt_abs_scale,dscale,θ,cand_val,cand,cand_gr)
            trials += 1
            Δm,bound_augmentation = solve_tr_subproblem!(gr,H,Δ,p,x,lb,ub,sqrt_abs_scale,dscale,θ)
            ns = norm(p ./ sqrt_abs_scale)

            copyto!(cand,x)
            cand .+= p

            if Δ <= sqrt(eps(zero(uType))) || x == cand
                retcode = ReturnCode.StalledSuccess
                stop = true
                break
            end

            cand_val = eval_val_and_grad!(cache,cand_gr,cand)
            fevals += 1
            update_hessian!(H,y,val,gr,p,cand_val,cand_gr)

            ρ = if Δm > 0
                (val-cand_val-bound_augmentation+δ)/(Δm+δ)
            else
                -one(uType)
            end

            # @info "" iter Δ val cand_val D ρ Δm ns ng bound_augmentation θ norm(x-cand) norm(gr-cand_gr)

            if ρ < c1
                # Large reduction in trust region
                Δ = min(α1*Δ,ns/4)
            elseif c1 <= ρ < c2
                # Trust region slighly reduced if below c2
                Δ *= α1 + ((ρ-c1)/(c2-c1))^2*(1-α1)
            else
                if ns > r*Δ # Only increase if step is at the edge
                    # For very successful iterations the radius is not increase
                    # because of probable model missmatch
                    Δ *= α3 + (α2-α3)*exp(-((ρ-1)/(c2-1))^2)
                end
            end

            # NonMonotone contribution
            if ρ + (D-val)/(Δm+δ) > c1 && Δm > 0
                break
            end
        end

        if stop break end

        # Update for next iter
        val = cand_val
        copyto!(x,cand)
        copyto!(gr,cand_gr)
        update_scaling!(sqrt_abs_scale,dscale,x,gr,lb,ub)
        D = (ξ*D+val)/(ξ+1)
        ξ,prevξ = ξ/2 + prevξ/2,ξ


        # Check best
        if val < best_val 
            best_val = val
            copyto!(best_x,x)
        end

        # Callback
        opt_state = OptimizationBase.OptimizationState(
            ; iter=iterations, u = x, objective = val, grad = gr, hess = H, p = cache.p, 
            original = (;Δ,D,ρ,lb,ub)
        )
        cb_call = cache.callback(opt_state, val)
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
                                               time = t1-t0, fevals, gevals=fevals)

    if !cache.solver_args.save_best
        return SciMLBase.build_solution(cache, cache.opt, x, val;
                                        stats,retcode)
    end


    return SciMLBase.build_solution(cache, cache.opt, best_x, best_val;
                                    stats,retcode,original = (;hess=H,Δ))
end

end
