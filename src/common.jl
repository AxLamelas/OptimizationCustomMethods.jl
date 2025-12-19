function eval_val_and_grad!(cache::OptimizationCache,g,x)
  if cache.f.fg !== nothing
    first(cache.f.fg(g,x,cache.p))
  else
    cache.f.grad(g,x,cache.p)
    cache.f(x,cache.p)
  end
end

function iden!(M::AbstractMatrix{T}) where T
  for I in CartesianIndices(M)
    if I[1] == I[2]
      M[I] = one(T)
    else
      M[I] = zero(T)
    end
  end
end

