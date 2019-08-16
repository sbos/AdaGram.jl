mutable struct Kahan{T <: AbstractFloat}
	sum::T
	c::T
end

Kahan(::Type{T}) where {T <: AbstractFloat} = Kahan{T}(convert(T, 0), convert(T, 0))

#import NumericExtensions.add!
function add!(k::Kahan{T}, x::T) where {T <: AbstractFloat}
	y = x - k.c
	t = k.sum + y
	k.c = (t - k.sum) - y
	k.sum = t
end

import Base.sum
sum(k::Kahan) = k.sum

mutable struct MeanCounter{T <: AbstractFloat}
	n::Int64
	mean::Kahan{T}
end

MeanCounter(::Type{T}) where {T <: AbstractFloat} = MeanCounter{T}(0, Kahan(T))

function add!(m::MeanCounter{T}, x::T) where {T <: AbstractFloat}
	m.n += 1
	add!(m.mean, (x - sum(m.mean)) / m.n)
	return m.mean
end

mean(m::MeanCounter) = sum(m.mean)

export Kahan, MeanCounter, add!, sum, mean
