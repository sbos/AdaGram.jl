type Kahan{T <: FloatingPoint}
	sum::T
	c::T
end

Kahan{T <: FloatingPoint}(::Type{T}) = Kahan{T}(convert(T, 0), convert(T, 0))

import NumericExtensions.add!
function add!{T <: FloatingPoint}(k::Kahan{T}, x::T)
	y = x - k.c
	t = k.sum + y
	k.c = (t - k.sum) - y
	k.sum = t
end

import Base.sum
sum(k::Kahan) = k.sum

type MeanCounter{T <: FloatingPoint}
	n::Int64
	mean::Kahan{T}
end

MeanCounter{T <: FloatingPoint}(::Type{T}) = MeanCounter{T}(0, Kahan(T))

function add!{T <: FloatingPoint}(m::MeanCounter{T}, x::T)
	m.n += 1
	add!(m.mean, (x - sum(m.mean)) / m.n)
	return m.mean
end

import Base.mean

mean(m::MeanCounter) = sum(m.mean)

export Kahan, MeanCounter, add!, sum, mean