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