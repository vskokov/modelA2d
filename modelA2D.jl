cd(@__DIR__)
#using Pkg
#Pkg.activate(".")

using Distributions
using StaticArrays
using Random
using DelimitedFiles
using SLEEF
using JLD2
using FFTW
using Printf


seed = parse(Int,ARGS[1])
Random.seed!(seed)

### Lattice Size
const L = parse(Int,ARGS[2])
### end

ξ = Normal(0.0f0, 1.0f0)

### Parameters
const λ = 4.0f0
const Γ = 1.0f0
const T = 1.0f0
### end

### Numerics
const Δt = 0.04f0/Γ
### end

const Rate = Float32(sqrt(2.0*Δt*Γ))

### Nearest neighbors
NNp_a = zeros(Int16,L)
NNm_a = zeros(Int16,L)
for i in 1:L
	NNp_a[i] = i + 1
	NNm_a[i] = i - 1
end
NNp_a[L] = 1
NNm_a[1] = L

# for performance
NNp=@SVector [NNp_a[i] for i in 1:L]
NNm=@SVector [NNm_a[i] for i in 1:L]
###

function hotstart(n)
	rand(ξ, (n, n))
end

function ΔH(m², ϕ, ϕt, x)
	@inbounds ϕold = ϕ[x[1],x[2]]
	Δϕ = ϕt - ϕold
	Δϕ2= ϕt^2 - ϕold^2

	## nearest neighbours
	@inbounds Σnn = ϕ[ NNp[x[1]] , x[2]] + ϕ[x[1], NNp[x[2]]] 
	@inbounds Σnn = Σnn + ϕ[ NNm[x[1]] ,x[2]] + ϕ[x[1],  NNm[x[2]]] 

	kinet_term =  2.0f0 * Δϕ2 - Δϕ * Σnn
	poten_term =  0.5f0 * m² * Δϕ2 + 0.25f0 * λ * (ϕt^4-ϕold^4)

	kinet_term + poten_term
end


function MCstep(m², ϕ, x)
	r = rand(Float32)
	@inbounds ϕt = ϕ[x[1],x[2]] + Rate*rand(ξ)
	p = min(1.0f0,SLEEF.exp(-ΔH(m², ϕ, ϕt, x)))
	if (r<=p)
		@inbounds ϕ[x[1],x[2]] = ϕt
	end
end


function sweep(m², ϕ, L)
	Threads.@threads for i in 1:L
		for j in 1:L
				if (i+j)%2 == 0
					MCstep(m², ϕ, (i,j))
				end
		end
	end

	Threads.@threads for i in 1:L
		for j in 1:L
			if (i+j)%2 !=0
				MCstep(m², ϕ, (i,j))
			end
		end
	end
end

function op(ϕ, L)
	# this function is what you need to compute things we discussed today 
	ϕk = fft(ϕ)
	average = ϕk[1,1]/L^2
	(real(average),ϕk[:,1])
end


function thermalize(m², ϕ, L,  N=10000)
	for t in 1:N
		sweep(m², ϕ, L)
	end
end

function save_state(filename, ϕ, m², i=nothing)
    if isnothing(i)
        jldsave(filename, true; ϕ=ϕ, m²=m²)
    else
        jldsave(filename, true; ϕ=ϕ, m²=m², i=i)
    end
end

###
###
### ************************************** Main ************************************
###
###




## Thermalization stud

max_iter = parse(Int, ARGS[3])

m² = -3.824f0

iter = 1

filename(iter) = "thermalized_L_$(L)_mass_$(m²)_id_$(seed)_$(iter).jld2"

if isfile(filename(max_iter))
	file = jldopen(filename(max_iter), "r")
    global ϕ = Array(file["ϕ"])
    close(file)
	generatedat(max_iter)
else	
	if isfile(filename(1))
    	file = jldopen(filename(1), "r")
    	global ϕ = Array(file["ϕ"])
    	close(file)
	else
    	global ϕ = hotstart(L)
    	save_state(filename(1), ϕ, m²)
	end

	for iter = 2:max_iter
    	if isfile(filename(iter))
        	file = jldopen(filename(iter), "r")
        	global ϕ = Array(file["ϕ"])
        	close(file)
    	else
        	thermalize(m², ϕ, L, 100*L^2)
        	save_state(filename(iter), ϕ, m²)
			end
    	end
	end
