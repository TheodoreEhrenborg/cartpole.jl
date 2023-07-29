### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ a10ebf42-ff08-49d9-8c2b-2d685b619552
using StatsBase

# ╔═╡ bde90012-e6b4-11ed-058f-2786b140fd16
using ReinforcementLearning

# ╔═╡ f00b4d6a-d77d-4943-aff3-c1ae6c9dd934
using Test

# ╔═╡ 277527c7-8f78-4b8d-8552-75eb74547472
using Flux

# ╔═╡ 25559ec2-1e6a-440e-a96a-7c767236c489
using Maybe

# ╔═╡ 7ae55b9f-944c-4a6d-a00e-fc0bfccce9cc
using Optimisers

# ╔═╡ 1632deff-3e48-42e9-86f8-3fd0cd46ea16
using JET

# ╔═╡ 91ac3fa9-715c-4d60-aa9d-f78a04416b8f
using Profile

# ╔═╡ 1a57012d-7162-431c-bad2-0dba54f20649
using PProf

# ╔═╡ 17ffd50e-ccf7-4769-8120-ebebb8e7353f
using ProfileVega

# ╔═╡ ab36c00b-0a21-440e-af9c-0d2b45b0b8b5
using StatProfilerHTML

# ╔═╡ 879818d6-93bc-4124-b2a0-15dd2991738c
using BenchmarkTools

# ╔═╡ 9b590668-6fa3-4138-8118-eb1ffdf0c291
how_long = 1000

# ╔═╡ 3882e4a7-bf84-49cc-a4f8-11db7ff90731
import Zygote

# ╔═╡ a5142872-e996-4955-a553-fe020229cd2f
begin
	import Base.:+
	+(a::NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}, b::NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}) = (weight=a.weight+b.weight, bias = a.bias+b.bias, σ=nothing )
	+(a::Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}}, b::Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}}) = map(+,a,b)
	+(a::NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float64}, Vector{Float64}, Nothing}}, b::NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float64}, Vector{Float64}, Nothing}}) = (weight=a.weight+b.weight, bias = a.bias+b.bias, σ=nothing )
	+(a::Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float64}, Vector{Float64}, Nothing}}}, b::Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float64}, Vector{Float64}, Nothing}}}) = map(+,a,b)
end

# ╔═╡ 342afffc-9b82-4e2b-ad07-e588459b94bf
"Uses model to guide decisions on cartpoleenv
and returns the total number of moves until
termination"
function value(model; max_steps = 100) 
    c3 = CartPoleEnv()
	i = 0
	while i < max_steps
		i+=1
        prob = model(state(c3))[1]
		choice = rand()<prob ? 1 : 2		
	    c3(choice)
	    is_terminated(c3) && break
	end
	i
end

# ╔═╡ 3447f889-db39-4936-8f40-63861a7e8504
function mean_value(model; max_steps = 100)
    mean(value(model,max_steps=max_steps) for _ in 1:how_long )
end

# ╔═╡ a50b6f90-6110-44b3-bc15-581b577ce5db
begin
	import Base.:/
	/(a::NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}, b::Int64) = (weight=a.weight/b, bias = a.bias/b, σ=nothing )
	/(a::Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}}, b::Int64) = a ./b
end

# ╔═╡ 194261d8-399e-4bbd-ac2d-269f78796a4d
normalize(v) = (v.-mean(v))/std(v)

# ╔═╡ a2cc4273-55e1-4557-a359-2ebe9df6a2c2
begin
	import Base.:*
	*(a::Float32, b::NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}) = (weight=a*b.weight, bias = a*b.bias, σ=nothing )
	*(a::Float32, b::Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}}) = a .* b
end

# ╔═╡ 43b8a22d-4e61-4f83-b9e4-194ce6cdd0f7
# ╠═╡ disabled = true
#=╠═╡
@report_opt train_loop(10)
  ╠═╡ =#

# ╔═╡ fafe5139-fada-4220-a3c3-9fd8b26aed9a
# ╠═╡ disabled = true
#=╠═╡
@report_opt train_loop()
  ╠═╡ =#

# ╔═╡ 83f60297-511a-41d3-bd3b-c949e52326df
# ╠═╡ disabled = true
#=╠═╡
@time train_loop(10)
  ╠═╡ =#

# ╔═╡ 7d984dfa-5422-4c64-97c9-4dfa363665f1
# ╠═╡ disabled = true
#=╠═╡
@time train_loop(100)
  ╠═╡ =#

# ╔═╡ 29856ae9-199b-4110-975e-84323b4cce17
# ╠═╡ disabled = true
#=╠═╡
@time train_loop(100)
  ╠═╡ =#

# ╔═╡ 10eb6772-bdb7-495c-86f4-987a7282a0eb
# ╠═╡ disabled = true
#=╠═╡
@time train_loop(100)
  ╠═╡ =#

# ╔═╡ 55eb8a75-8936-4068-9f13-6b7f0bbdae99
# ╠═╡ disabled = true
#=╠═╡
train_loop(1000)
  ╠═╡ =#

# ╔═╡ 79bea52d-49ce-4e17-9956-b8d827b7d6b9
# Achieves 40.843 game steps with train_loop(1000)

# ╔═╡ c4e91730-4e06-4c47-b30b-6c6efffd0410
# ╠═╡ disabled = true
#=╠═╡
train_loop(100)
  ╠═╡ =#

# ╔═╡ 68125a71-3e66-4015-9a66-6678055f5ff5
# ╠═╡ disabled = true
#=╠═╡
train_loop(1000)
  ╠═╡ =#

# ╔═╡ 7f977297-cc87-453f-962c-46abb21d17c9
# ╠═╡ disabled = true
#=╠═╡
@time train_loop(1000)
  ╠═╡ =#

# ╔═╡ 9bd01014-a0ed-4ec0-b373-d0285dca1e96
# 2023 07 13 2024: 
# @time train_loop()  # 2nd time around, with 100 iterations inside
# 1.600064 seconds (10.15 M allocations: 416.659 MiB, 8.33% gc time)

# ╔═╡ 02b21f61-1a79-4485-b70e-91b2a4f95f9d


# ╔═╡ 14f949cc-83f0-4a80-8761-c92cd266498e
# ╠═╡ disabled = true
#=╠═╡
Profile.clear()
  ╠═╡ =#

# ╔═╡ 0872cf33-00c2-4957-80cb-1a9df2c682af
# ╠═╡ disabled = true
#=╠═╡
ProfileVega.view() 
  ╠═╡ =#

# ╔═╡ ab02baf2-a158-41d1-bce9-6e65c9d9ed87
# ╠═╡ disabled = true
#=╠═╡
statprofilehtml()
  ╠═╡ =#

# ╔═╡ d89da777-d2b9-4239-bbcf-f80f1b5e642a


# ╔═╡ 4f2c101b-9d9c-4fe0-bbec-7d900092fdeb
# ╠═╡ disabled = true
#=╠═╡
 Profile.Allocs.@profile train_loop(10)
  ╠═╡ =#

# ╔═╡ cab0fd58-a026-4ab3-ba2e-d9fe63b12819
# ╠═╡ disabled = true
#=╠═╡
 @profile train_loop(10)
  ╠═╡ =#

# ╔═╡ e68d263e-78be-4e15-bc02-ff4838d0e748
# ╠═╡ disabled = true
#=╠═╡
pprof()
  ╠═╡ =#

# ╔═╡ a476c0ee-9c9e-4fd2-9d8a-97428268bf8f
# ╠═╡ disabled = true
#=╠═╡
Profile.print()
  ╠═╡ =#

# ╔═╡ ae713b21-8f8b-49dc-b9e0-4031dff7dd1d
#https://stackoverflow.com/a/53645744
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

# ╔═╡ eb4dedbc-2786-4ce7-8e91-9eebc159e6d0
# ╠═╡ disabled = true
#=╠═╡
@benchmark train_loop(10)
  ╠═╡ =#

# ╔═╡ 3c085ea5-a5da-40e1-baef-0136eff9e8a8
m = Chain(Dense(4=>1))

# ╔═╡ 45ca0fbf-fa47-4bc6-be8c-5d6255c2e0bd
@benchmark value(m)

# ╔═╡ 84e3bf36-c6d6-4f49-a683-0c46007b7a3d
@code_warntype value(m)

# ╔═╡ da1718b3-5be2-4ef0-8eaf-c647830bff05
@report_opt value(m)

# ╔═╡ 101f5f45-c6f8-49d2-b62c-306388f97089
@report_call value(m)

# ╔═╡ 5f7658aa-d96e-45bd-9942-98f138fc8535
# ╠═╡ disabled = true
#=╠═╡
m2 = Dense(4=>1)
  ╠═╡ =#

# ╔═╡ 63a2ffc2-aefe-4599-a042-e3d42447ac3f
#=╠═╡
m2.bias
  ╠═╡ =#

# ╔═╡ 108093f2-9496-4f75-9b04-cbefe9fad675
#=╠═╡
m2.weight  .= [0. 0. 0. -1. ]
  ╠═╡ =#

# ╔═╡ 94a9129e-113f-41ad-bf17-b93f3e4a1fdd
#=╠═╡
mean_value(m2)
  ╠═╡ =#

# ╔═╡ e4ed6a37-82ae-45b6-a7c7-b6b0ef19439d
#=╠═╡
m2.bias
  ╠═╡ =#

# ╔═╡ 5490a57f-5bf4-40c3-9532-86ad6aadb8a6
#=╠═╡
m2.weight
  ╠═╡ =#

# ╔═╡ 48992180-9233-4d63-89f5-f57fb15d58cd
Dense

# ╔═╡ 57e0206e-6d3d-407f-8794-1edc2c0fab66
m.bias

# ╔═╡ 05351604-dddc-4663-80bc-a66e4c4aff08
Dense

# ╔═╡ 2108b686-4675-42e9-8fbe-bd5a11e101f5
mean_value(Chain(Dense(4=>2), softmax))

# ╔═╡ 97417a67-0749-4ae0-bd48-dd000ee9eac2
i = 9


# ╔═╡ d73af3b6-2a82-4c2d-883c-469437378821
typeof(one(Float32)/i)

# ╔═╡ f9f1e243-ba0a-456d-b218-abe9f452d26d
typeof(1/i)

# ╔═╡ 4d8c2c4f-18dd-49a6-a369-ac098fbd3357
function add_grads(a, b)
    println(a)
	println( b)
	println(a[1])
		println(a[1][1
		])
end

# ╔═╡ 48a596d7-2306-44b5-9b3e-408eee296742
md"""
Looks like a gradient of a chain has the form:
1-element Tuple a, containing a named tuple b.
b has one key, layers. b["layers"] is a tuple
of c1, c2, .... Each of the c's is either a named
tuple containing a weight, a bias, and sigma = nothing.
Or it's just nothing, in the case where the layer is
a softmax etc

((layers = ((weight = Float32[0.0 0.0 -0.0 0.0; -0.0 -0.0 0.0 -0.0; -0.0049724816 -0.0039834105 0.004610585 -0.0016869778], bias = Float32[0.0, -0.0, -0.13413496], σ = nothing), (weight = Float32[0.0 0.0 0.007637123; -0.0 -0.0 -0.007637123], bias = Float32[0.2499832, -0.2499832], σ = nothing), nothing),),)
"""

# ╔═╡ 053a6034-b95e-4e36-a02b-2b5a4f68423c
mul_scal_grad_chain4(a::Nothing, s::Float32) = nothing

# ╔═╡ 30113064-75c8-4970-ad40-3033436880e3
mul_scal_grad_chain4(a::NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}, s::Float32 ) = (weight=a.weight*s, bias = a.bias*s, σ=nothing )

# ╔═╡ 6836a3a0-1817-45de-8edc-d6f3b9a59fc2
function mul_scal_grad_chain3(a,s::Float32)
    return mul_scal_grad_chain4.(a,s)
end

# ╔═╡ 287cc341-d327-4daf-8e71-b80f3beb017e
mul_scal_grad_chain2(a, s::Float32) = (layers = mul_scal_grad_chain3(a[1],s),)

# ╔═╡ 064825df-9182-4d7e-b2ee-486675573393
mul_scal_grad_chain(a, s::Float32) = (mul_scal_grad_chain2(a[1],s),)

# ╔═╡ 788842d4-7c1f-46ef-b8b0-599764b7b428
function add_grad_chain4(a::Nothing,b::Nothing)
    return nothing
end

# ╔═╡ d3576111-4444-4f20-b760-2d81f0a6f664
function add_grad_chain4(a::NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}}, b::NamedTuple{(:weight, :bias, :σ), Tuple{Matrix{Float32}, Vector{Float32}, Nothing}})
    return (weight=a.weight+b.weight, bias = a.bias+b.bias, σ=nothing )
end

# ╔═╡ 151b065e-f294-4ef8-820b-b9324e117d9c
function add_grad_chain3(a,b)
    return add_grad_chain4.(a,b)
end

# ╔═╡ 53bd7558-da62-4e17-b8b0-4d4bcb43f6f7
function add_grad_chain2(a, b)
    return (layers = add_grad_chain3(a[1],b[1]),)
end

# ╔═╡ d304a360-898d-426d-b6d4-4b6ee3bd2943
function add_grad_chain(a, b)
    return (add_grad_chain2(a[1],b[1]),)
end

# ╔═╡ c65a1343-e23e-49e7-9562-fb0e58c70bf7
"Like value but also returns the gradient of model's
parameters, averaged over all choices the model made.
Adjusting the parameters by this gradient will make the model
more likely to act as it does in this playout"
function value2(model) 
	max_steps = 100
    c3 = CartPoleEnv()
	i = 0
	grad_list = []
	choices = []
	while i < max_steps
		i+=1
		choice = 1
		the_state = state(c3)
        prob_chosen, grads_chosen  = Flux.withgradient(model) do mm
	        # prob of picking 1
            prob = mm(the_state)[1] 
			#println(prob)
			choice = rand()<prob ? 1 : 2
			return choice == 1 ? prob : 1 - prob
        end
		push!(grad_list,grads_chosen)
	    c3(choice)
	    is_terminated(c3) && break
	end
	add_grad_chain(grad_list[1], grad_list[2])
	i , mul_scal_grad_chain( reduce(add_grad_chain, grad_list), one(Float32) / i)
end

# ╔═╡ 9517e49e-1b4c-454f-8230-4007240a4316
function mean_value2(model)
    mean(value2(model)[1] for _ in 1:how_long )
end

# ╔═╡ b5b942e8-2d4f-45ac-b030-c7c346c0dbe5
function run_test()
    d = Dense(4=>1)
    @test abs(mean_value(d) - mean_value(d)) < 0.5
	@test abs(mean_value(d) - mean_value2(d)) < 0.5
end

# ╔═╡ 4be52fa9-7506-4639-9e08-a88abdf57d9c
run_test()

# ╔═╡ 956e3dbc-cfea-42da-8082-4e5ce4a7c4b1
mean_value2(Chain(Dense(4=>2), softmax))

# ╔═╡ 88145d6a-6460-4f57-9e36-ca73b88f002c
mean_value2(Chain(Dense(4=>3, relu), Dense(3=>2),softmax))

# ╔═╡ 5b5973fc-cfa1-4529-b72d-a65fa8771f17
function inner(model, opt, runs)
		scores, grads = unzip([value2(model) for _ in 1:runs])
	   
	  println(mean(scores))
	  #collect(scores)
	  #scores, normalize(scores)
	  # This will give negative coefficients to 
	  # games that were longer than average.
	  # The optimizer will move in the opposite
	  # direction from the gradient, and hence
	  # push us towards longer games
	  #print(typeof(scores))
	  #print(typeof(convert(Vector{Float32},normalize(scores))))
	  net_grad = reduce(add_grad_chain, 
		  mul_scal_grad_chain.(grads, 
		  -convert(Vector{Float32},normalize(scores)) ) )
	  #println(net_grad)
	  #println(value2(the_model)[2][1])
      #println(Base.summarysize.([scores, grads, net_grad, the_model, opt])  )
	  Optimisers.update(opt, model, net_grad[1])
end

# ╔═╡ 344797a6-6083-4201-afce-6ef338cf14e1
function train_loop(the_model, iters, runs_per_iter)
	rule = Optimisers.Adam()
	opt = Optimisers.setup(rule, the_model)
	##println(mean_value(the_model))
	for _ in 1:iters
        opt, the_model = inner(the_model, opt, runs_per_iter)
	end
	the_model
end

# ╔═╡ fed8b5e5-0040-4972-90fa-de52ff612ba1
# ╠═╡ disabled = true
#=╠═╡
train_loop(100,1000)
  ╠═╡ =#

# ╔═╡ fadc6568-4f23-4d7b-aeac-50d92d184666
# ╠═╡ disabled = true
#=╠═╡
train_loop(1000,100)
  ╠═╡ =#

# ╔═╡ ecd79dd2-6f3d-4081-844d-25bcc5e92ed2
# ╠═╡ disabled = true
#=╠═╡
train_loop(1000,100)
  ╠═╡ =#

# ╔═╡ f2f41afc-e3e6-4420-80dc-045652ef4e2e
# ╠═╡ disabled = true
#=╠═╡
train_loop(100,1000)
  ╠═╡ =#

# ╔═╡ 4b148b5d-39b8-4ff6-8ec6-9f3f38eb7e29
# ╠═╡ disabled = true
#=╠═╡
train_loop(1000,10)
  ╠═╡ =#

# ╔═╡ 6bdc8595-1e1a-40d7-b627-b421c317eedc
# ╠═╡ disabled = true
#=╠═╡
train_loop(1000,10)
  ╠═╡ =#

# ╔═╡ 08f7943e-af41-4b71-b582-6ae5e2ac1c6f
@benchmark value2(m)

# ╔═╡ 3958b805-ea77-435c-aef4-c98de806d90a
@code_warntype value2(m)

# ╔═╡ 56d8e9a3-698f-4034-8291-efc5973cbcc8
@report_opt value2(m)

# ╔═╡ fec94493-eefc-4182-91d5-329897775763
value2(Chain(Dense(4=>10),Dense(10=>1)))

# ╔═╡ db552ebf-b80b-4bb3-afc8-13dd072db940
(foo=3,)

# ╔═╡ c9f95559-fe36-4f17-bca8-8517caa5d01c
g = ((layers = ((weight = Float32[0.0 0.0 -0.0 0.0; -0.0 -0.0 0.0 -0.0; -0.0049724816 -0.0039834105 0.004610585 -0.0016869778], bias = Float32[0.0, -0.0, -0.13413496], σ = nothing), (weight = Float32[0.0 0.0 0.007637123; -0.0 -0.0 -0.007637123], bias = Float32[0.2499832, -0.2499832], σ = nothing), nothing),),)

# ╔═╡ 9572abdd-f934-4733-ac27-b8f302f9bfbd
add_grad_chain(g,g)

# ╔═╡ 5cf76fd1-3688-467b-af65-607c3c1892d1
# Next step is to define special functions for multiplying a scalar by a chain grad (and division is just 1/x)

# ╔═╡ df57d71a-cf54-4b65-a430-d81665e16ab9
m1 = Chain(Dense(4=>3, relu), Dense(3=>2),softmax)

# ╔═╡ e62457c8-a81b-47e7-8574-ce41efffef83
m2( [1, 2, 3, 4])[1]

# ╔═╡ bfaf492f-9664-4281-9fae-7bc02696ead0
m2( [1, 2, 3, 4])

# ╔═╡ 46927cd4-ddb6-4bee-9d1e-2794034abe48
m1.layers[1].weight

# ╔═╡ 2b26270d-5ccd-463f-80a5-4eee4bf0439c
m2.layers[1].weight

# ╔═╡ 288fecc0-a9c2-4a19-aa5c-9cbffeb95982
mean_value(m3)

# ╔═╡ 94718dc4-ab7b-46bb-8e66-2e21cba162f2
m4 = train_loop(m3,100,100)

# ╔═╡ cbf3fcd1-ab1f-422c-b611-8a79b105ec56
m5 = train_loop(m4,100,100)

# ╔═╡ b96d59c4-9684-4945-9828-90c1bff9f3f9
m6 = train_loop(m5,100,100)

# ╔═╡ 343c520d-61bf-4b83-9fd4-3028d47b5168
mean_value(m6)

# ╔═╡ b66d2c21-160e-49ce-bd6e-12ef2f04d74a
m7 = train_loop(m6,1000,100)

# ╔═╡ 21b4a207-df33-4262-afb9-cab8bc933e8f
mean_value(m7)

# ╔═╡ 1493cf9c-0e54-4b81-b986-6efa6b512d95
mean_value(m7, max_steps=1000)

# ╔═╡ 6c25e268-bcef-41da-88fb-b6e3d0293788
value(m7, max_steps = 200)

# ╔═╡ 4c452fad-2ca3-4694-99c9-9c147af13e73
# ╠═╡ disabled = true
#=╠═╡
m2 = Chain(Dense(4=>2), softmax)
  ╠═╡ =#

# ╔═╡ 5560b842-6c93-4115-bb0d-6d79bd713e50
m3 = train_loop(m2,100,100)

# ╔═╡ 0d960581-15aa-4af8-92b0-3fbae3869ceb
# ╠═╡ disabled = true
#=╠═╡
m3=Dense([-1.1152266 0.023610264 0.60426503 -0.49385205], [0.40893412] )
  ╠═╡ =#

# ╔═╡ be3ae2d0-3711-43fd-a7c6-74b6d8ccfef3
m2 = train_loop(m1,100,100)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
JET = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
Maybe = "334f122f-1118-46cc-837f-bff747ee6f78"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PProf = "e4faabce-9ead-11e9-39d9-4379958e3056"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
ProfileVega = "4391764f-db79-4bd7-a4c6-f9062de4300e"
ReinforcementLearning = "158674fc-8238-5cab-b5ba-03dfc80d1318"
StatProfilerHTML = "a8a75453-ed82-57c9-9e16-4cd1196ecbf5"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
BenchmarkTools = "~1.3.2"
Flux = "~0.13.4"
JET = "~0.8.7"
Maybe = "~0.1.7"
Optimisers = "~0.2.9"
PProf = "~2.2.2"
ProfileVega = "~1.1.1"
ReinforcementLearning = "~0.10.2"
StatProfilerHTML = "~1.5.0"
StatsBase = "~0.33.21"
Zygote = "~0.6.62"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "c92c699d1ec24bcde823225f98aa4c91f66f588c"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "cad4c758c0038eea30394b1b671526921ca85b21"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "2b301c2388067d655fe5e4ca6d4aa53b61f895b4"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.31"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "SnoopPrecompile", "Static"]
git-tree-sha1 = "dedc16cbdd1d32bead4617d27572f582216ccf23"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.25"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BufferedStreams]]
git-tree-sha1 = "5bcb75a2979e40b29eb250cb26daab67aa8f97f5"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "6717cb9a3425ebb7b31ca4f832823615d175f64a"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.13.1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "2afc496e94d15a1af5502625246d172361542133"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.52.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CircularArrayBuffers]]
deps = ["Adapt"]
git-tree-sha1 = "a05b83d278a5c52111af07e2b2df64bf7b122f8c"
uuid = "9de3a189-e0c0-4e15-ba3b-b14b9fb0aec1"
version = "0.1.10"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "d730914ef30a06732bdd9f763f6cc32e92ffbff1"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonRLInterface]]
deps = ["Tricks"]
git-tree-sha1 = "6c7d1ebb157fdf0f696698ef01946fe93c9efff4"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.3"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "6c0100a8cf4ed66f66e2039af7cde3357814bad2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cf25ccb972fec4e4817764d01c82386ae94f77b4"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.14"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "e76a3281de2719d7c81ed62c6ea7057380c87b1d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.98"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "e1c40d78de68e9a2be565f0202693a158ec9ad85"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.11"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "03b753748fd193a7f2730c02d880da27c5a24508"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.ExternalDocstrings]]
git-tree-sha1 = "1224740fc4d07c989949e1c1b508ebd49a65a5f6"
uuid = "e189563c-0753-4f5e-ad5c-be4293c83fb4"
version = "0.1.1"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7072f1e3e5a8be51d525d64f63d3ec1287ff2790"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.11"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "d9eee53657f6a13ee51120337f98684c9c702264"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "0.2.10"

[[deps.Flux]]
deps = ["Adapt", "ArrayInterface", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test", "Zygote"]
git-tree-sha1 = "96dc065bf4a998e8adeebc0ff1302902b6e59362"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.4"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "cdba9b84cad7ddb89a326e10bf48d6dd4ffd0252"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.2"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "b5c7fe9cea653443736d264b85466bad8c574f4a"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.9"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "2e57b4a4f9cc15e85a24d603256fe08e527f48d1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.8.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "19d693666a304e8c371798f4900f7435558c7cde"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.17.3"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "bb198ff907228523f3dee1070ceee63b9359b6ab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "659140c9375afa2f685e37c1a0b9c9a60ef56b40"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.7"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphviz_jll]]
deps = ["Artifacts", "Cairo_jll", "Expat_jll", "JLLWrappers", "Libdl", "Pango_jll", "Pkg"]
git-tree-sha1 = "a5d45833dda71048117e8a9828bef75c03b18b1c"
uuid = "3c863552-8265-54e4-a6dc-903eb78fde85"
version = "2.50.0+1"

[[deps.HAML]]
deps = ["DataStructures", "Markdown", "Requires"]
git-tree-sha1 = "0e2bbef3c669498254a034394d6dd809e7a97ad6"
uuid = "0bc81568-2411-4001-9bf1-c899fa54f385"
version = "0.3.5"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "2ee0eb8746650f498ed9a109383aa399b2a0c515"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.10"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "ce7ea9cc5db29563b1fe20196b6d23ab3b111384"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.18"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "bcf640979ee55b652f3b01650444eb7bbe3ea837"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "edd1c1ac227767c75e8518defdf6e48dbfa7c3b0"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.10"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JET]]
deps = ["InteractiveUtils", "JuliaInterpreter", "LoweredCodeUtils", "MacroTools", "Pkg", "PrecompileTools", "Preferences", "Revise", "Test"]
git-tree-sha1 = "06a42720332b8442d1d651370917e40771c503a3"
uuid = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
version = "0.8.7"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSONSchema]]
deps = ["Downloads", "HTTP", "JSON", "URIs"]
git-tree-sha1 = "58cb291b01508293f7a9dc88325bc00d797cf04d"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "1.1.0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6a125e6a4cb391e0b9adbd1afa9e771c2179f8ef"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.23"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "b48617c5d764908b5fac493cd907cf33cc11eec1"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.6"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "f044a2796a9e18e0531b9b3072b0019a61f264bc"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.17.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "070e4b5b65827f82c16ae0916376cb47377aa1b5"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.18+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "FLoops", "FoldsThreads", "Random", "ShowCases", "Statistics", "StatsBase", "Transducers"]
git-tree-sha1 = "824e9dfc7509cab1ec73ba77b55a916bb2905e26"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.11"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MarchingCubes]]
deps = ["PrecompileTools", "StaticArrays"]
git-tree-sha1 = "c8e29e2bacb98c9b6f10445227a8b0402f2f173a"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Maybe]]
deps = ["Compat", "ExprTools", "ExternalDocstrings", "Try"]
git-tree-sha1 = "20da985e239bb56c8c9f76bdc9cfb87c1638c8f0"
uuid = "334f122f-1118-46cc-837f-bff747ee6f78"
version = "0.1.7"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "72240e3f5ca031937bd536182cb2c031da5f46dd"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.21"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "b05a082b08a3af0e5c576883bc6dfb6513e7e478"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.6"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NodeJS]]
deps = ["Pkg"]
git-tree-sha1 = "905224bbdd4b555c69bb964514cfa387616f0d3a"
uuid = "2bd173c7-0d6d-553b-b6af-13a54713934c"
version = "1.3.0"

[[deps.NodeJS_18_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "82dbe46101dfce63c5a66e7805e6ac453addc4cb"
uuid = "c1e1d063-8311-5f52-a749-c7b05e91ae37"
version = "18.16.1+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cae3153c7f6cf3f069a853883fd1919a6e5bab5b"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.9+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1ef34738708e3f31994b52693286dabcb3d29f6b"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.9"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.PProf]]
deps = ["AbstractTrees", "EnumX", "FlameGraphs", "Libdl", "OrderedCollections", "Profile", "ProgressMeter", "ProtoBuf", "pprof_jll"]
git-tree-sha1 = "16e2ef982c4d637f45553611e96d135490841769"
uuid = "e4faabce-9ead-11e9-39d9-4379958e3056"
version = "2.2.2"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "84a314e3926ba9ec66ac097e3635e270986b0f10"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.9+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "4b2e829ee66d4218e0cef22c0a64ee37cf258c29"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProfileVega]]
deps = ["FlameGraphs", "LeftChildRightSiblingTrees", "Profile", "VegaLite"]
git-tree-sha1 = "a8d273dbc334c3ce54f0f0815d14366a3ec4fcaa"
uuid = "4391764f-db79-4bd7-a4c6-f9062de4300e"
version = "1.1.1"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.ProtoBuf]]
deps = ["BufferedStreams", "Dates", "EnumX", "TOML", "TranscodingStreams"]
git-tree-sha1 = "e957b28fc98ecd13da4f2bdc7e121832e5d18e3e"
uuid = "3349acd9-ac6a-5e09-bcdb-63829b23a429"
version = "1.0.11"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.ReinforcementLearning]]
deps = ["Pkg", "Reexport", "ReinforcementLearningBase", "ReinforcementLearningCore", "ReinforcementLearningEnvironments", "ReinforcementLearningZoo"]
git-tree-sha1 = "fbc0568769d8020c9255384acdfa3d66453a01d8"
uuid = "158674fc-8238-5cab-b5ba-03dfc80d1318"
version = "0.10.2"

[[deps.ReinforcementLearningBase]]
deps = ["AbstractTrees", "CommonRLInterface", "Markdown", "Random", "Test"]
git-tree-sha1 = "1827f00111ea7731d632b8382031610dc98d8747"
uuid = "e575027e-6cd6-5018-9292-cdc6200d2b44"
version = "0.9.7"

[[deps.ReinforcementLearningCore]]
deps = ["AbstractTrees", "Adapt", "ArrayInterface", "CUDA", "CircularArrayBuffers", "Compat", "Dates", "Distributions", "ElasticArrays", "FillArrays", "Flux", "Functors", "GPUArrays", "LinearAlgebra", "MacroTools", "Markdown", "ProgressMeter", "Random", "ReinforcementLearningBase", "Setfield", "Statistics", "StatsBase", "UnicodePlots", "Zygote"]
git-tree-sha1 = "b27ce227a4b98f3353f169e805fd9ca757e54e40"
uuid = "de1b191a-4ae0-4afa-a27b-92d07f46b2d6"
version = "0.8.12"

[[deps.ReinforcementLearningEnvironments]]
deps = ["DelimitedFiles", "IntervalSets", "LinearAlgebra", "MacroTools", "Markdown", "Pkg", "Random", "ReinforcementLearningBase", "Requires", "SparseArrays", "StatsBase"]
git-tree-sha1 = "c47e65c7cdbc8ddaa034af2185d5bf0fc55f5a80"
uuid = "25e41dd2-4622-11e9-1641-f1adca772921"
version = "0.6.12"

[[deps.ReinforcementLearningZoo]]
deps = ["AbstractTrees", "CUDA", "CircularArrayBuffers", "DataStructures", "Dates", "Distributions", "Flux", "IntervalSets", "LinearAlgebra", "Logging", "MacroTools", "Random", "ReinforcementLearningBase", "ReinforcementLearningCore", "Setfield", "Statistics", "StatsBase", "StructArrays", "Zygote"]
git-tree-sha1 = "e4957a13e69e6344620c38a6f957e8e017fc254a"
uuid = "d607f57d-ee1e-4ba7-bcf2-7734c1e31854"
version = "0.5.12"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "1e597b93700fa4045d7189afa7c004e0584ea548"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "7beb031cf8145577fbccacd94b8a8f4ce78428d3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random", "Test"]
git-tree-sha1 = "3be7d49667040add7ee151fefaf1f8c04c8c8276"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.0"

[[deps.StatProfilerHTML]]
deps = ["DataStructures", "Dates", "FlameGraphs", "HAML", "Profile", "Random", "SHA", "StableRNGs", "Test"]
git-tree-sha1 = "770309b9ecbc5e085a3ca4b2c2207d6d9a59125a"
uuid = "a8a75453-ed82-57c9-9e16-4cd1196ecbf5"
version = "1.5.0"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "dbde6766fc677423598138a5951269432b0fcc90"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0da7e6b70d1bb40b1ace3b576da9ea2992f76318"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.0"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "a66fb81baec325cf6ccafa243af573b031e87b00"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.77"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.Try]]
git-tree-sha1 = "56e72d45a7d77e23f6e943d28183eb509119843a"
uuid = "bf1d0ff0-c4a9-496b-85f0-2b0d71c4f32a"
version = "0.1.1"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodePlots]]
deps = ["ColorTypes", "Contour", "Crayons", "Dates", "FileIO", "FreeTypeAbstraction", "LazyModules", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "SparseArrays", "StaticArrays", "StatsBase", "Unitful"]
git-tree-sha1 = "ae67ab0505b9453655f7d5ea65183a1cd1b3cfa0"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "2.12.4"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c4d2a349259c8eba66a00a540d550f122a3ab228"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.15.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "ead6292c02aab389cb29fe64cc9375765ab1e219"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.1"

[[deps.Vega]]
deps = ["BufferedStreams", "DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "JSONSchema", "MacroTools", "NodeJS_18_jll", "Pkg", "REPL", "Random", "Setfield", "TableTraits", "TableTraitsUtils", "URIParser"]
git-tree-sha1 = "4ceef33ea3094377ac0d72be4fa0126e3741089d"
uuid = "239c3e63-733f-47ad-beb7-a12fde22c578"
version = "2.6.1"

[[deps.VegaLite]]
deps = ["Base64", "DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "MacroTools", "NodeJS", "Pkg", "REPL", "Random", "TableTraits", "TableTraitsUtils", "URIParser", "Vega"]
git-tree-sha1 = "3e23f28af36da21bfb4acef08b144f92ad205660"
uuid = "112f6efa-9a02-5b7d-90c0-432ed331239a"
version = "2.6.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5be3ddb88fc992a7d8ea96c3f10a49a7e98ebc7b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.62"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.pprof_jll]]
deps = ["Artifacts", "Graphviz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b004c9fd6294afe24efccc7e2f055436b63cb809"
uuid = "cf2c5f97-e748-59fa-a03f-dda3c62118cb"
version = "1.0.1+0"
"""

# ╔═╡ Cell order:
# ╠═a10ebf42-ff08-49d9-8c2b-2d685b619552
# ╠═bde90012-e6b4-11ed-058f-2786b140fd16
# ╠═9b590668-6fa3-4138-8118-eb1ffdf0c291
# ╠═3447f889-db39-4936-8f40-63861a7e8504
# ╠═9517e49e-1b4c-454f-8230-4007240a4316
# ╠═f00b4d6a-d77d-4943-aff3-c1ae6c9dd934
# ╠═b5b942e8-2d4f-45ac-b030-c7c346c0dbe5
# ╠═4be52fa9-7506-4639-9e08-a88abdf57d9c
# ╠═342afffc-9b82-4e2b-ad07-e588459b94bf
# ╠═277527c7-8f78-4b8d-8552-75eb74547472
# ╠═194261d8-399e-4bbd-ac2d-269f78796a4d
# ╠═3882e4a7-bf84-49cc-a4f8-11db7ff90731
# ╠═25559ec2-1e6a-440e-a96a-7c767236c489
# ╠═a5142872-e996-4955-a553-fe020229cd2f
# ╠═a50b6f90-6110-44b3-bc15-581b577ce5db
# ╠═a2cc4273-55e1-4557-a359-2ebe9df6a2c2
# ╠═7ae55b9f-944c-4a6d-a00e-fc0bfccce9cc
# ╠═1632deff-3e48-42e9-86f8-3fd0cd46ea16
# ╠═344797a6-6083-4201-afce-6ef338cf14e1
# ╠═5b5973fc-cfa1-4529-b72d-a65fa8771f17
# ╠═43b8a22d-4e61-4f83-b9e4-194ce6cdd0f7
# ╠═fafe5139-fada-4220-a3c3-9fd8b26aed9a
# ╠═91ac3fa9-715c-4d60-aa9d-f78a04416b8f
# ╠═1a57012d-7162-431c-bad2-0dba54f20649
# ╠═83f60297-511a-41d3-bd3b-c949e52326df
# ╠═7d984dfa-5422-4c64-97c9-4dfa363665f1
# ╠═29856ae9-199b-4110-975e-84323b4cce17
# ╠═10eb6772-bdb7-495c-86f4-987a7282a0eb
# ╠═55eb8a75-8936-4068-9f13-6b7f0bbdae99
# ╠═79bea52d-49ce-4e17-9956-b8d827b7d6b9
# ╠═c4e91730-4e06-4c47-b30b-6c6efffd0410
# ╠═68125a71-3e66-4015-9a66-6678055f5ff5
# ╠═7f977297-cc87-453f-962c-46abb21d17c9
# ╠═9bd01014-a0ed-4ec0-b373-d0285dca1e96
# ╠═02b21f61-1a79-4485-b70e-91b2a4f95f9d
# ╠═14f949cc-83f0-4a80-8761-c92cd266498e
# ╠═17ffd50e-ccf7-4769-8120-ebebb8e7353f
# ╠═0872cf33-00c2-4957-80cb-1a9df2c682af
# ╠═ab36c00b-0a21-440e-af9c-0d2b45b0b8b5
# ╠═ab02baf2-a158-41d1-bce9-6e65c9d9ed87
# ╠═d89da777-d2b9-4239-bbcf-f80f1b5e642a
# ╠═4f2c101b-9d9c-4fe0-bbec-7d900092fdeb
# ╠═cab0fd58-a026-4ab3-ba2e-d9fe63b12819
# ╠═e68d263e-78be-4e15-bc02-ff4838d0e748
# ╠═a476c0ee-9c9e-4fd2-9d8a-97428268bf8f
# ╠═ae713b21-8f8b-49dc-b9e0-4031dff7dd1d
# ╠═879818d6-93bc-4124-b2a0-15dd2991738c
# ╠═eb4dedbc-2786-4ce7-8e91-9eebc159e6d0
# ╠═3c085ea5-a5da-40e1-baef-0136eff9e8a8
# ╠═08f7943e-af41-4b71-b582-6ae5e2ac1c6f
# ╠═45ca0fbf-fa47-4bc6-be8c-5d6255c2e0bd
# ╠═84e3bf36-c6d6-4f49-a683-0c46007b7a3d
# ╠═da1718b3-5be2-4ef0-8eaf-c647830bff05
# ╠═101f5f45-c6f8-49d2-b62c-306388f97089
# ╠═3958b805-ea77-435c-aef4-c98de806d90a
# ╠═56d8e9a3-698f-4034-8291-efc5973cbcc8
# ╠═5f7658aa-d96e-45bd-9942-98f138fc8535
# ╠═63a2ffc2-aefe-4599-a042-e3d42447ac3f
# ╠═108093f2-9496-4f75-9b04-cbefe9fad675
# ╠═94a9129e-113f-41ad-bf17-b93f3e4a1fdd
# ╠═e4ed6a37-82ae-45b6-a7c7-b6b0ef19439d
# ╠═5490a57f-5bf4-40c3-9532-86ad6aadb8a6
# ╠═fed8b5e5-0040-4972-90fa-de52ff612ba1
# ╠═fadc6568-4f23-4d7b-aeac-50d92d184666
# ╠═ecd79dd2-6f3d-4081-844d-25bcc5e92ed2
# ╠═f2f41afc-e3e6-4420-80dc-045652ef4e2e
# ╠═4b148b5d-39b8-4ff6-8ec6-9f3f38eb7e29
# ╠═6bdc8595-1e1a-40d7-b627-b421c317eedc
# ╠═fec94493-eefc-4182-91d5-329897775763
# ╠═0d960581-15aa-4af8-92b0-3fbae3869ceb
# ╠═288fecc0-a9c2-4a19-aa5c-9cbffeb95982
# ╠═48992180-9233-4d63-89f5-f57fb15d58cd
# ╠═57e0206e-6d3d-407f-8794-1edc2c0fab66
# ╠═05351604-dddc-4663-80bc-a66e4c4aff08
# ╠═4c452fad-2ca3-4694-99c9-9c147af13e73
# ╠═e62457c8-a81b-47e7-8574-ce41efffef83
# ╠═bfaf492f-9664-4281-9fae-7bc02696ead0
# ╠═2108b686-4675-42e9-8fbe-bd5a11e101f5
# ╠═956e3dbc-cfea-42da-8082-4e5ce4a7c4b1
# ╠═88145d6a-6460-4f57-9e36-ca73b88f002c
# ╠═c65a1343-e23e-49e7-9562-fb0e58c70bf7
# ╠═97417a67-0749-4ae0-bd48-dd000ee9eac2
# ╠═d73af3b6-2a82-4c2d-883c-469437378821
# ╠═f9f1e243-ba0a-456d-b218-abe9f452d26d
# ╠═4d8c2c4f-18dd-49a6-a369-ac098fbd3357
# ╠═48a596d7-2306-44b5-9b3e-408eee296742
# ╠═d304a360-898d-426d-b6d4-4b6ee3bd2943
# ╠═064825df-9182-4d7e-b2ee-486675573393
# ╠═287cc341-d327-4daf-8e71-b80f3beb017e
# ╠═053a6034-b95e-4e36-a02b-2b5a4f68423c
# ╠═30113064-75c8-4970-ad40-3033436880e3
# ╠═53bd7558-da62-4e17-b8b0-4d4bcb43f6f7
# ╠═6836a3a0-1817-45de-8edc-d6f3b9a59fc2
# ╠═151b065e-f294-4ef8-820b-b9324e117d9c
# ╠═788842d4-7c1f-46ef-b8b0-599764b7b428
# ╠═d3576111-4444-4f20-b760-2d81f0a6f664
# ╠═db552ebf-b80b-4bb3-afc8-13dd072db940
# ╠═c9f95559-fe36-4f17-bca8-8517caa5d01c
# ╠═9572abdd-f934-4733-ac27-b8f302f9bfbd
# ╠═5cf76fd1-3688-467b-af65-607c3c1892d1
# ╠═df57d71a-cf54-4b65-a430-d81665e16ab9
# ╠═be3ae2d0-3711-43fd-a7c6-74b6d8ccfef3
# ╠═46927cd4-ddb6-4bee-9d1e-2794034abe48
# ╠═2b26270d-5ccd-463f-80a5-4eee4bf0439c
# ╠═5560b842-6c93-4115-bb0d-6d79bd713e50
# ╠═94718dc4-ab7b-46bb-8e66-2e21cba162f2
# ╠═cbf3fcd1-ab1f-422c-b611-8a79b105ec56
# ╠═b96d59c4-9684-4945-9828-90c1bff9f3f9
# ╠═343c520d-61bf-4b83-9fd4-3028d47b5168
# ╠═b66d2c21-160e-49ce-bd6e-12ef2f04d74a
# ╠═21b4a207-df33-4262-afb9-cab8bc933e8f
# ╠═1493cf9c-0e54-4b81-b986-6efa6b512d95
# ╠═6c25e268-bcef-41da-88fb-b6e3d0293788
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
