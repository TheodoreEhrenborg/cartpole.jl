### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ bde90012-e6b4-11ed-058f-2786b140fd16
using ReinforcementLearning

# ╔═╡ c655ef32-e93b-4997-a782-c235b3796b37
using StatsBase # for mean

# ╔═╡ f00b4d6a-d77d-4943-aff3-c1ae6c9dd934
using Test

# ╔═╡ 277527c7-8f78-4b8d-8552-75eb74547472
using Flux

# ╔═╡ 7ae55b9f-944c-4a6d-a00e-fc0bfccce9cc
using Optimisers

# ╔═╡ 9b590668-6fa3-4138-8118-eb1ffdf0c291
how_long = 1000

# ╔═╡ 342afffc-9b82-4e2b-ad07-e588459b94bf
"Uses model to guide decisions on cartpoleenv
and returns the total number of moves until
termination"
function value(model; max_steps = 100) 
    c3 = CartPoleEnv(max_steps=max_steps)
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

# ╔═╡ 15ac24f3-0230-4f75-af53-7e1af8e9d4b3
"Uses model to guide decisions on cartpoleenv
and returns the total number of moves until
termination. No randomness"
function value_max(model; max_steps = 100) 
    c3 = CartPoleEnv(max_steps=max_steps)
	i = 0
	while i < max_steps
		i+=1
        prob = model(state(c3))[1]
		#println(prob)
		choice = 0.5<prob ? 1 : 2		
	    c3(choice)
	    is_terminated(c3) && break
	end
	i
end

# ╔═╡ 194261d8-399e-4bbd-ac2d-269f78796a4d
normalize(v) = (v.-mean(v))/std(v)

# ╔═╡ ae713b21-8f8b-49dc-b9e0-4031dff7dd1d
#https://stackoverflow.com/a/53645744
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

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
function value2(model; max_steps=100) 
    c3 = CartPoleEnv(max_steps=max_steps)
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
    d = Chain(Dense(4=>1))
    @test abs(mean_value(d) - mean_value(d)) < 0.5
	@test abs(mean_value(d) - mean_value2(d)) < 0.5
end

# ╔═╡ 4be52fa9-7506-4639-9e08-a88abdf57d9c
run_test()

# ╔═╡ 5b5973fc-cfa1-4529-b72d-a65fa8771f17
function inner(model, opt, runs, max_steps)
		scores, grads = unzip([value2(model, max_steps=max_steps) for _ in 1:runs])
	   
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
function train_loop(the_model; iters=100, runs_per_iter=100, max_steps=100)
	rule = Optimisers.Adam()
	opt = Optimisers.setup(rule, the_model)
	##println(mean_value(the_model))
	for _ in 1:iters
        opt, the_model = inner(the_model, opt, runs_per_iter, max_steps)
	end
	the_model
end

# ╔═╡ fec94493-eefc-4182-91d5-329897775763
value2(Chain(Dense(4=>10),Dense(10=>1)))

# ╔═╡ 8c540ca4-36bd-4589-a1a9-6cc3791df90a
m1 = Chain(Dense(4=>3, relu), Dense(3=>2),softmax)

# ╔═╡ c2883f65-3789-45e7-b38f-9c3702458053
mean_value(m1)

# ╔═╡ d93a8ba1-9f52-4268-a546-6ef93c7335e0
m2 = train_loop(m1)

# ╔═╡ 7e28f6f3-6878-4e5c-b2bd-35645a3b8d7c
mean_value(m2)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
ReinforcementLearning = "158674fc-8238-5cab-b5ba-03dfc80d1318"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
Flux = "~0.13.4"
Optimisers = "~0.2.9"
ReinforcementLearning = "~0.10.2"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "2ea21e4a3580ab1e5884829e35f2bbe72f3703a3"

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
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

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

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
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

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

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

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

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

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═bde90012-e6b4-11ed-058f-2786b140fd16
# ╠═c655ef32-e93b-4997-a782-c235b3796b37
# ╠═18e44fbc-30b2-4a27-a11b-dfe50008f9ee
# ╠═9b590668-6fa3-4138-8118-eb1ffdf0c291
# ╠═3447f889-db39-4936-8f40-63861a7e8504
# ╠═9517e49e-1b4c-454f-8230-4007240a4316
# ╠═f00b4d6a-d77d-4943-aff3-c1ae6c9dd934
# ╠═b5b942e8-2d4f-45ac-b030-c7c346c0dbe5
# ╠═4be52fa9-7506-4639-9e08-a88abdf57d9c
# ╠═342afffc-9b82-4e2b-ad07-e588459b94bf
# ╠═15ac24f3-0230-4f75-af53-7e1af8e9d4b3
# ╠═277527c7-8f78-4b8d-8552-75eb74547472
# ╠═194261d8-399e-4bbd-ac2d-269f78796a4d
# ╠═7ae55b9f-944c-4a6d-a00e-fc0bfccce9cc
# ╠═344797a6-6083-4201-afce-6ef338cf14e1
# ╠═5b5973fc-cfa1-4529-b72d-a65fa8771f17
# ╠═ae713b21-8f8b-49dc-b9e0-4031dff7dd1d
# ╠═fec94493-eefc-4182-91d5-329897775763
# ╠═c65a1343-e23e-49e7-9562-fb0e58c70bf7
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
# ╠═8c540ca4-36bd-4589-a1a9-6cc3791df90a
# ╠═c2883f65-3789-45e7-b38f-9c3702458053
# ╠═d93a8ba1-9f52-4268-a546-6ef93c7335e0
# ╠═7e28f6f3-6878-4e5c-b2bd-35645a3b8d7c
# ╠═5560b842-6c93-4115-bb0d-6d79bd713e50
# ╠═94718dc4-ab7b-46bb-8e66-2e21cba162f2
# ╠═cbf3fcd1-ab1f-422c-b611-8a79b105ec56
# ╠═b96d59c4-9684-4945-9828-90c1bff9f3f9
# ╠═343c520d-61bf-4b83-9fd4-3028d47b5168
# ╠═b66d2c21-160e-49ce-bd6e-12ef2f04d74a
# ╠═21b4a207-df33-4262-afb9-cab8bc933e8f
# ╠═1493cf9c-0e54-4b81-b986-6efa6b512d95
# ╠═f2aaca41-4a17-4f20-85ba-c0ae11ed875e
# ╠═6c25e268-bcef-41da-88fb-b6e3d0293788
# ╠═d1d676e7-527a-4bc7-a380-b54689d8dbb0
# ╠═c6aa1a92-237b-4e9e-b455-0cfe9a20d99b
# ╠═139e884a-fd83-4a15-8653-9fdd76e54882
# ╠═223f0d82-ccbb-4fe4-b006-7141a690eda2
# ╠═8611444a-5b97-41a7-add5-8f894f3cfd0d
# ╠═71da9797-573b-4e08-a77d-0bea7959c463
# ╠═076a98d2-c1bc-409b-b1f5-b29c68dca6d7
# ╠═e7bdbbf7-b4da-48d8-bdb0-ba202a408a72
# ╠═66ed1157-c4bb-45f8-b8af-f9ba6ec71f64
# ╠═30731751-ed31-4aec-9d20-10b813881d0d
# ╠═2dbfaf3f-4a51-41ff-a419-a01231c77236
# ╠═1b7d5cae-01ac-46c7-9e43-d0d7c9f47317
# ╠═f7f4563c-3955-4473-a0e0-08e2fd0e3b4a
# ╠═144456b9-b521-4d98-858c-2d81de1c707b
# ╠═72cc54c2-30b1-4703-a45b-8a0c78bf215f
# ╠═2f16ca8b-fad5-4f14-a8e5-07aae0fa1df6
# ╠═34b51055-a7a8-446c-801b-15ed4ecdd473
# ╠═feb46d13-8548-4222-abf2-e4122f613ccd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
