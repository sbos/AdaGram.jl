push!(LOAD_PATH, "./src/")

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
  "model"
    help = "path to serialized model"
    arg_type = String
    required = true
  "text"
    help = "text to measure likelihood"
    arg_type = String
    required = true
  "--window"
    help = "window size"
    arg_type = Int
    default = 5
  "--workers"
    help = "number of workers"
    arg_type = Int
    default = 1
  "--minprob"
    help = "minimum probability of a prototype"
    arg_type = Float64
    default = 0.005
  "--remove-top-K"
    help = "ignore K most frequent words"
    arg_type = Int
    default = 0
end

args = parse_args(ARGS, s)
addprocs(args["workers"])

require("AdaGram.jl")

using AdaGram

vm, dict = load_model(args["model"])
println(parallel_likelihood(vm, dict, args["text"], args["window"], args["minprob"]; remove_top_k=args["remove-top-K"]))
