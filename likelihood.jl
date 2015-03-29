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
end

args = parse_args(ARGS, s)
addprocs(args["workers"])

require("AdaGram.jl")

using AdaGram

vm, dict = load_model(args["model"])
println(parallel_likelihood(vm, dict, args["text"], args["window"]))
