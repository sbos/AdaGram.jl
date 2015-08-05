using AdaGram

type Context
    position::Int
    words::Array{String}
end

type Sense
    sense::String
    weight::Int
    has_utf::Bool
    contexts::Array{Context}
end

type Word
    plain_word::String
    word::String
    weight::Int
    number_of_senses::Int
    senses::Array{Sense}
end

const MIN_WORD_WEIGHT = 500
const MAX_NUMBER_OF_CONTEXTS = 500

function read_and_process_dataset(dataset_number, vm, dict, stopwords, CONTEXT_SEMIWIDTH, result_filename)
    aris = Float64[]
    f = open("datasets/dataset-$dataset_number", "r")
    while !eof(f)
        tokens = split(strip(readline(f)), '\t')
        plain_word = tokens[1]
        word = lowercase(tokens[2])
        word_weight = int32(tokens[3])
        number_of_senses = int32(tokens[4])
        senses = Array(Sense, number_of_senses)
        for i in 1:number_of_senses
            if eof(f) break end

            tokens = split(strip(readline(f)), '\t')
            has_utf = endof(tokens[1]) != length(tokens[1])
            sense = remove_utf_characters(lowercase(tokens[1]))
            sense_weight = int32(tokens[2])
            number_of_contexts = int32(tokens[3])
            contexts = Array(Context, number_of_contexts)

            for j in 1:number_of_contexts
                position = int32(readline(f)) + 1
                words = split(lowercase(strip(readline(f))), " ")
                if length(words) == 0
                    contexts[j] = Context(position, words)
                end
                left_context = reverse(words[1:min(position,length(words))])
                filter!(w -> !isempty(w) && w in keys(dict.word2id) && !(w in stopwords), left_context)
                left_bound = min(length(left_context), CONTEXT_SEMIWIDTH)
                if position < length(words)
                    right_context = words[(position+1):end]
                    filter!(w -> !isempty(w) && w in keys(dict.word2id) && !(w in stopwords), right_context)
                    right_bound = min(length(right_context), CONTEXT_SEMIWIDTH)
                    words = vcat(left_context[1:left_bound], right_context[1:right_bound])
                else
                    words = left_context[1:left_bound]
                end
                contexts[j] = Context(position, words)
            end

            senses[i] = Sense(sense, sense_weight, has_utf, contexts)
        end
        if endof(word) != length(word) || !(word in keys(dict.word2id))
            # skip non-ASCII words
            continue
        end
        a = process_one_word(Word(plain_word, word, word_weight, number_of_senses, senses), vm, dict)
        if a > -100
            push!(aris, a)
        end
    end
    @printf("%f\n", mean(aris))
    close(f)
end

function remove_utf_characters(s::String)
    if endof(s) == length(s) return s end
    r = Char[]
    for i in 1:endof(s)
        if nextind(s, i) == i + 1 && prevind(s, i + 1) == i
            push!(r, s[i])
        end
    end
    return join(r, "")
end

function get_number_of_senses(vm, dict, word)
    z = Array(Float64,T(vm))
    N = expected_pi!(z, vm, dict.word2id[word])
    return N
end

function get_senses_mask(vm, dict, word)
    z = Array(Float64,T(vm))
    expected_pi!(z, vm, dict.word2id[word])
    return z .> 0
end

function disambiguate_senses(vm, dict, word::Word, N)
    clustering_matrix = zeros(word.number_of_senses, T(vm))
    mask = get_senses_mask(vm, dict, word.word)
    for i in 1:word.number_of_senses
        for j in 1:min(length(word.senses[i].contexts), MAX_NUMBER_OF_CONTEXTS)
            p = disambiguate(vm, dict, word.word, word.senses[i].contexts[j].words)
            clustering_matrix[i, indmax(p)] += 1
        end
    end
    return clustering_matrix[:,mask]
end

function compute_ARI(x)
    a = sum(x, 2)
    b = sum(x, 1)
    index = sum(x.*(x-1)*0.5)
    a1 = sum(a.*(a-1)*0.5)
    b1 = sum(b.*(b-1)*0.5)
    n = sum(x)
    expected_index = a1*b1/(n*(n-1)*0.5)
    max_index = (a1+b1)/2
    return (index - expected_index) / (max_index - expected_index)
end

function process_one_word(word, vm, dict)
    if word.weight >= MIN_WORD_WEIGHT
        N = get_number_of_senses(vm, dict, word.word)
        if N > 1
            clustering_matrix = disambiguate_senses(vm, dict, word, N)
            number_of_all_contexts = sum(clustering_matrix)
            ARI = compute_ARI(clustering_matrix)

#            @printf(f_verbose, "%s\t%d\t%d\n", word.word, word.number_of_senses, N)
#            for i in 1:word.number_of_senses
#                @printf(f_verbose, "%s\n", word.senses[i].sense)
#            end
#            for i in 1:word.number_of_senses
#                for j in 1:N
#                    @printf(f_verbose, "%d ", clustering_matrix[i, j])
#                end
#                @printf(f_verbose, "\n")
#            end
        else
            # hack for faster processing
            ARI = 0
            number_of_all_contexts = 0
            for i in 1:word.number_of_senses
                number_of_all_contexts += length(word.senses[i].contexts)
            end
        end
#        @printf(f, "%s\t%d\t%d\t%d\t%f\n", word.word, N, word.number_of_senses, number_of_all_contexts, ARI)
#        flush(f)
#        flush(f_verbose)
        return ARI
    end
    return -100
end

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
  "model"
    help = "path to serialized model"
    arg_type = String
    required = true
  "result-filename"
    help = "where to save result"
    arg_type = String
    required = true
  "--window"
    help = "window size"
    arg_type = Int
    default = 5
end

args = parse_args(ARGS, s)

vm, dict = load_model(args["model"])
#println("read stopwords")
stopwords = readdlm("benchmark/english.txt")
#println("process dataset")
read_and_process_dataset(14, vm, dict, stopwords, args["window"], args["result-filename"])

