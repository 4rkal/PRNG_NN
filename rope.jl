using Pkg
Pkg.activate(".")
using Flux, Plots, Statistics, Random


Random.seed!(1234)

function getRandomNumbers(n)
    return rand(1:100, n)
end

struct NumberGuesser{L}
    layers::L
end

struct RoPE{A<:AbstractArray}
    cos::A
    sin::A
end

function RoPE(dim::Int, end_pos::Int; theta::T=10000f0, start_pos=0) where T
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    freqs_complex = cis.(T.(start_pos:end_pos-1) * freqs')
    cos = permutedims(real(freqs_complex), (2, 1))  # (head_dim/2, seq_len)
    sin = permutedims(imag(freqs_complex), (2, 1))
    cos = reshape(cos, (dim÷2, end_pos - start_pos, 1, 1))
    sin = reshape(sin, (dim÷2, end_pos - start_pos, 1, 1))
    return RoPE(cos, sin)
end 

Flux.@layer RoPE trainable=()

Base.getindex(rope::RoPE, i) = RoPE(rope.cos[:,i,:,:], rope.sin[:,i,:,:])

function (rope::RoPE)(x)
    head_dim = size(x, 1)
    x1 = x[1:head_dim÷2, :, :, :]
    x2 = x[head_dim÷2+1:end, :, :, :]
    return vcat(  
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
    )
end

Flux.@layer NumberGuesser

function NumberGuesser()
    embed_dim = 64
    rope = RoPE(embed_dim, 99)
    
    embedding = Dense(99 => embed_dim * 99, bias=false)
    
    layers = Chain(
        embedding,
        x -> begin
            batch_size = size(x, 2)
            return reshape(x, (embed_dim, 99, 1, batch_size))
        end,
        rope,
        x -> begin
            return reshape(x, (embed_dim * 99, size(x, 4)))
        end,
        Dense(embed_dim * 99 => 512),
        Dense(512 => 100)
    )
    return NumberGuesser(layers)
end

function (m::NumberGuesser)(input)
    return m.layers(input)
end

model = NumberGuesser()

opt_state = Flux.setup(AdamW(eta = 0.001), model)

function create_batch(batch_size)
    inputs = zeros(Float32, 99, batch_size)
    targets = zeros(Int, batch_size)
    
    for i in 1:batch_size
        sequence = getRandomNumbers(100)
        inputs[:, i] = Float32.(sequence[1:99]) ./ 100
        targets[i] = sequence[100]
    end
    
    return inputs, Flux.onehotbatch(targets, 1:100)
end

losses = Float32[]
success_rates = Float32[]

for epoch in 1:10
    tot_loss = 0f0
    correct_predictions = 0
    total_predictions = 0
    
    for i in 1:1000
        inputs, targets = create_batch(32)
        
        l, grad = Flux.withgradient(model) do m
            predictions = m(inputs)
            Flux.logitcrossentropy(predictions, targets)
        end
        
        Flux.update!(opt_state, model, grad[1])
        tot_loss += l
        
        predicted_classes = Flux.onecold(model(inputs), 1:100)
        actual_classes = Flux.onecold(targets, 1:100)
        correct_predictions += sum(predicted_classes .== actual_classes)
        total_predictions += length(actual_classes)
        
        if mod(i, 100) == 0
            avg_loss = tot_loss / 100
            success_rate = (correct_predictions / total_predictions) * 100
            println("Epoch $epoch, Step $i: Loss = $(round(avg_loss, digits=4)), Success Rate = $(round(success_rate, digits=2))%")
            
            push!(losses, avg_loss)
            push!(success_rates, success_rate)
            
            if success_rate > 1.0
            end
            
            tot_loss = 0f0
            correct_predictions = 0
            total_predictions = 0
        end
    end
end

final_inputs, final_targets = create_batch(10000)
final_predictions = model(final_inputs)
predicted_classes = Flux.onecold(final_predictions, 1:100)
actual_classes = Flux.onecold(final_targets, 1:100)
final_success_rate = (sum(predicted_classes .== actual_classes) / length(actual_classes)) * 100

println("\nFinal Success Rate: $(round(final_success_rate, digits=2))%")


p1 = plot(losses, 
          title="Training Loss Over Time",
          xlabel="Training Steps (×100)", 
          ylabel="Average Loss",
          linewidth=2,
          color=:red,
          legend=false)

p2 = plot(success_rates,
          title="Success Rate Over Time", 
          xlabel="Training Steps (×100)",
          ylabel="Success Rate (%)",
          linewidth=2,
          color=:blue,
          legend=false)

combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
display(combined_plot)

savefig(combined_plot, "training_progress_rope.png")

println("\nTraining Statistics:")
println("Final Loss: $(round(losses[end], digits=4))")
println("Best Success Rate: $(round(maximum(success_rates), digits=2))%")
println("Average Success Rate: $(round(mean(success_rates), digits=2))%")
println("Total Training Steps: $(length(losses) * 100)")