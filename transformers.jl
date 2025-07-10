using Pkg
Pkg.activate(".")
using Flux, Plots, Statistics, Random, Onion
using BSON: @save 


Random.seed!(1234)

function getRandomNumbers(n)
    return rand(1:100, n)
end

struct NumberGuesser{L}
    layers::L
end

Flux.@layer NumberGuesser

function NumberGuesser()
    embed_dim = 16
    n_heads = 8
    seq_len = 99
    
    layers = (
        encoder = Dense(1 => embed_dim, bias=false),
        transformers = Tuple(Onion.TransformerBlock(embed_dim, n_heads) for _ in 1:2),
        rope = RoPE(div(embed_dim, n_heads), seq_len),
        decoder = Dense(embed_dim => 1, bias=false),
    )
    return NumberGuesser(layers)
end

function (m::NumberGuesser)(input)
    l = m.layers
    
    x = reshape(input, 1, :)
    x = l.encoder(x)

    x = reshape(x, size(x, 1), size(input, 1), size(input, 2))

    for transformerblock in l.transformers
        x = transformerblock(x, 0, l.rope)
    end

    x = x[:, end, :]

    output = l.decoder(x)
    return output
end

model = NumberGuesser()

opt_state = Flux.setup(AdamW(eta = 0.001), model)

function create_batch(batch_size)
    inputs = zeros(Float32, 99, batch_size)
    targets = zeros(Float32, 1, batch_size)
    
    for i in 1:batch_size
        sequence = getRandomNumbers(100)
        inputs[:, i] = Float32.(sequence[1:99]) ./ 100
        targets[1, i] = Float32(sequence[100]) / 100
    end
    
    return inputs, targets
end

losses = Float32[]
success_rates = Float32[]

for epoch in 1:10
    tot_loss = 0f0
    correct_predictions = 0
    total_predictions = 0
    
    for i in 1:1000
        inputs, targets = create_batch(128)
        
        l, grad = Flux.withgradient(model) do m
            predictions = m(inputs)
            Flux.mse(predictions, targets)
        end
        
        Flux.update!(opt_state, model, grad[1])
        tot_loss += l
        
        predicted_numbers = round.(Int, model(inputs) .* 100)
        actual_numbers = round.(Int, targets .* 100)
        correct_predictions += sum(predicted_numbers .== actual_numbers)
        total_predictions += length(actual_numbers)
        
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

@save "number_guesser_model.bson" model
println("Model saved to number_guesser_model.bson")

final_inputs, final_targets = create_batch(5000)
final_predictions = model(final_inputs)
predicted_numbers = round.(Int, final_predictions .* 100)
actual_numbers = round.(Int, final_targets .* 100)
final_success_rate = (sum(predicted_numbers .== actual_numbers) / length(actual_numbers)) * 100

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
