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

Flux.@layer NumberGuesser

function NumberGuesser()
    # layers = Chain(
    #     Dense(99 => 256, bias=false),
    #     Dropout(.1),
    #     Dense(256 => 512, bias=false),
    #     Dropout(.2),
    #     Dense(512 => 256, bias=false),
    #     Dropout(.2),
    #     Dense(256 => 100, relu),
    # )
    layers = Chain(
        Dense(99 => 100, bias=false),
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
        inputs[:, i] = Float32.(sequence[1:99]) ./100
        targets[i] = sequence[100]
    end
    
    return inputs, Flux.onehotbatch(targets, 1:100)
end

losses = Float32[]
success_rates = Float32[]

for epoch in 1:100
    tot_loss = 0f0
    correct_predictions = 0
    total_predictions = 0
    
    for i in 1:10000
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

Random.seed!(1234)

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

savefig(combined_plot, "training_progress.png")

println("\nTraining Statistics:")
println("Final Loss: $(round(losses[end], digits=4))")
println("Best Success Rate: $(round(maximum(success_rates), digits=2))%")
println("Average Success Rate: $(round(mean(success_rates), digits=2))%")
println("Total Training Steps: $(length(losses) * 100)")