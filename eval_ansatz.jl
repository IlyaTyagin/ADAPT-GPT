import ADAPT
import CSV
import DataFrames
import DataFrames: groupby
import Serialization
import LinearAlgebra: norm
import Graphs
import JSON
import JuMP, MQLib
using ProgressBars
import SimpleWeightedGraphs

function suppress_output(f, args...)
    redirect_stdout(devnull) do
        return f(args...)
    end
end

function edgelist_to_graph(edgelist; num_vertices=0)
    if num_vertices == 0
        num_vertices = maximum(max(src, dst) for (src, dst, w) in edgelist);
    end
    g = SimpleWeightedGraphs.SimpleWeightedGraph(num_vertices);

    if length(edgelist[1]) == 3
        # Add edges with weights to the graph
        for (src, dst, w) in edgelist
            Graphs.add_edge!(g, src, dst, w);
        end
    else
        for (src, dst) in edgelist 
            w = 1
            Graphs.add_edge!(g, src, dst, w);
        end
    end
    
    return g
end

function graph_to_edgelist(g)
    weighted_edge_list = [
        (
            SimpleWeightedGraphs.src(e),
            SimpleWeightedGraphs.dst(e),
            SimpleWeightedGraphs.weight(e)
        ) for e in SimpleWeightedGraphs.edges(g)
    ];

    return weighted_edge_list
end 

function eval_ansatz(
    edgelist,
    q_circuit,
    n_nodes,
    op_pool,
)
    #println("Using diagonal QAOA")
    if op_pool == "qaoa_double_pool"
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_nodes); pooltype = "qaoa_double_pool"
    elseif op_pool == "qaoa_mixer"
        pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_mixer(n_nodes); pooltype = "qaoa_mixer"
    elseif op_pool == "two_local_pool"
        pool = ADAPT.Pools.two_local_pool(n_nodes); pooltype = "two_local_pool"
    else
        throw(error("op_pool is not valid."))
    end
    g = edgelist_to_graph(edgelist, num_vertices=n_nodes);
    e_list = graph_to_edgelist(g);
    
    #H = ADAPT.Hamiltonians.maxcut_hamiltonian(n_nodes, e_list);
    
    H_spv = ADAPT.Hamiltonians.maxcut_hamiltonian(n_nodes, e_list);
    # Wrap in a QAOAObservable view.
    H = ADAPT.ADAPT_QAOA.QAOAObservable(H_spv);
    
    ψ0 = ones(ComplexF64, 2^n_nodes) / sqrt(2^n_nodes); ψ0 /= norm(ψ0);
    
    op_indices = [];
    angle_values = [];
    
    generated_list = q_circuit;
    
    # Iterate over the list in steps of 4
    for j in 1:4:length(generated_list)
        #push!(op_indices, generated_list[j+1]);
        op_val = generated_list[j+1]
    
        # Convert to integer if it's a string like "op_5" or "7"
        if isa(op_val, String)
            # Extract digits if the string has a prefix like "op_5"
            parsed_val = tryparse(Int, op_val)
            if isnothing(parsed_val)
                # remove non-digits and try again
                digits_only = replace(op_val, r"\D+" => "")
                parsed_val = tryparse(Int, digits_only)
            end
            push!(op_indices, parsed_val)
        else
            push!(op_indices, Int(op_val))
        end
        push!(angle_values, generated_list[j+3]);
        push!(angle_values, generated_list[j+2]);
    end

    angles = convert(Array{Float64}, angle_values);

    #ansatz = ADAPT.ADAPT_QAOA.QAOAAnsatz(0.01, H);
    ansatz = ADAPT.ADAPT_QAOA.DiagonalQAOAAnsatz(0.01, pool, H);
    #ansatz = ADAPT.Basics.Ansatz(1.0, pool) 

    for op_idx in op_indices
        push!(ansatz, pool[op_idx] => 0.0);
        # NOTE: this step adds both H and the pool operator to the ansatz
    end

    ADAPT.bind!(ansatz, angles);  #= <- this is your reconstructed ansatz =#

    # TEST: EVALUATE FINAL ENERGY - SHOULD MATCH LAST ENERGY FOR THAT "run"
    ψEND = ADAPT.evolve_state(ansatz, ψ0);
    E_final = ADAPT.evaluate(H, ψEND);
    return E_final
end