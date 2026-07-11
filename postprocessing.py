import pandas as pd
import numpy as np
from scipy import stats
from config import MODEL_INFO

def compute_activation_similarity_metrics(model, seq):
    layer_indices = [x for x in range(MODEL_INFO[model]['num_layers'])]
    values = [seq[f'activation_cos_sim_layer_{i}'].mean() for i in layer_indices]

    slope, _, r, p, _ = stats.linregress(layer_indices, values)
    auc = np.trapezoid(values, layer_indices)
    
    return slope, p, r**2, auc, values[-1]

def compute_aggregate_metrics(model, ptb_type):
    ptb_pcts = [x*5 for x in range(1, 11)] if ptb_type != 'shuffle' else [x*5 for x in range(1, 21)]
    nlls = []
    output_divs = []
    act_sim_slope = []
    act_sim_p_val = []
    act_sim_r2 = []
    act_sim_auc = []
    act_sim_final = []

    for pct in ptb_pcts:
        seq = pd.read_csv(f'results_{model}/{ptb_type}/{pct}/evals.csv')
        nlls.append(np.mean(np.log(seq['perplexity'])))
        output_divs.append(np.mean(seq['output_divergence']))

        slope, p_val, r2, auc, final = compute_activation_similarity_metrics(model, seq)
        act_sim_slope.append(slope)
        act_sim_p_val.append(p_val)
        act_sim_r2.append(r2)
        act_sim_auc.append(auc)
        act_sim_final.append(final)

    # Saves Model x Ptb type
    aggregate_data = {
        "ptb_pct": ptb_pcts,
        "nll": nlls,
        "output_divs": output_divs,
        "act_sim_slope": act_sim_slope,
        "act_sim_p_val": act_sim_p_val,
        "act_sim_r2": act_sim_r2,
        "act_sim_auc": act_sim_auc,
        "act_sim_final": act_sim_final
    }
    
    df = pd.DataFrame(aggregate_data)
    df.to_csv(f'results_{model}/{ptb_type}/aggregate_metrics_no_entropy.csv', index=False)

if __name__ == "__main__":
    for model in MODEL_INFO.keys():
        for ptb_type in ['shuffle', 'token', 'char']:
            print(f"Computing aggregate metrics for model: {model}, perturbation type: {ptb_type}")
            compute_aggregate_metrics(model, ptb_type)