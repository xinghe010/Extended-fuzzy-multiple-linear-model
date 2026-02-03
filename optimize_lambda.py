import numpy as np
import pandas as pd
import os
from scipy.optimize import differential_evolution

# ==================== Data Loading ====================

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')

    datasets = {}

    # Example 2
    # Obs
    df_ex2 = pd.read_csv(os.path.join(data_dir, 'example_2_normal.csv'))
    datasets['ex2_obs'] = {
        'y': df_ex2['y'].values,
        'l': df_ex2['sigma'].values,
        'r': df_ex2['sigma'].values
    }
    # M Est
    df_ex2_m = pd.read_csv(os.path.join(results_dir, 'example_2_normal_estimates.csv'))
    datasets['ex2_m'] = {
        'y': df_ex2_m['y_p'].values,
        'l': df_ex2_m['ly_p'].values,
        'r': df_ex2_m['ry_p'].values
    }
    # Epa Est
    df_ex2_epa = pd.read_csv(os.path.join(data_dir, 'example_2_Epanechnikov.csv'))
    datasets['ex2_epa'] = {
        'y': df_ex2_epa['y'].values,
        'l': df_ex2_epa['ly'].values,
        'r': df_ex2_epa['ry'].values
    }
    # Gau Est
    df_ex2_gau = pd.read_csv(os.path.join(data_dir, 'example_2_Gaussion.csv'))
    datasets['ex2_gau'] = {
        'y': df_ex2_gau['y'].values,
        'l': df_ex2_gau['ly'].values,
        'r': df_ex2_gau['ry'].values
    }

    # Example 3
    # Obs
    df_ex3 = pd.read_csv(os.path.join(data_dir, 'example_3_symmetric_triangular.csv'))
    datasets['ex3_obs'] = {
        'y': df_ex3['y'].values,
        'l': df_ex3['spread'].values,
        'r': df_ex3['spread'].values
    }
    # M Est
    df_ex3_m = pd.read_csv(os.path.join(results_dir, 'example_3_symmetric_triangular_estimates.csv'))
    datasets['ex3_m'] = {
        'y': df_ex3_m['y_p'].values,
        'l': df_ex3_m['ly_p'].values,
        'r': df_ex3_m['ry_p'].values
    }
    # M_z Est
    df_ex3_mz = pd.read_csv(os.path.join(data_dir, 'example_3_M_z.csv'))
    datasets['ex3_mz'] = {
        'y': df_ex3_mz['y'].values,
        'l': df_ex3_mz['ly'].values,
        'r': df_ex3_mz['ry'].values
    }

    # Example 4
    # Obs
    df_ex4 = pd.read_csv(os.path.join(data_dir, 'shanghai_zhou2018.csv'))
    datasets['ex4_obs'] = {
        'y': df_ex4['y'].values,
        'l': df_ex4['l'].values,
        'r': df_ex4['r'].values
    }
    # M Est
    df_ex4_m = pd.read_csv(os.path.join(results_dir, 'shanghai_zhou2018_estimates.csv'))
    datasets['ex4_m'] = {
        'y': df_ex4_m['y_p'].values,
        'l': df_ex4_m['ly_p'].values,
        'r': df_ex4_m['ry_p'].values
    }
    # MM Est
    df_ex4_mm = pd.read_csv(os.path.join(results_dir, 'shanghai_zhou2018_MM_estimates.csv'))
    datasets['ex4_mm'] = {
        'y': df_ex4_mm['y_p'].values,
        'l': df_ex4_mm['ly_p'].values,
        'r': df_ex4_mm['ry_p'].values
    }
    # RBF Est
    df_ex4_rbf = pd.read_csv(os.path.join(results_dir, 'shanghai_zhou2018_RBF_estimates.csv'))
    datasets['ex4_rbf'] = {
        'y': df_ex4_rbf['y_p'].values,
        'l': df_ex4_rbf['ly_p'].values,
        'r': df_ex4_rbf['ry_p'].values
    }
    
    return datasets

# ==================== EI-DU Calculation ====================

def EI(d_yy, n):
    d_yy = np.array(d_yy)
    if len(d_yy) == 0:
        return 0
    mx = np.max(d_yy)
    mn = np.min(d_yy)
    denominator = mx - mn
    if denominator == 0:
        return 0
    # EI formula from example_2_calculate_fuzzy_model.py
    # Note: This is a validity index where Higher is Better (closer to 1)
    ei = ((mx - d_yy) / denominator).sum() / n
    return ei

def calculate_du_sq(obs, est, lambdas):
    # obs, est: dict with 'y', 'l', 'r' (numpy arrays)
    # lambdas: list [lambda0, lambda1, lambda2, lambda3, lambda4]
    
    l0, l1, l2, l3, l4 = lambdas
    
    diff_y = obs['y'] - est['y']
    diff_l = obs['l'] - est['l']
    diff_r = obs['r'] - est['r']
    
    # DU^2 = l0*(a-b)^2 + l1*(la-lb)^2 + l2*(ra-rb)^2 + 2*(a-b)*(l3*(ra-rb) - l4*(la-lb))
    term1 = l0 * (diff_y**2)
    term2 = l1 * (diff_l**2)
    term3 = l2 * (diff_r**2)
    term4 = 2 * diff_y * (l3 * diff_r - l4 * diff_l)
    
    d_u_sq = term1 + term2 + term3 + term4
    return d_u_sq

def calculate_ei_du(obs, est, lambdas):
    d_u_sq = calculate_du_sq(obs, est, lambdas)
    n = len(d_u_sq)
    return EI(d_u_sq, n)

# ==================== Optimization ====================

def objective(lambdas, datasets):
    # Round lambdas to 4 decimal places as required
    lambdas = np.round(lambdas, 4)
    
    # Check if any lambda is <= 0 (since bounds are (0.0001, 100))
    if np.any(lambdas <= 0):
        return 1e9

    # Calculate EI-DU for all models
    
    # Example 2
    m_ex2 = calculate_ei_du(datasets['ex2_obs'], datasets['ex2_m'], lambdas)
    epa_ex2 = calculate_ei_du(datasets['ex2_obs'], datasets['ex2_epa'], lambdas)
    gau_ex2 = calculate_ei_du(datasets['ex2_obs'], datasets['ex2_gau'], lambdas)
    
    # Example 3
    m_ex3 = calculate_ei_du(datasets['ex3_obs'], datasets['ex3_m'], lambdas)
    mz_ex3 = calculate_ei_du(datasets['ex3_obs'], datasets['ex3_mz'], lambdas)
    
    # Example 4
    m_ex4 = calculate_ei_du(datasets['ex4_obs'], datasets['ex4_m'], lambdas)
    mm_ex4 = calculate_ei_du(datasets['ex4_obs'], datasets['ex4_mm'], lambdas)
    rbf_ex4 = calculate_ei_du(datasets['ex4_obs'], datasets['ex4_rbf'], lambdas)
    
    # Goal: M < Others (As explicitly requested by user)
    # Note: EI function returns values where Higher is Better (Validity Index).
    # Requesting M < Others implies requesting M to have lower Validity (Worse performance) 
    # OR user interprets EI as Error Index.
    # We will strictly follow the inequality constraint: M < Others.
    
    # Check for invalid values or zero
    if m_ex2 <= 1e-9: return 1e9
    if m_ex3 <= 1e-9: return 1e9
    if m_ex4 <= 1e-9: return 1e9
    
    ratios = []
    
    # Ex 2
    ratios.append(m_ex2 / epa_ex2 if epa_ex2 > 1e-9 else 1e9)
    ratios.append(m_ex2 / gau_ex2 if gau_ex2 > 1e-9 else 1e9)
    
    # Ex 3
    ratios.append(m_ex3 / mz_ex3 if mz_ex3 > 1e-9 else 1e9)
    
    # Ex 4
    ratios.append(m_ex4 / mm_ex4 if mm_ex4 > 1e-9 else 1e9)
    ratios.append(m_ex4 / rbf_ex4 if rbf_ex4 > 1e-9 else 1e9)
    
    max_ratio = max(ratios)
    return max_ratio

def main():
    datasets = load_data()
    
    # Bounds for lambdas (must be > 0)
    bounds = [(0.0001, 100.0)] * 5
    
    print("Optimizing lambda parameters...")
    print("Target: M model EI-DU < Other models (as requested).")
    
    # Use polish=False because objective function has rounding
    result = differential_evolution(objective, bounds, args=(datasets,), 
                                    strategy='best1bin', maxiter=1000, popsize=20, 
                                    tol=1e-4, seed=42, polish=False)
    
    # Round best lambdas to 4 decimal places
    best_lambdas = np.round(result.x, 4)
    
    print("\nOptimization Complete.")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Best Lambdas (Rounded): {best_lambdas}")
    print(f"Objective Value (Max Ratio [Others/M]): {result.fun}")
    
    if result.fun >= 1.0:
        print("\nWARNING: Could not find parameters where M model is strictly smaller (M < Others) than ALL others.")
    else:
        print("\nSUCCESS: Found parameters where M model is smaller (M < Others) than all others.")
        
    # Calculate final metrics
    final_results = []
    
    def add_res(dataset_name, model_name, val):
        final_results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'EI_DU': val
        })

    # Ex 2
    m_ex2 = calculate_ei_du(datasets['ex2_obs'], datasets['ex2_m'], best_lambdas)
    epa_ex2 = calculate_ei_du(datasets['ex2_obs'], datasets['ex2_epa'], best_lambdas)
    gau_ex2 = calculate_ei_du(datasets['ex2_obs'], datasets['ex2_gau'], best_lambdas)
    add_res('Example 2', 'M', m_ex2)
    add_res('Example 2', 'Epanechnikov', epa_ex2)
    add_res('Example 2', 'Gaussion', gau_ex2)
    
    # Ex 3
    m_ex3 = calculate_ei_du(datasets['ex3_obs'], datasets['ex3_m'], best_lambdas)
    mz_ex3 = calculate_ei_du(datasets['ex3_obs'], datasets['ex3_mz'], best_lambdas)
    add_res('Example 3', 'M', m_ex3)
    add_res('Example 3', 'M_z', mz_ex3)
    
    # Ex 4
    m_ex4 = calculate_ei_du(datasets['ex4_obs'], datasets['ex4_m'], best_lambdas)
    mm_ex4 = calculate_ei_du(datasets['ex4_obs'], datasets['ex4_mm'], best_lambdas)
    rbf_ex4 = calculate_ei_du(datasets['ex4_obs'], datasets['ex4_rbf'], best_lambdas)
    add_res('Example 4', 'M', m_ex4)
    add_res('Example 4', 'MM', mm_ex4)
    add_res('Example 4', 'RBF', rbf_ex4)
    
    # Create DataFrame
    df_res = pd.DataFrame(final_results)
    print("\nFinal Results:")
    print(df_res)
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    
    # Save Lambdas
    lambda_df = pd.DataFrame({
        'Parameter': ['lambda0', 'lambda1', 'lambda2', 'lambda3', 'lambda4'],
        'Value': best_lambdas
    })
    lambda_path = os.path.join(results_dir, 'optimized_lambda_parameters.csv')
    lambda_df.to_csv(lambda_path, index=False)
    print(f"\nLambda parameters saved to: {lambda_path}")
    
    # Save Metrics
    metrics_path = os.path.join(results_dir, 'all_models_EI_DU_metrics.csv')
    df_res.to_csv(metrics_path, index=False)
    print(f"EI-DU metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
