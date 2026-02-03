
import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path to allow importing calculate_6th_order_model
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

import calculate_6th_order_model

def measure_time_ms(func, *args, **kwargs):
    """Measure execution time of a function in milliseconds."""
    # Warmup? Maybe not needed for simple calculations, but good practice.
    # However, these functions might be expensive, so just run once.
    
    start = time.perf_counter()
    func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000 # Convert to ms

def main():
    print("Starting specific model timing tests (M, MM, RBF) from calculate_6th_order_model.py...")
    results = []

    # Load data using logic similar to process_dataset
    filename = 'shanghai_zhou2018.csv'
    data_dir = os.path.join(current_dir, 'data')
    source_file = os.path.join(data_dir, filename)
    
    try:
        df = pd.read_csv(source_file)
        print(f"Loaded {filename}: {len(df)} rows")
        
        x1 = df.iloc[:, 1]
        x2 = df.iloc[:, 2]
        x3 = df.iloc[:, 3]
        x4 = df.iloc[:, 4]
        x5 = df.iloc[:, 5]
        x6 = df.iloc[:, 6]
        
        y = df.iloc[:, 7]
        ly = df.iloc[:, 8]
        ry = df.iloc[:, 9]
        
        # 1. RBF Model
        print("\n--- Timing RBF Model ---")
        X_rbf = np.column_stack([x1, x2, x3, x4, x5, x6])
        # hesamian_rbf_fit_predict(X_train, y, l, r, X_test=None, M=10, lam=1e-3)
        t_rbf = measure_time_ms(
            calculate_6th_order_model.hesamian_rbf_fit_predict, 
            X_rbf, y, ly, ry, M=10
        )
        print(f"RBF Time: {t_rbf:.4f} ms")
        results.append({
            "Dataset": "Shanghai",
            "Model": "RBF",
            "Time (ms)": t_rbf,
            "Note": "hesamian_rbf_fit_predict (M=10)"
        })
        
        # 2. M Model (6th Order)
        print("\n--- Timing M Model ---")
        # fuzzy_six_linear_regression(x1, x2, x3, x4, x5, x6, y, ly, ry)
        t_m = measure_time_ms(
            calculate_6th_order_model.fuzzy_six_linear_regression,
            x1, x2, x3, x4, x5, x6, y, ly, ry
        )
        print(f"M Model Time: {t_m:.4f} ms")
        results.append({
            "Dataset": "Shanghai",
            "Model": "M",
            "Time (ms)": t_m,
            "Note": "fuzzy_six_linear_regression"
        })
        
        # 3. MM Model
        print("\n--- Timing MM Model ---")
        # calculate_mm_model(x1, x2, x3, x4, x5, x6)
        t_mm = measure_time_ms(
            calculate_6th_order_model.calculate_mm_model,
            x1, x2, x3, x4, x5, x6
        )
        print(f"MM Model Time: {t_mm:.4f} ms")
        results.append({
            "Dataset": "Shanghai",
            "Model": "MM",
            "Time (ms)": t_mm,
            "Note": "calculate_mm_model (Prediction only)"
        })

    except Exception as e:
        print(f"Error measuring time: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    df = pd.DataFrame(results)
    out_file = "model_calculation_times.csv"
    out_path = current_dir / out_file
    df.to_csv(out_path, index=False)
    
    print("\n" + "="*30)
    print(f"Timing tests completed.")
    print(f"Results saved to: {out_path}")
    print("="*30)
    print(df)

if __name__ == "__main__":
    main()
