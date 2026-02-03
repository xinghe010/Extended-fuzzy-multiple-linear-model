import numpy as np
import pandas as pd
from scipy import integrate
import os

# Set file paths
# Source data files are in code/data directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
results_dir = os.path.join(os.path.dirname(__file__), 'results')

# Ensure results directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def process_dataset(filename, is_normal=False):
    """
    Process a single dataset
    is_normal: If True, indicates normal fuzzy number format (x, y, sigma), otherwise (x, y, ly, ry)
    """
    print(f"\n{'='*20} Processing file: {filename} {'='*20}")
    source_file = os.path.join(data_dir, filename)
    
    # Read data
    try:
        df = pd.read_csv(source_file)
    except FileNotFoundError:
        print(f"Error: File not found {source_file}")
        return

    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    
    if is_normal:
        # example_2_normal.csv format: x, y, sigma
        # Assume ly = ry = sigma
        ly = df.iloc[:, 2]
        ry = df.iloc[:, 2]
    else:
        # Other file formats: x, y, ly, ry
        ly = df.iloc[:, 2]
        ry = df.iloc[:, 3]

    n = len(x)
    print(f"Number of data rows: {n}")

    print("Calculating model parameters...")
    # 1. Calculate 1st-order model parameters
    a, la, ra, b, lb, rb = calculate_parameters(x, y, ly, ry)
    
    print(f"Parameter calculation results: a={a}, la={la}, ra={ra}, b={b}, lb={lb}, rb={rb}")

    # Save parameters to results folder
    base_name = os.path.splitext(filename)[0]
    params_df = pd.DataFrame({
        'Parameter': ['a', 'la', 'ra', 'b', 'lb', 'rb'],
        'Value': [a, la, ra, b, lb, rb]
    })
    params_path = os.path.join(results_dir, f'{base_name}_parameters.csv')
    params_df.to_csv(params_path, index=False)
    print(f"Parameters saved to: {params_path}")

    # 2. Calculate estimates for the 1st-order model
    print("Calculating estimates...")
    y_p = x * a + b
    ly_p = x * la + lb
    ry_p = x * ra + rb
    
    # Round to 3 decimal places
    y_p = round(y_p, 2)
    ly_p = round(ly_p, 2)
    ry_p = round(ry_p, 2)

    # Save estimates
    estimates_df = pd.DataFrame({
        'y_p': y_p,
        'ly_p': ly_p,
        'ry_p': ry_p
    })
    estimates_path = os.path.join(results_dir, f'{base_name}_estimates.csv')
    estimates_df.to_csv(estimates_path, index=False)
    print(f"Estimates saved to: {estimates_path}")

    # 3. Calculate EI_DC, EI_DH, EI_DL, EI_DB between estimates and actual values
    print("Calculating error indices EI...")
    
    # EI_DC (Coppi)
    d_Coppi = D_Coppi_nfn(y, ly, ry, y_p, ly_p, ry_p)
    ei_dc = round(EI(d_Coppi, n), 4)
    print(f"EI_DC (Coppi): {ei_dc}")

    # EI_DH (Hassanpour)
    d_Hassanpour = D_Hassanpour(y, ly, ry, y_p, ly_p, ry_p)
    ei_dh = round(EI(d_Hassanpour, n), 4)
    print(f"EI_DH (Hassanpour): {ei_dh}")

    # EI_DL (Li)
    d_Li = D_Li(y, ly, ry, y_p, ly_p, ry_p)
    ei_dl = round(EI(d_Li, n), 4)
    print(f"EI_DL (Li): {ei_dl}")

    # EI_DB (Bargiela)
    d_Bargiela = D_Bargiela(y, ly, ry, y_p, ly_p, ry_p)
    ei_db = round(EI(d_Bargiela, n), 4)
    print(f"EI_DB (Bargiela): {ei_db}")

    # Save EI results
    ei_df = pd.DataFrame({
        'Metric': ['EI_DC', 'EI_DH', 'EI_DL', 'EI_DB'],
        'Value': [ei_dc, ei_dh, ei_dl, ei_db]
    })
    ei_path = os.path.join(results_dir, f'{base_name}_error_indices.csv')
    ei_df.to_csv(ei_path, index=False)
    print(f"Error indices saved to: {ei_path}")

def calculate_parameters(x, y, ly, ry):
    """
    Calculate parameters a, la, ra, b, lb, rb for 1st-order fuzzy regression model
    """
    n = len(x)
    # Calculate parameters
    x_sum = x.sum()# Calculate sum of x
    y_sum = y.sum()# Calculate sum of y
    ly_sum = ly.sum()# Calculate sum of ly
    ry_sum = ry.sum()# Calculate sum of ry
    xx_sum = (x ** 2).sum()# Calculate sum of x^2
    xy_sum = (x * y).sum()# Calculate sum of x*y
    xly_sum = (x * ly).sum()# Calculate sum of x*ly
    xry_sum = (x * ry).sum()# Calculate sum of x*ry

    # Calculate a, la, ra
    # Denominator
    denominator = n * xx_sum - x_sum * x_sum
    
    a = (n * xy_sum - x_sum * y_sum) / denominator
    la = (n * xly_sum - x_sum * ly_sum) / denominator
    ra = (n * xry_sum - x_sum * ry_sum) / denominator

    # Round to 3 decimal places (refer to original notebook)
    a = round(a, 3)
    la = round(la, 3)
    ra = round(ra, 3)

    # Calculate b, lb, rb
    b = (y_sum - a * x_sum) / n
    lb = (ly_sum - la * x_sum) / n
    rb = (ry_sum - ra * x_sum) / n

    # Round to 3 decimal places
    b = round(b, 3)
    lb = round(lb, 3)
    rb = round(rb, 3)

    return a, la, ra, b, lb, rb

# Coppi(2006) distance function - Calculate normal fuzzy number - D_Coppi_nfn
def D_Coppi_nfn(a, la, ra, b, lb, rb):
    # Here a, la, ra are actual values y, ly, ry
    # b, lb, rb are estimated values y_p, ly_p, ry_p
    d_Coppi_nfn = (a - b)**2 + (la - lb)**2 / 2 * np.pi + (ra - rb)**2 / 6 * np.pi
    return d_Coppi_nfn

# Hassanpour(2010) distance function - D_Hassanpour
def D_Hassanpour(a, la, ra, b, lb, rb):
    d_Hassanpour = (abs(a - b) + abs(la - lb) + abs(ra - rb))**2
    return d_Hassanpour

# Li(2001) distance function - D_Li
def integrand_Li(x, a, la, ra, b, lb, rb):
    # Integrand function
    return x * (((a - b) - (1 - x) * (la - lb))**2 + ((a - b) + (1 - x) * (ra - rb))**2)

def D_Li(a, la, ra, b, lb, rb):
    d_Li = []
    # Reset index to ensure alignment
    a = a.reset_index(drop=True)
    la = la.reset_index(drop=True)
    ra = ra.reset_index(drop=True)
    b = b.reset_index(drop=True)
    lb = lb.reset_index(drop=True)
    rb = rb.reset_index(drop=True)
    
    for i in range(len(a)):
        d_Li_result, error_Li_result = integrate.quad(integrand_Li, 0, 1, args=(a[i], la[i], ra[i], b[i], lb[i], rb[i]))
        d_Li.append(d_Li_result)
    d_Li = pd.Series(d_Li)
    return d_Li

# Bargiela(2007) distance function - D_Bargiela
def integrand_Bargiela_1(x, a, la, b, lb):
    return (a - (1 - x) * la - b + (1 - x) * lb)**2

def integrand_Bargiela_2(x, a, ra, b, rb):
    return (a + (1 - x) * ra - b - (1 - x) * rb)**2

def D_Bargiela(a, la, ra, b, lb, rb):
    d_Bargiela = []
    # Reset index to ensure alignment
    a = a.reset_index(drop=True)
    la = la.reset_index(drop=True)
    ra = ra.reset_index(drop=True)
    b = b.reset_index(drop=True)
    lb = lb.reset_index(drop=True)
    rb = rb.reset_index(drop=True)
    
    for i in range(len(a)):
        d_Bargiela_result_1, _ = integrate.quad(integrand_Bargiela_1, 0, 1, args=(a[i], la[i], b[i], lb[i]))
        d_Bargiela_result_2, _ = integrate.quad(integrand_Bargiela_2, 0, 1, args=(a[i], ra[i], b[i], rb[i]))
        d_Bargiela_result = d_Bargiela_result_1 + d_Bargiela_result_2
        d_Bargiela.append(d_Bargiela_result)
    d_Bargiela = pd.Series(d_Bargiela)
    return d_Bargiela

# Calculate EI (Error Index)
def EI(d_yy, n):
    if len(d_yy) == 0:
        return 0
    # Avoid division by zero
    denominator = max(d_yy) - min(d_yy)
    if denominator == 0:
        return 0 # Or other appropriate value when all distances are equal
    ei = ((max(d_yy) - d_yy) / denominator).sum() / n
    return ei

def process_estimates(filename, y_true, ly_true, ry_true):
    """
    Process file containing estimates and calculate error indices
    filename: Filename containing estimates (x, y_p, ly_p, ry_p)
    y_true, ly_true, ry_true: Actual values
    """
    print(f"\n{'='*20} Evaluating estimates file: {filename} {'='*20}")
    source_file = os.path.join(data_dir, filename)
    
    # Read estimate data
    try:
        df = pd.read_csv(source_file)
    except FileNotFoundError:
        print(f"Error: File not found {source_file}")
        return

    # Assume file format is x, y, ly, ry, where y, ly, ry are actually y_p, ly_p, ry_p
    x = df.iloc[:, 0]
    y_p = df.iloc[:, 1]
    ly_p = df.iloc[:, 2]
    ry_p = df.iloc[:, 3]

    n = len(x)
    print(f"Number of data rows: {n}")
    
    # Check if row counts match
    if n != len(y_true):
        print(f"Warning: Estimate row count ({n}) does not match actual value row count ({len(y_true)})!")
        return

    # 3. Calculate EI_DC, EI_DH, EI_DL, EI_DB between estimates and actual values
    print("Calculating error indices EI...")
    
    # EI_DC (Coppi)
    d_Coppi = D_Coppi_nfn(y_true, ly_true, ry_true, y_p, ly_p, ry_p)
    ei_dc = round(EI(d_Coppi, n), 4)
    print(f"EI_DC (Coppi): {ei_dc}")

    # EI_DH (Hassanpour)
    d_Hassanpour = D_Hassanpour(y_true, ly_true, ry_true, y_p, ly_p, ry_p)
    ei_dh = round(EI(d_Hassanpour, n), 4)
    print(f"EI_DH (Hassanpour): {ei_dh}")

    # EI_DL (Li)
    d_Li = D_Li(y_true, ly_true, ry_true, y_p, ly_p, ry_p)
    ei_dl = round(EI(d_Li, n), 4)
    print(f"EI_DL (Li): {ei_dl}")

    # EI_DB (Bargiela)
    d_Bargiela = D_Bargiela(y_true, ly_true, ry_true, y_p, ly_p, ry_p)
    ei_db = round(EI(d_Bargiela, n), 4)
    print(f"EI_DB (Bargiela): {ei_db}")

    # Save EI results
    base_name = os.path.splitext(filename)[0]
    ei_df = pd.DataFrame({
        'Metric': ['EI_DC', 'EI_DH', 'EI_DL', 'EI_DB'],
        'Value': [ei_dc, ei_dh, ei_dl, ei_db]
    })
    ei_path = os.path.join(results_dir, f'{base_name}_error_indices.csv')
    ei_df.to_csv(ei_path, index=False)
    print(f"Error indices saved to: {ei_path}")

def main():
    # 1. First read actual value data (example_2_normal.csv)
    normal_file = 'example_2_normal.csv'
    print(f"Reading benchmark actual value file: {normal_file}")
    source_file = os.path.join(data_dir, normal_file)
    
    try:
        df_normal = pd.read_csv(source_file)
    except FileNotFoundError:
        print(f"Error: File not found {source_file}")
        return

    x_true = df_normal.iloc[:, 0]
    y_true = df_normal.iloc[:, 1]
    # example_2_normal.csv format: x, y, sigma (ly=ry=sigma)
    ly_true = df_normal.iloc[:, 2]
    ry_true = df_normal.iloc[:, 2]
    
    # 2. Perform original modeling and self-test on example_2_normal.csv (keep original logic)
    process_dataset(normal_file, is_normal=True)

    # 3. Evaluate example_2_Epanechnikov.csv and example_2_Gaussion.csv
    # These files contain estimates, compare directly with y_true, ly_true, ry_true above
    
    # Process example_2_Epanechnikov.csv
    process_estimates('example_2_Epanechnikov.csv', y_true, ly_true, ry_true)

    # Process example_2_Gaussion.csv
    process_estimates('example_2_Gaussion.csv', y_true, ly_true, ry_true)

if __name__ == "__main__":
    main()
