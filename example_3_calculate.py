import numpy as np
import pandas as pd
from scipy import integrate
import os

# Set file paths
data_dir = os.path.join(os.path.dirname(__file__), 'data')
results_dir = os.path.join(os.path.dirname(__file__), 'results')

# Ensure results directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ==================== Distance Function Definitions ====================

# Coppi(2006) distance function - Calculate triangular fuzzy number - D_Coppi_tfn
def D_Coppi_tfn(a,la,ra,b,lb,rb):
    d_Coppi_tfn = (a-b)**2 + (la-lb)**2/2 + (ra-rb)**2/6
    return d_Coppi_tfn

# Hassanpour(2010) distance function - D_Hassanpour
def D_Hassanpour(a,la,ra,b,lb,rb):
    d_Hassanpour = (abs(a-b) + abs(la-lb) + abs(ra-rb))**2
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

# 005-Calculate estimated standard error of regression equation - using tfnDYK as distance metric
def SS_tfnDYK_005(a,la,ra,b,lb,rb,k): # k is the order
    n = len(a)
    dd_tfnDYK_005 = 3*(a-b)**2 + 0.25*(la-lb)**2 + 0.25*(ra-rb)**2 + (a-b)*(ra-rb-la+lb)    
    ss_tfnDYK_005 = dd_tfnDYK_005.sum()/(n-k-1)
    return ss_tfnDYK_005

# 005-Calculate estimated standard error of regression equation - using tfnDDK as distance metric
def SS_tfnDDK_005(a,la,ra,b,lb,rb,k): # k is the order
    n = len(a)
    dd_tfnDDK_005 = (a-b)**2 + 1.0/6.0*(la-lb)**2 + 1.0/6.0*(ra-rb)**2 + 0.5*(a-b)*(ra-rb-la+lb)    
    ss_tfnDDK_005 = dd_tfnDDK_005.sum()/(n-k-1)
    return ss_tfnDDK_005

# 004-Calculate average fit degree of symmetric triangular fuzzy numbers
def fit_degree_004tfn(c1, s1, c2, s2):
    n = len(c1)
    q = 0.0
    for i in range(n):
        if c1[i] < c2[i] and c1[i] + s1[i] >= c2[i] - s2[i]:
            q += 1 - (c2[i] - c1[i]) / (s1[i] + s2[i])
        elif c1[i] > c2[i] and c2[i] + s2[i] >= c1[i] - s1[i]:
            q += 1 - (c1[i] - c2[i]) / (s1[i] + s2[i])
        elif c1[i] == c2[i] and s1[i] <= s2[i]:
            q += s1[i] / s2[i]
        elif c1[i] == c2[i] and s1[i] > s2[i]:
            q += s2[i] / s1[i]
    return q/n

# Calculate EI
def EI(d_yy, n):
    if (max(d_yy)-min(d_yy)) == 0:
        return 0
    ei = ((max(d_yy)-d_yy)/(max(d_yy)-min(d_yy))).sum()/n
    return ei

# ==================== Parameter Calculation Functions ====================

def calculate_parameters(x, y, ly, ry):
    """
    Calculate 1st-order fuzzy regression model parameters
    """
    # Convert inputs to NumPy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    ly = np.asarray(ly)
    ry = np.asarray(ry)

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
    a = (n*xy_sum - x_sum*y_sum)/(n*xx_sum - x_sum*x_sum)
    la = (n*xly_sum - x_sum*ly_sum)/(n*xx_sum - x_sum*x_sum)
    ra = (n*xry_sum - x_sum*ry_sum)/(n*xx_sum - x_sum*x_sum)
    a = round(a,3)
    la = round(la,3)
    ra = round(ra,3)

    # Calculate b, lb, rb
    b = (y_sum - a*x_sum)/n
    lb = (ly_sum - la*x_sum)/n
    rb = (ry_sum - ra*x_sum)/n
    b = round(b,3)
    lb = round(lb,3)
    rb = round(rb,3)

    return a, la, ra, b, lb, rb

# ==================== Main Processing Logic ====================

def process_dataset(filename):
    print(f"\n{'='*20} Processing file: {filename} {'='*20}")
    source_file = os.path.join(data_dir, filename)
    
    # Read data
    try:
        df = pd.read_csv(source_file)
    except FileNotFoundError:
        print(f"Error: File not found {source_file}")
        return

    # example_3_symmetric_triangular.csv format: x, y, spread
    # spread is the left/right width, so ly = ry = spread
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    spread = df.iloc[:, 2]
    ly = spread
    ry = spread

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
    y_p = round(y_p, 3)
    ly_p = round(ly_p, 3)
    ry_p = round(ry_p, 3)

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
    
    # EI_DC (Coppi) - Using triangular fuzzy number version
    d_Coppi = D_Coppi_tfn(y, ly, ry, y_p, ly_p, ry_p)
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

    # Calculate additional metrics
    print("Calculating additional metrics (SS, R^2)...")
    k = 1 # 1st-order model
    
    # SS_tfnDYK
    ss_dyk = SS_tfnDYK_005(y, ly, ry, y_p, ly_p, ry_p, k)
    ss_dyk = round(ss_dyk, 4)
    print(f"SS_tfnDYK: {ss_dyk}")
    
    # SS_tfnDDK
    ss_ddk = SS_tfnDDK_005(y, ly, ry, y_p, ly_p, ry_p, k)
    ss_ddk = round(ss_ddk, 4)
    print(f"SS_tfnDDK: {ss_ddk}")
    
    # R^2
    r2 = fit_degree_004tfn(y, ly, y_p, ly_p)
    r2 = round(r2, 4)
    print(f"R^2: {r2}")

    # Save EI results
    ei_df = pd.DataFrame({
        'Metric': ['EI_DC', 'EI_DH', 'EI_DL', 'EI_DB', 'SS_tfnDYK', 'SS_tfnDDK', 'R^2'],
        'Value': [ei_dc, ei_dh, ei_dl, ei_db, ss_dyk, ss_ddk, r2]
    })
    ei_path = os.path.join(results_dir, f'{base_name}_error_indices.csv')
    ei_df.to_csv(ei_path, index=False)
    print(f"Error indices saved to: {ei_path}")

def process_direct_test(pred_filename, true_filename):
    print(f"\n{'='*20} Direct testing file: {pred_filename} {'='*20}")
    pred_path = os.path.join(data_dir, pred_filename)
    true_path = os.path.join(data_dir, true_filename)
    
    # Read predicted values
    try:
        pred_df = pd.read_csv(pred_path)
    except FileNotFoundError:
        print(f"Error: File not found {pred_path}")
        return
        
    # Read actual values
    try:
        true_df = pd.read_csv(true_path)
    except FileNotFoundError:
        print(f"Error: File not found {true_path}")
        return

    # Extract data
    # Predicted values (example_3_M_z.csv: x, y, ly, ry)
    y_p = pred_df.iloc[:, 1]
    ly_p = pred_df.iloc[:, 2]
    ry_p = pred_df.iloc[:, 3]
    
    # Actual values (example_3_symmetric_triangular.csv: x, y, spread)
    # ly = ry = spread
    y = true_df.iloc[:, 1]
    ly = true_df.iloc[:, 2]
    ry = true_df.iloc[:, 2] # spread
    
    n = len(y)
    print(f"Number of data rows: {n}")
    
    # Calculate metrics
    print("Calculating error indices EI...")
    
    # EI_DC (Coppi)
    d_Coppi = D_Coppi_tfn(y, ly, ry, y_p, ly_p, ry_p)
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

    # Calculate additional metrics
    print("Calculating additional metrics (SS, R^2)...")
    k = 1 # 1st-order model
    
    # SS_tfnDYK
    ss_dyk = SS_tfnDYK_005(y, ly, ry, y_p, ly_p, ry_p, k)
    ss_dyk = round(ss_dyk, 4)
    print(f"SS_tfnDYK: {ss_dyk}")
    
    # SS_tfnDDK
    ss_ddk = SS_tfnDDK_005(y, ly, ry, y_p, ly_p, ry_p, k)
    ss_ddk = round(ss_ddk, 4)
    print(f"SS_tfnDDK: {ss_ddk}")
    
    # R^2
    r2 = fit_degree_004tfn(y, ly, y_p, ly_p)
    r2 = round(r2, 4)
    print(f"R^2: {r2}")

    # Save EI results
    base_name = os.path.splitext(pred_filename)[0]
    ei_df = pd.DataFrame({
        'Metric': ['EI_DC', 'EI_DH', 'EI_DL', 'EI_DB', 'SS_tfnDYK', 'SS_tfnDDK', 'R^2'],
        'Value': [ei_dc, ei_dh, ei_dl, ei_db, ss_dyk, ss_ddk, r2]
    })
    ei_path = os.path.join(results_dir, f'{base_name}_error_indices.csv')
    ei_df.to_csv(ei_path, index=False)
    print(f"Error indices saved to: {ei_path}")

def main():
    # Process example_3_symmetric_triangular.csv
    process_dataset('example_3_symmetric_triangular.csv')

    # Directly test example_3_M_z.csv, using example_3_symmetric_triangular.csv as actual values
    process_direct_test('example_3_M_z.csv', 'example_3_symmetric_triangular.csv')

if __name__ == "__main__":
    main()
