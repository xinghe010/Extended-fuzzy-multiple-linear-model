import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import nnls
from sklearn.cluster import KMeans
import os

# Set file paths
data_dir = os.path.join(os.path.dirname(__file__), 'data')
results_dir = os.path.join(os.path.dirname(__file__), 'results')

# Ensure results directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ==================== RBF Model Functions ====================
def rbf_design(X, centers, sigmas):
    # X: (n,p), centers: (M,p), sigmas: (M,)
    n = X.shape[0]
    M = centers.shape[0]
    Phi = np.zeros((n, M))
    for m in range(M):
        diff = X - centers[m]
        Phi[:, m] = np.exp(-np.sum(diff**2, axis=1) / (2.0 * sigmas[m]**2))
    return Phi

def choose_centers_sigmas(X, M=10, method="kmeans"):
    if method == "kmeans":
        km = KMeans(n_clusters=M, random_state=0, n_init=10).fit(X)
        centers = km.cluster_centers_
    else:
        # use data points directly
        idx = np.random.choice(len(X), size=M, replace=False)
        centers = X[idx]
    # heuristic width: median pairwise distance
    dists = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dists.append(np.linalg.norm(centers[i]-centers[j]))
    med = np.median(dists) if len(dists)>0 else 1.0
    sigmas = np.full(M, med)
    return centers, sigmas

def ridge_fit(Phi, t, lam=1e-3):
    # closed-form ridge: (Phi^T Phi + lam I)^{-1} Phi^T t
    M = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(M)
    b = Phi.T @ t
    return np.linalg.solve(A, b)

def hesamian_rbf_fit_predict(X_train, y, l, r, X_test=None, M=10, lam=1e-3):
    if X_test is None:
        X_test = X_train

    centers, sigmas = choose_centers_sigmas(X_train, M=M, method="kmeans")
    Phi_train = rbf_design(X_train, centers, sigmas)
    Phi_test  = rbf_design(X_test,  centers, sigmas)

    wy = ridge_fit(Phi_train, y, lam)
    wl = ridge_fit(Phi_train, l, lam)
    wr = ridge_fit(Phi_train, r, lam)

    yhat = Phi_test @ wy
    lhat = Phi_test @ wl
    rhat = Phi_test @ wr

    # feasibility projection
    lhat = np.maximum(lhat, 0.0)
    rhat = np.maximum(rhat, 0.0)

    return yhat, lhat, rhat, (wy, wl, wr), (centers, sigmas)

# ==================== Distance Function Definitions ====================

# Coppi(2006) distance function - Calculate triangular fuzzy number - D_Coppi_tfn
def D_Coppi_tfn(a,la,ra,b,lb,rb):
    d_Coppi_tfn = (a-b)**2 + (la-lb)**2/2 + (ra-rb)**2/6
    return d_Coppi_tfn

# Hassanpour(2010) distance function - D_Hassanpour
def D_Hassanpour(a,la,ra,b,lb,rb):
    d_Hassanpour = (abs(a-b) + abs(la-lb) + abs(ra-rb))**2
    return d_Hassanpour

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

# Calculate EI
def EI(d_yy, n):
    if (max(d_yy)-min(d_yy)) == 0:
        return 0
    ei = ((max(d_yy)-d_yy)/(max(d_yy)-min(d_yy))).sum()/n
    return ei

# ==================== Model Calculation Functions ====================

def fuzzy_six_linear_regression(x1, x2, x3, x4, x5, x6, y, ly, ry):
    """
    Solve parameters for fuzzy 6th-order linear regression model and calculate model estimates.
    
    Parameters:
    x1, x2, x3, x4, x5, x6 (array-like): Independent variable arrays
    y (array-like): Dependent variable (center value) array
    ly (array-like): Dependent variable lower bound array
    ry (array-like): Dependent variable upper bound array
    
    Returns:
    tuple: A tuple containing (a, la, ra, b, lb, rb, y_p, ly_p, ry_p), representing regression coefficients, intercept, and predicted values respectively
    """
    
    n = len(y)  # Number of samples
    
    # Construct design matrix X and dependent variable vector Y
    X = np.column_stack([x1, x2, x3, x4, x5, x6])
    Y = y
    LY = ly
    RY = ry
    
    # Calculate covariance matrix C_XX and cross-product matrix C_XY
    C_XX = np.dot(X.T, X) / n - np.outer(np.mean(X, axis=0), np.mean(X, axis=0))
    C_XY = np.dot(X.T, Y) / n - np.mean(X, axis=0) * np.mean(Y)
    C_XLy = np.dot(X.T, LY) / n - np.mean(X, axis=0) * np.mean(LY)
    C_XRy = np.dot(X.T, RY) / n - np.mean(X, axis=0) * np.mean(RY)
    
    # Calculate inverse matrix
    try:
        inverse_C_XX = np.linalg.inv(C_XX)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular, please check input data")
    
    # Calculate regression coefficients a (Center - OLS)
    a = np.dot(C_XY, inverse_C_XX).flatten()
    
    # Round coefficients to 4 decimal places
    a = np.round(a, 4)

    # Calculate intercept term b
    b = np.mean(Y) - np.dot(np.mean(X, axis=0), a)
    b = round(b, 4)

    # Calculate spread coefficients using NNLS
    # Construct augmented design matrix Z = [1, X]
    Z = np.column_stack([np.ones(n), X])

    # Left spread
    beta_l, _ = nnls(Z, LY)
    lb = beta_l[0]
    la = beta_l[1:]

    # Right spread
    beta_r, _ = nnls(Z, RY)
    rb = beta_r[0]
    ra = beta_r[1:]

    # Round spread coefficients to 4 decimal places
    la = np.round(la, 4)
    ra = np.round(ra, 4)
    lb = round(lb, 4)
    rb = round(rb, 4)
    
    # Calculate model estimates
    y_p = np.dot(X, a) + b
    ly_p = np.dot(X, la) + lb
    ry_p = np.dot(X, ra) + rb
    
    # Convert estimates to integers
    y_p = y_p.astype(int)
    ly_p = ly_p.astype(int)
    ry_p = ry_p.astype(int)
    
    return a, la, ra, b, lb, rb, y_p, ly_p, ry_p

def calculate_mm_model(x1, x2, x3, x4, x5, x6):
    """
    Calculate MM model estimates using hardcoded parameters.
    """
    # MM model parameters (hardcoded)
    mm_b, mm_lb, mm_rb = -580.3821, 2.7140, 2.7732
    mm_a = np.array([7.8695, -5.9958, -57.7137, -0.7828, 13.8969, -3.1045])
    mm_la = np.array([0.0181, 0.0124, 0.0313, -0.0016, 0.0189, 0.0072])
    mm_ra = np.array([0.0109, -0.0004, -0.0037, -0.0012, 0.0241, 0.0081])
    
    # Calculate MM model estimates
    # y_p = sum(xi * ai) + b
    mm_y_p = mm_b + x1*mm_a[0] + x2*mm_a[1] + x3*mm_a[2] + x4*mm_a[3] + x5*mm_a[4] + x6*mm_a[5]
    mm_ly_p = mm_lb + x1*mm_la[0] + x2*mm_la[1] + x3*mm_la[2] + x4*mm_la[3] + x5*mm_la[4] + x6*mm_la[5]
    mm_ry_p = mm_rb + x1*mm_ra[0] + x2*mm_ra[1] + x3*mm_ra[2] + x4*mm_ra[3] + x5*mm_ra[4] + x6*mm_ra[5]
        
    # Exponential transformation (refer to notebook logic)
    mm_ly_p = np.exp(mm_ly_p)
    mm_ry_p = np.exp(mm_ry_p)
    
    # Round to integer
    mm_y_p = mm_y_p.astype(int)
    mm_ly_p = mm_ly_p.astype(int)
    mm_ry_p = mm_ry_p.astype(int)
    
    return mm_y_p, mm_ly_p, mm_ry_p

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

    # shanghai_zhou2018.csv format: i, x1, x2, x3, x4, x5, x6, y, l, r
    # Assume l, r correspond to ly, ry
    x1 = df.iloc[:, 1]
    x2 = df.iloc[:, 2]
    x3 = df.iloc[:, 3]
    x4 = df.iloc[:, 4]
    x5 = df.iloc[:, 5]
    x6 = df.iloc[:, 6]
    
    y = df.iloc[:, 7]
    ly = df.iloc[:, 8]
    ry = df.iloc[:, 9]

    n = len(y)
    print(f"Number of data rows: {n}")
    
    base_name = os.path.splitext(filename)[0]

    # ==================== RBF Model Calculation ====================
    print(f"\n{'='*20} RBF Model Calculation {'='*20}")
    # Construct X matrix for RBF
    X_rbf = np.column_stack([x1, x2, x3, x4, x5, x6])
    
    # RBF Calculation
    # Note: M=10 centers as default
    rbf_y_p, rbf_l_p, rbf_r_p, (wy, wl, wr), (centers, sigmas) = hesamian_rbf_fit_predict(X_rbf, y, ly, ry, M=10)
    
    # Save RBF Parameters
    rbf_params_df = pd.DataFrame(centers, columns=[f'Center_Dim_{i}' for i in range(centers.shape[1])])
    rbf_params_df['Sigma'] = sigmas
    rbf_params_df['Weight_y'] = wy
    rbf_params_df['Weight_l'] = wl
    rbf_params_df['Weight_r'] = wr
    
    rbf_params_path = os.path.join(results_dir, f'{base_name}_RBF_parameters.csv')
    rbf_params_df.to_csv(rbf_params_path, index=False)
    print(f"RBF Parameters saved to: {rbf_params_path}")
    
    # Save RBF Estimates
    rbf_estimates_df = pd.DataFrame({
        'y_p': rbf_y_p,
        'ly_p': rbf_l_p,
        'ry_p': rbf_r_p
    })
    rbf_estimates_path = os.path.join(results_dir, f'{base_name}_RBF_estimates.csv')
    rbf_estimates_df.to_csv(rbf_estimates_path, index=False)
    print(f"RBF Estimates saved to: {rbf_estimates_path}")
    
    # Calculate RBF Metrics
    print("Calculating RBF model error indices EI and SS...")
    rbf_d_Coppi = D_Coppi_tfn(y, ly, ry, rbf_y_p, rbf_l_p, rbf_r_p)
    rbf_ei_dc = round(EI(rbf_d_Coppi, n), 4)
    print(f"RBF_EI_DC (Coppi): {rbf_ei_dc}")

    rbf_d_Hassanpour = D_Hassanpour(y, ly, ry, rbf_y_p, rbf_l_p, rbf_r_p)
    rbf_ei_dh = round(EI(rbf_d_Hassanpour, n), 4)
    print(f"RBF_EI_DH (Hassanpour): {rbf_ei_dh}")

    k_rbf = X_rbf.shape[1] # k=6
    rbf_ss_dyk = SS_tfnDYK_005(y, ly, ry, rbf_y_p, rbf_l_p, rbf_r_p, k_rbf)
    rbf_ss_dyk = round(rbf_ss_dyk, 4)
    print(f"RBF_SS_DYK: {rbf_ss_dyk}")

    rbf_ss_ddk = SS_tfnDDK_005(y, ly, ry, rbf_y_p, rbf_l_p, rbf_r_p, k_rbf)
    rbf_ss_ddk = round(rbf_ss_ddk, 4)
    print(f"RBF_SS_DDK: {rbf_ss_ddk}")
    
    # Save RBF Metrics
    rbf_metrics_df = pd.DataFrame({
        'Metric': ['EI_DC', 'EI_DH', 'SS_DYK', 'SS_DDK'],
        'Value': [rbf_ei_dc, rbf_ei_dh, rbf_ss_dyk, rbf_ss_ddk]
    })
    rbf_metrics_path = os.path.join(results_dir, f'{base_name}_RBF_error_indices.csv')
    rbf_metrics_df.to_csv(rbf_metrics_path, index=False)
    print(f"RBF error indices saved to: {rbf_metrics_path}")

    # 1. Calculate 6th-order model parameters
    a, la, ra, b, lb, rb, y_p, ly_p, ry_p = fuzzy_six_linear_regression(x1, x2, x3, x4, x5, x6, y, ly, ry)
    print("Parameter calculation results:")
    print(f"a={a}, la={la}, ra={ra}, b={b}, lb={lb}, rb={rb}")

    # Save parameters to results folder
    base_name = os.path.splitext(filename)[0]
    
    # Construct parameter DataFrame
    params_data = {
        'Parameter': [f'x{i+1}' for i in range(6)] + ['Intercept'],
        'a': list(a) + [b],
        'la': list(la) + [lb],
        'ra': list(ra) + [rb]
    }
    params_df = pd.DataFrame(params_data)
    params_path = os.path.join(results_dir, f'{base_name}_parameters.csv')
    params_df.to_csv(params_path, index=False)
    print(f"Parameters saved to: {params_path}")

    # 2. Calculate estimates for the 6th-order model
    print("Calculating estimates...")
    # Estimates have been calculated and returned in fuzzy_six_linear_regression

    # Save estimates
    estimates_df = pd.DataFrame({
        'y_p': y_p,
        'ly_p': ly_p,
        'ry_p': ry_p
    })
    estimates_path = os.path.join(results_dir, f'{base_name}_estimates.csv')
    estimates_df.to_csv(estimates_path, index=False)
    print(f"Estimates saved to: {estimates_path}")

    # 3. Calculate EI_DC, EI_DH, SS_DYK, SS_DDK between estimates and actual values
    print("Calculating error indices EI and SS...")
    
    # EI_DC (Coppi)
    d_Coppi = D_Coppi_tfn(y, ly, ry, y_p, ly_p, ry_p)
    ei_dc = round(EI(d_Coppi, n), 4)
    print(f"EI_DC (Coppi): {ei_dc}")

    # EI_DH (Hassanpour)
    d_Hassanpour = D_Hassanpour(y, ly, ry, y_p, ly_p, ry_p)
    ei_dh = round(EI(d_Hassanpour, n), 4)
    print(f"EI_DH (Hassanpour): {ei_dh}")

    # SS_DYK (k=6)
    k = 6
    ss_dyk = SS_tfnDYK_005(y, ly, ry, y_p, ly_p, ry_p, k)
    ss_dyk = round(ss_dyk, 4)
    print(f"SS_DYK: {ss_dyk}")
    
    # SS_DDK (k=6)
    ss_ddk = SS_tfnDDK_005(y, ly, ry, y_p, ly_p, ry_p, k)
    ss_ddk = round(ss_ddk, 4)
    print(f"SS_DDK: {ss_ddk}")

    # Save metric results
    metrics_df = pd.DataFrame({
        'Metric': ['EI_DC', 'EI_DH', 'SS_DYK', 'SS_DDK'],
        'Value': [ei_dc, ei_dh, ss_dyk, ss_ddk]
    })
    metrics_path = os.path.join(results_dir, f'{base_name}_error_indices.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Error indices saved to: {metrics_path}")

    # ==================== MM Model Test ====================
    print(f"\n{'='*20} Testing MM Model {'='*20}")
    
    # Calculate MM model estimates
    print("Calculating MM model estimates...")
    mm_y_p, mm_ly_p, mm_ry_p = calculate_mm_model(x1, x2, x3, x4, x5, x6)
        
    # Save MM model estimates
    mm_estimates_df = pd.DataFrame({
        'y_p': mm_y_p,
        'ly_p': mm_ly_p,
        'ry_p': mm_ry_p
    })
    mm_estimates_path = os.path.join(results_dir, f'{base_name}_MM_estimates.csv')
    mm_estimates_df.to_csv(mm_estimates_path, index=False)
    print(f"MM model estimates saved to: {mm_estimates_path}")
    
    # Calculate MM model error indices
    print("Calculating MM model error indices EI and SS...")
    
    # EI_DC (Coppi)
    mm_d_Coppi = D_Coppi_tfn(y, ly, ry, mm_y_p, mm_ly_p, mm_ry_p)
    mm_ei_dc = round(EI(mm_d_Coppi, n), 4)
    print(f"MM_EI_DC (Coppi): {mm_ei_dc}")

    # EI_DH (Hassanpour)
    mm_d_Hassanpour = D_Hassanpour(y, ly, ry, mm_y_p, mm_ly_p, mm_ry_p)
    mm_ei_dh = round(EI(mm_d_Hassanpour, n), 4)
    print(f"MM_EI_DH (Hassanpour): {mm_ei_dh}")

    # SS_DYK (k=6)
    mm_ss_dyk = SS_tfnDYK_005(y, ly, ry, mm_y_p, mm_ly_p, mm_ry_p, k)
    mm_ss_dyk = round(mm_ss_dyk, 4)
    print(f"MM_SS_DYK: {mm_ss_dyk}")
    
    # SS_DDK (k=6)
    mm_ss_ddk = SS_tfnDDK_005(y, ly, ry, mm_y_p, mm_ly_p, mm_ry_p, k)
    mm_ss_ddk = round(mm_ss_ddk, 4)
    print(f"MM_SS_DDK: {mm_ss_ddk}")
    
    # Save MM model metric results
    mm_metrics_df = pd.DataFrame({
        'Metric': ['EI_DC', 'EI_DH', 'SS_DYK', 'SS_DDK'],
        'Value': [mm_ei_dc, mm_ei_dh, mm_ss_dyk, mm_ss_ddk]
    })
    mm_metrics_path = os.path.join(results_dir, f'{base_name}_MM_error_indices.csv')
    mm_metrics_df.to_csv(mm_metrics_path, index=False)
    print(f"MM model error indices saved to: {mm_metrics_path}")

def main():
    process_dataset('shanghai_zhou2018.csv')

if __name__ == "__main__":
    main()
