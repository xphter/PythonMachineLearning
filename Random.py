from ImportNumpy import *;


'''
generate random positive-definite matrix.
'''
def randomPDM(N : int) -> np.ndarray:
    A = np.random.randn(N, N).astype(defaultDType);
    return A.T @ A + np.eye(N, dtype = defaultDType);
