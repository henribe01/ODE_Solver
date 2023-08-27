def logistic_growth(x: float, t: float):
    k = 1
    K = 100
    return k * x * (1 - x / K)