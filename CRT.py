import torch

def _pick_points(t):
    # sequência 0, 1, -1, 2, -2, 3, -3, ...
    pts = []
    k = 0
    while len(pts) < t:
        if k == 0:
            pts.append(0.)
        else:
            pts.append(float(k))
            if len(pts) < t:
                pts.append(float(-k))
        k += 1
    return torch.tensor(pts, dtype=torch.double)

def _vandermonde(a, deg):  # retorna [len(a), deg]
    # V[i,j] = a_i^j  (j=0..deg-1)
    powers = [a**j for j in range(deg)]
    return torch.stack(powers, dim=1)

def winograd_matrices_1d(m, r, points=None, dtype=torch.double):
    """
    Gera (B, G, A) 1D para F(m, r) usando pontos de interpolação.
    Retorna B (t×t), G (t×r), A (t×m) — compatíveis com Y = A^T[(Gg)⊙(B^Td)].
    """
    t = m + r - 1
    a = _pick_points(t) if points is None else torch.tensor(points, dtype=torch.double)
    assert len(a) == t and torch.unique(a).numel() == t, "precisa de t pontos distintos"

    V_t_t = _vandermonde(a, t).to(dtype)     # t x t
    V_t_r = _vandermonde(a, r).to(dtype)     # t x r

    Vinv = torch.linalg.inv(V_t_t)           # t x t
    S = torch.zeros((m, t), dtype=dtype)     # m x t
    S[torch.arange(m), torch.arange(m)] = 1  # seleciona coef 0..m-1

    G = V_t_r                                 # t x r
    B = V_t_t.T                               # t x t  (pois B^T = V)
    A_T = S @ Vinv                            # m x t
    A = A_T.T                                 # t x m

    return B, G, A, a  # também retorna os pontos usados

# --- Exemplo: reproduzir F(2,3) ---
B, G, A, pts = winograd_matrices_1d(m=2, r=3, points=[0, 1, -1, 2])
print("Pontos:", pts.tolist())
print("B=\n", B)
print("G=\n", G)
print("A=\n", A)
