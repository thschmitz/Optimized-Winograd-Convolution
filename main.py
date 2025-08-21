import torch

class Winograd(object):
    def __init__(self, m=2, r=3, points=None, canonical_F23=True, dtype=torch.double, device=None, filter_value=None):
        """
        m: tamanho do bloco de saída (ex.: 2)
        r: tamanho do kernel (ex.: 3)
        points: pontos de interpolação (opcional)
        canonical_F23: se (m,r)==(2,3), usa {0,1,-1,∞} e normaliza G
        """
        super(Winograd, self).__init__()
        self.m = m
        self.r = r
        self.a = m + r - 1
        self.dtype = dtype
        self.device = device
        if filter_value is not None:
            self.filter = filter_value

        # Gera B,G,A via CRT
        B, G, A, a = self.winograd_matrices_1d(
            m, r,
            points=points,
            canonical_F23=(canonical_F23 and (m, r) == (2, 3)),
            dtype=dtype,
            device=device
        )
        # Armazena (e transpostas)
        self.B = B
        self.B_T = B.transpose(0, 1)
        self.G = G
        self.G_T = G.transpose(0, 1)
        self.A = A
        self.A_T = A.transpose(0, 1)

        self.pontosUsados = a

    def _pick_points(self, t):
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

    def _vandermonde(self, a, deg):  # retorna [len(a), deg]
        # V[i,j] = a_i^j  (j=0..deg-1)
        powers = [a**j for j in range(deg)]
        return torch.stack(powers, dim=1)

    def winograd_matrices_1d(self, m, r, points=None, canonical_F23=False, dtype=torch.double, device=None):
        """
        Gera (B, G, A) 1D.
        - Se canonical_F23 e (m,r)==(2,3): retorna as matrizes canônicas que batem bit a bit com a convolução.
        - Caso contrário: gerador "experimental" à base de Vandermonde (pode não coincidir com a conv direta).
        Retorna: B (t×t), G (t×r), A (t×m) e os 'pontos' usados (tensor).
        """
        t = m + r - 1

        # ---- RAMO CANÔNICO: F(2,3) ----
        if canonical_F23 and (m, r) == (2, 3):
            B = torch.tensor([
                [ 1.,  0.,  0.,  0.],
                [ 0.,  1., -1.,  1.],
                [-1.,  1.,  1.,  0.],
                [ 0.,  0.,  0., -1.],
            ], dtype=dtype, device=device)
            G = torch.tensor([
                [1.,   0.,   0. ],
                [0.5,  0.5,  0.5],
                [0.5, -0.5,  0.5],
                [0.,   0.,   1. ],
            ], dtype=dtype, device=device)
            A = torch.tensor([
                [1.,  0.],
                [1.,  1.],
                [1., -1.],
                [0., -1.],
            ], dtype=dtype, device=device)
            pts = torch.tensor([0.0, 1.0, -1.0, float("inf")], dtype=dtype, device=device)
            return B, G, A, pts

        # ---- GERADOR EXPERIMENTAL (não garante igualdade exata) ----
        # pontos
        if points is None:
            a = self._pick_points(t).to(dtype=dtype, device=device)  # 0, 1, -1, 2, -2, ...
        else:
            a = torch.tensor(points, dtype=dtype, device=device)

        # Vandermonde simples (sem ∞)
        V_t_t = self._vandermonde(a, t).to(dtype=dtype, device=device)  # t x t
        V_t_r = self._vandermonde(a, r).to(dtype=dtype, device=device)  # t x r
        Vinv = torch.linalg.inv(V_t_t)

        # SELETOR CORRETO: graus r-1 .. r+m-2
        S = torch.zeros((m, t), dtype=dtype, device=device)
        S[torch.arange(m, device=device), torch.arange(m, device=device) + (r - 1)] = 1.0

        G = V_t_r                    # t x r  (avaliar kernel)
        B = V_t_t.T                  # t x t  (avaliar tile 1D em 2D via B^T d B)
        A = (S @ Vinv).T             # t x m  (interpolar e selecionar coeficientes centrais)

        return B, G, A, a



    def forward(self, input, filter):
        """
        Convolução via Winograd (genérico F(m,r)).
        input:  [N, C, H, W]
        filter: [K, C, r, r]
        return: [N, K, H-r+1, W-r+1]
        """
        m, r, a = self.m, self.r, self.a
        N, C, H, W = input.size()
        K, Cprime, rH, rW = filter.size()
        assert rH == rW == r
        assert C == Cprime
        assert H == W
        if not (H >= a and (H - a) % (r - 1) == 0):
            raise Exception("Only input for perfect tiling is supported (H>=a and (H-a) divisible by r-1).")

        # Reorganiza para [C, N, H, W] (como no teu código)
        input = torch.transpose(input, 0, 1)
        T = (W - a) // (r - 1) + 1
        P = N * T * T

        # U: [K, C, a, a], V: [C, P, a, a]
        U = torch.zeros(K, C, a, a, dtype=self.dtype, device=self.device)
        V = torch.zeros(C, P, a, a, dtype=self.dtype, device=self.device)

        # 1) Filtro: U = G g G^T
        for k in range(K):
            for c in range(C):
                U[k, c] = self.G @ filter[k, c].to(self.dtype).to(self.device) @ self.G_T

        # 2) Dados: V = B^T d B
        for n in range(N):
            for tH in range(T):
                for tW in range(T):
                    for c in range(C):
                        b = n * (T * T) + tH * T + tW
                        vH = tH * (r - 1)
                        vW = tW * (r - 1)
                        patch = input[c, n, vH:vH + a, vW:vW + a].to(self.dtype).to(self.device)
                        V[c, b] = self.B_T @ patch @ self.B

        # 3) Hadamard + soma em C: M[k,b] = sum_c U[k,c] ⊙ V[c,b]
        M = torch.zeros(K, P, a, a, dtype=self.dtype, device=self.device)
        for k in range(K):
            for b in range(P):
                acc = torch.zeros(a, a, dtype=self.dtype, device=self.device)
                for c in range(C):
                    acc += U[k, c] * V[c, b]
                M[k, b] = acc

        # 4) Inversa por tile: Y_block = A^T M A  (2D via separabilidade)
        out_size = H - r + 1
        Y = torch.zeros(K, N, out_size, out_size, dtype=self.dtype, device=self.device)
        for k in range(K):
            for n in range(N):
                for tH in range(T):
                    for tW in range(T):
                        b = n * (T * T) + tH * T + tW
                        oH, oW = tH * m, tW * m
                        Y[k, n, oH:oH + m, oW:oW + m] = self.A_T @ M[k, b] @ self.A

        return torch.transpose(Y, 0, 1)  # [N, K, H-r+1, W-r+1]

    # Versões unitárias (tiles/filtros únicos)
    def winograd_F_m_r_tile(self, tile4x4, filt3x3):
        U = self.G @ filt3x3 @ self.G_T
        V = self.B_T @ tile4x4 @ self.B
        return self.A_T @ (U * V) @ self.A


def _naive_conv2d_same_stride1(x, w):
    # x: [N,C,H,W], w:[K,C,r,r]; saída: [N,K,H-r+1,W-r+1]
    N,C,H,W = x.shape
    K,C2,r,_ = w.shape
    assert C==C2
    outH = H - r + 1
    outW = W - r + 1
    y = torch.zeros((N,K, outH, outW), dtype=x.dtype, device=x.device)
    for n in range(N):
        for k in range(K):
            acc = torch.zeros((outH,outW), dtype=x.dtype, device=x.device)
            for c in range(C):
                for i in range(outH):
                    for j in range(outW):
                        patch = x[n,c, i:i+r, j:j+r]
                        acc[i,j] += torch.sum(patch * w[k,c])
            y[n,k] = acc
    return y

# ---- teste winograd vs. convolução direta ----
torch.manual_seed(0)
N,C,K,H = 1, 3, 2, 8
r = 3
x = torch.randn(N,C,H,H, dtype=torch.double)
w = torch.randn(K,C,r,r, dtype=torch.double)

Wg = Winograd(m=2, r=3, canonical_F23=True, dtype=torch.double)
y_win = Wg.forward(x, w)
y_ref = _naive_conv2d_same_stride1(x, w)

print("max|diff|:", (y_win - y_ref).abs().max().item())