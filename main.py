import matricesSimpyGeneration
import torch
import torch.nn.functional as F

class Winograd(object):
    def __init__(self, filter=None):
        super(Winograd, self).__init__()
        if filter is not None:
            self.filter = filter

    @staticmethod
    def __convert_sympy_to_torch_tensor(M, dtype, device, tol=1e-12):
        rows, cols = M.shape
        data = []
        complex_detected = False

        for i in range(rows):
            row = []
            for j in range(cols):
                # Avalia numericamente e converte para número complexo do Python
                z = complex(M[i, j].evalf())
                if abs(z.imag) > tol:
                    complex_detected = True
                # Guardamos sempre como float real aqui; se for "quase real", tomamos a parte real
                row.append(float(z.real))
            data.append(row)

        if complex_detected:
            raise ValueError(
                "As matrizes de transformação (AT/G/BT) possuem entradas complexas "
                "(parte imaginária significativa). Ajuste os pontos de interpolação "
                "ou o método de geração para obter matrizes reais, ou promova o fluxo "
                "todo para dtype complexo."
            )

        return torch.tensor(data, dtype=dtype, device=device)

    def forward(self, input, filter):
        numberInput, channelsInput, heightInput, widthInput = input.size()

        numberFilter, channelsFilter, heightFilter, widthFilter = filter.size() 

        assert heightFilter == widthFilter
        assert channelsFilter == channelsInput
        outputTileSize = 2 

        AT_sym,G_sym,BT_sym,f = matricesSimpyGeneration.constructTransformationMatrices(outputTileSize, heightFilter)

        self.A_T = self.__convert_sympy_to_torch_tensor(AT_sym, input.dtype, input.device)
        self.A = torch.transpose(self.A_T, 0, 1)
        self.B_T = self.__convert_sympy_to_torch_tensor(BT_sym, input.dtype, input.device)
        self.B = torch.transpose(self.B_T, 0, 1)
        self.G = self.__convert_sympy_to_torch_tensor(G_sym, input.dtype, input.device)
        self.G_T = torch.transpose(self.G, 0 ,1)

        entryBlockSize = outputTileSize + heightFilter - 1 

        assert heightInput >= entryBlockSize and widthInput >= entryBlockSize

        assert heightInput >= entryBlockSize and widthInput >= entryBlockSize, "Input menor que um tile Winograd."
        assert (heightInput - entryBlockSize) % outputTileSize == 0 and (widthInput - entryBlockSize) % outputTileSize == 0, \
            "Somente tiling perfeito (sem padding) é suportado: (heightInput-entryBlockSize) e (widthInput-entryBlockSize) devem ser múltiplos de outputTileSize"

        tilesPerDimH = (heightInput - entryBlockSize) // outputTileSize + 1
        tilesPerDimW = (widthInput - entryBlockSize) // outputTileSize + 1
        totalTiles = numberInput * tilesPerDimH * tilesPerDimW

        filterTransformed = torch.zeros((numberFilter, channelsInput, entryBlockSize, entryBlockSize), dtype=input.dtype, device=input.device)
        for k in range(numberFilter):
            for c in range(channelsInput):
                filterTransformed[k,c] = self.G @ filter[k, c] @ self.G_T

        tileIndex = 0
        tilesTransformed = torch.zeros((totalTiles, channelsInput, entryBlockSize, entryBlockSize), dtype=input.dtype, device=input.device)
        for n in range(numberInput):
            for tH in range(tilesPerDimH):
                for tW in range(tilesPerDimW):
                    beginBlockY = tH * outputTileSize
                    beginBlockX = tW * outputTileSize

                    patch = input[n, :, beginBlockY:beginBlockY+entryBlockSize, beginBlockX:beginBlockX+entryBlockSize]

                    for c in range(channelsInput):
                        tilesTransformed[tileIndex, c] = self.B_T @ patch[c] @ self.B
                    tileIndex += 1

        hadamardMatrice = torch.zeros((numberFilter, totalTiles, entryBlockSize, entryBlockSize), dtype=input.dtype, device=input.device)
        for i in range(numberFilter):
            for b in range(totalTiles):
                productMatrice = torch.zeros((entryBlockSize, entryBlockSize), dtype=input.dtype, device=input.device)

                for c in range(channelsInput):
                    productMatrice += filterTransformed[i, c] * tilesTransformed[b, c]
                
                hadamardMatrice[i, b] = productMatrice

        outputHeight = heightInput - heightFilter + 1
        outputWidth = widthInput - widthFilter + 1

        outputMatrice = torch.zeros((numberInput, numberFilter, outputHeight, outputWidth), dtype=input.dtype, device=input.device)
        tile = 0
        for i in range(numberInput):
            for tileHeight in range(tilesPerDimH):
                for tileWidth in range(tilesPerDimW):
                    beginBlockY = tileHeight * outputTileSize
                    beginBlockX = tileWidth * outputTileSize

                    for j in range(numberFilter):
                        block = self.A_T @ hadamardMatrice[j, tile] @ self.A
                        outputMatrice[i, j, beginBlockY:beginBlockY+outputTileSize, beginBlockX:beginBlockX+outputTileSize] = block
                    tile += 1

        return outputMatrice

    def conv_via_block_sum(self, x, w, block_size):
        """
        Convolução exata de kernel grande via soma de blocos NÃO sobrepostos.
        - Blocos t×t usam Winograd F(2,t) com padding mínimo p/ tiling perfeito.
        - Caudas (blocos menores que t ou retangulares) usam conv2d direta.
        """
        assert x.dim() == 4 and w.dim() == 4
        N, C, H, W = x.shape
        K, Cw, R, S = w.shape
        assert C == Cw and R == S, "Apenas kernels K×C×R×R suportados."
        t = block_size
        assert 1 <= t <= R

        outH = H - R + 1
        outW = W - S + 1
        assert outH > 0 and outW > 0, "Input menor que o kernel (modo 'valid')."

        y = torch.zeros((N, K, outH, outW), dtype=x.dtype, device=x.device)

        win_t = Winograd()    # para blocos exatamente t×t
        m = 2

        # Particiona o kernel em blocos NÃO sobrepostos (stride = t)
        for p in range(0, R, t):
            bh = min(t, R - p)   # altura do bloco (pode ser < t na borda)
            for q in range(0, S, t):
                bw = min(t, S - q)  # largura do bloco (pode ser < t na borda)

                w_block = w[:, :, p:p+bh, q:q+bw].contiguous()
                x_crop  = x[:, :, p:, q:]            # [N, C, H-p, W-q]
                Hp, Wq  = H - p, W - q

                if bh == t and bw == t:
                    # --- Winograd F(2,t) ---
                    entry = m + t - 1  # t+1
                    padH  = (m - ((Hp - entry) % m)) % m
                    padW  = (m - ((Wq - entry) % m)) % m
                    x_pad = F.pad(x_crop, (0, padW, 0, padH)) if (padH or padW) else x_crop
                    y_part = win_t.forward(x_pad, w_block)   # [N,K,(Hp+padH)-t+1,(Wq+padW)-t+1]
                    valid_h = Hp - t + 1
                    valid_w = Wq - t + 1
                    y_part = y_part[:, :, :valid_h, :valid_w]
                else:
                    # --- Caudas (blocos menores/retangulares) → conv2d direta (exata) ---
                    y_part = F.conv2d(x_crop, w_block, bias=None, stride=1, padding=0)
                    # Saída válida do parcial:
                    valid_h = Hp - bh + 1
                    valid_w = Wq - bw + 1
                    y_part = y_part[:, :, :valid_h, :valid_w]

                # Alinhar ao tamanho global e somar
                y[:, :, :outH, :outW] += y_part[:, :, :outH, :outW]

        return y

N, C, H, W = 1, 3, 32, 32
K, r = 4, 15
x = torch.randn(N, C, H, W, dtype=torch.double)
w = torch.randn(K, C, r, r, dtype=torch.double)

win = Winograd()
y_win = win.forward(x, w)
y_win_blocked = win.conv_via_block_sum(x, w, 3)
y_ref = F.conv2d(x, w, bias=None, stride=1, padding=0)

print("shapes:", y_win.shape, y_ref.shape)
print("max|diff|:", (y_win - y_ref).abs().max().item())
print("max|diff|:", (y_win_blocked - y_ref).abs().max().item())

# Entrada 32x32: ( Winograd raiz, sem dividir o fitlro em filtros 3x3 e convolucionar parcialmente )
# Filtro 3x3 -> 8.9e-15 
# Filtro 5x5 -> 9.0e-14
# Filtro 7x7 -> 1.4e-11
# Filtro 9x9 -> 8.81e-9
# Filtro 11x11 -> 3.02e-5
# Filtro 13x13 -> 0.2
# Filtro 15x15 -> 1434.62