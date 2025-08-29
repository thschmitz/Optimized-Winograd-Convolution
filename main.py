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

        assert heightInput == widthInput
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

        assert heightInput >= entryBlockSize and widthInput >= entryBlockSize, "Input menor que um tile Winograd."
        assert (heightInput - entryBlockSize) % outputTileSize == 0 and (widthInput - entryBlockSize) % outputTileSize == 0, \
            "Somente tiling perfeito (sem padding) é suportado: (H-a) e (W-a) devem ser múltiplos de m."
        
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

N, C, H, W = 1, 3, 32, 32
K, r = 4, 3
x = torch.randn(N, C, H, W, dtype=torch.double)
w = torch.randn(K, C, r, r, dtype=torch.double)

win = Winograd()
y_win = win.forward(x, w)
y_ref = F.conv2d(x, w, bias=None, stride=1, padding=0)

print("shapes:", y_win.shape, y_ref.shape)
print("max|diff|:", (y_win - y_ref).abs().max().item())
