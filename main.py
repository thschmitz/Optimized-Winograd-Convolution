import matricesSimpyGeneration
import torch

class Winograd(object):

    def __init__(self, filter=None):
        super(Winograd, self).__init__()


        if filter is not None:
            self.filter = filter

    def forward(self, input, filter):
        numberInput, channelsInput, heightInput, widthInput = input.size()

        numberFilter, channelsFilter, heightFilter, widthFilter = filter.size() 

        assert heightInput == widthInput
        assert heightFilter == widthFilter

        outputSize = heightInput - heightFilter + 1

        AT,G,BT,f = matricesSimpyGeneration.constructTransformationMatrices(outputSize, heightFilter)

        self.A_T = AT
        self.A = torch.transpose(AT, 0, 1)
        self.B_T = BT
        self.B = torch.transpose(BT, 0, 1)
        self.G = G
        self.G_T = torch.transpose(G, 0 ,1)

        entryBlockSize = outputSize + heightFilter - 1

        assert heightInput >= entryBlockSize and widthInput >= entryBlockSize, "Input menor que um tile Winograd."
        assert (heightInput - entryBlockSize) % outputSize == 0 and (widthInput - entryBlockSize) % outputSize == 0, \
            "Somente tiling perfeito (sem padding) é suportado: (H-a) e (W-a) devem ser múltiplos de m."
        
        tilesPerDimH = (heightInput - entryBlockSize) // outputSize + 1
        tilesPerDimW = (widthInput - entryBlockSize) // outputSize + 1
        totalTiles = numberInput * tilesPerDimH * tilesPerDimW

        filterTransformed = torch.zeros((numberFilter, channelsInput, entryBlockSize, entryBlockSize), dtype=input.dtype, device=input.device)
        for k in range(numberFilter):
            for c in range(channelsFilter):
                filterTransformed[k,c] = self.G @ filter[k, c] @ self.G_T

        tileIndex = 0
        tilesTransformed = torch.zeros((totalTiles, channelsInput, entryBlockSize, entryBlockSize), dtype=input.dtype, device=input.device)
        for n in range(numberInput):
            for tH in range(tilesPerDimH):
                for tW in range(tilesPerDimW):
                    beginBlockY = tH * outputSize
                    beginBlockX = tW * outputSize

                    patch = input[n, :, beginBlockY:beginBlockY+entryBlockSize, beginBlockX:beginBlockX+entryBlockSize]

                    for c in range(channelsInput):
                        tilesTransformed[c, tileIndex] = self.B_T @ patch[c] @ self.B
                    tileIndex += 1

        hamadardMatrice = torch.zeros((numberFilter, totalTiles, entryBlockSize, entryBlockSize), dtype=input.dtype, device=input.device)
        for i in range(numberFilter):
            for b in range(totalTiles):
                productMatrice = torch.zeros((entryBlockSize, entryBlockSize), dtype=input.dtype, device=input.device)

                for c in range(channelsInput):
                    productMatrice += filterTransformed[i, c] * tilesTransformed[c, b]
                
                hamadardMatrice[i, b] = productMatrice