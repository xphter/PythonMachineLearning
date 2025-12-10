from NN import *;


class _TemporalContrastiveLoss:
    def __init__(self):
        self._inputShape : Tuple[int, ...] = ();
        self._L, self._Y, self._S = np.empty(0), np.empty(0), np.empty(0);
    

    def forward(self, Z1 : np.ndarray, Z2 : np.ndarray) -> np.floating:
        self._inputShape = Z1.shape;

        N = Z1.shape[-2];
        if N == 1:
            return defaultDType(0);

        X = np.concatenate((Z1, Z2), axis = -2);
        A = X @ np.swapaxes(X, -2, -1);

        M = np.eye(2 * N);
        G = A * ~M.astype(np.bool);
        np.fill_diagonal(M, -np.inf);
        G += M;
        # G = np.tril(A, k = -1)[..., : -1] + np.triu(A, k = 1)[..., 1:];

        L, S = logSoftmax(G, returnSoftmax = True);
        indices = np.arange(N);
        Y = L[..., indices, indices + N] + L[..., indices + N, indices];

        self._X, self._S, self._Y = X, S, Y;

        return np.mean(Y) / -2;
    

    def backward(self, dy : np.floating) -> Tuple[np.ndarray, np.ndarray]:
        if self._inputShape[-2] == 1:
            return np.zeros(self._inputShape, dtype = defaultDType), np.zeros(self._inputShape, dtype = defaultDType);

        N = self._Y.shape[-1];
        dY = dy / (-2 * self._Y.size) * np.ones_like(self._Y);

        dL = np.zeros(self._Y.shape[: -1] + (2 * N, 2 * N), dtype = dY.dtype);
        indices = np.arange(N);
        dL[..., indices, indices + N] = dY;
        dL[..., indices + N, indices] = dY;

        dG = logSoftmaxGradient(self._S, dL);
        dA = dG; # the gradients on diagonal are always zero!
        dX = (dA + np.swapaxes(dA, -2, -1)) @ self._X;

        dZ1, dZ2 = np.split(dX, 2, axis = -2);

        return dZ1, dZ2;


class _InstanceContrastiveLoss(_TemporalContrastiveLoss):
    def forward(self, Z1: np.ndarray, Z2: np.ndarray) -> np.floating:
        return super().forward(np.swapaxes(Z1, -3, -2), np.swapaxes(Z2, -3, -2));
    

    def backward(self, dy: np.floating) -> Tuple[np.ndarray, np.ndarray]:
        dZ1, dZ2 = super().backward(dy);

        return np.swapaxes(dZ1, -3, -2), np.swapaxes(dZ2, -3, -2);


class _ContrastiveLoss:
    def __init__(self, instanceLossWeight : float, temporalLossWeight : Optional[float] = None):
        self._instanceLossWeight = max(0.0, min(1.0, instanceLossWeight));
        self._temporalLossWeight = temporalLossWeight if temporalLossWeight is not None else 1.0 - self._instanceLossWeight;

        self._zeroLoss = defaultDType(0);
        self._inputShape : Tuple[int, ...] = ();
        self._instanceLoss = _InstanceContrastiveLoss();
        self._temporalLoss = _TemporalContrastiveLoss();
    

    def __repr__(self) -> str:
        return self.__str__();
    

    def __str__(self) -> str:
        return f"_ContrastiveLoss({self._instanceLossWeight}, {self._temporalLossWeight})";


    def forward(self, Z1: np.ndarray, Z2: np.ndarray) -> np.floating:
        self._inputShape = Z1.shape;
        loss1 = self._instanceLoss.forward(Z1, Z2) if self._instanceLossWeight > 0 else self._zeroLoss;
        loss2 = self._temporalLoss.forward(Z1, Z2) if self._temporalLossWeight > 0 else self._zeroLoss;

        return self._instanceLossWeight * loss1 + self._temporalLossWeight * loss2;
    

    def backward(self, dy: np.floating) -> Tuple[np.ndarray, np.ndarray]:
        dZ1 = np.zeros(self._inputShape, dtype = defaultDType);
        dZ2 = np.zeros(self._inputShape, dtype = defaultDType);

        if self._instanceLossWeight > 0:
            dZ1_, dZ2_ = self._instanceLoss.backward(dy * self._instanceLossWeight);
            dZ1 += dZ1_;
            dZ2 += dZ2_;
        
        if self._temporalLossWeight > 0:
            dZ1_, dZ2_ = self._temporalLoss.backward(dy * self._temporalLossWeight);
            dZ1 += dZ1_;
            dZ2 += dZ2_;

        return dZ1, dZ2;


class _ConvolutionBlock(AggregateNetModule):
    def __init__(self, inputChannel : int, outputChannel : int, kernelSize : int, dilation : int, isFinal : bool = False):
        self._convLayer = SequentialContainer(
            GeluLayer(),
            Convolution1DLayer(inputChannel, outputChannel, kernelSize, padding = "same", dilation = dilation),
            GeluLayer(),
            Convolution1DLayer(outputChannel, outputChannel, kernelSize, padding = "same", dilation = dilation),
        );
        self._projector = Convolution1DLayer(inputChannel, outputChannel, 1) if inputChannel != outputChannel or isFinal else None;

        if self._projector is not None:
            super().__init__(self._convLayer, self._projector);
        else:
            super().__init__(self._convLayer);
        
        self._name = f"_ConvolutionBlock {inputChannel}*{outputChannel}";
    

    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        F, = self._convLayer.forward(X);

        if self._projector is not None:
            Y = self._projector.forward(X)[0] + F;
        else:
            Y = X + F;

        return Y, ;
    

    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX, = self._convLayer.backward(dY);

        if self._projector is not None:
            dX_, = self._projector.backward(dY);
        else:
            dX_ = dY;
        dX += dX_;
        
        return dX, ;


class _Encoder(AggregateNetModule):
    def __init__(self, intputSize : int, outputSize : int, hiddenSize : int, kernelSize : int = 3, blockNum : int = 10, latentDropout : float = 0.5, representationDropout : float = 0.1):
        outputChannels = [hiddenSize] * blockNum + [outputSize];
        self._latentDropout = latentDropout;

        self._fnn = AffineLayer(intputSize, hiddenSize);
        self._extractor = SequentialContainer(
            *(_ConvolutionBlock(hiddenSize if i == 0 else outputChannels[i - 1], outputChannels[i], kernelSize, dilation = 2 ** i, isFinal = (i == blockNum)) for i in range(len(outputChannels)))
        );
        self._dropout = DropoutLayer(dropoutRatio = representationDropout);

        super().__init__(self._fnn, self._extractor, self._dropout);

        self._temporalMask = np.empty(0);
        self._name = f"_Encoder {intputSize}*({hiddenSize}*{blockNum})*{outputSize}";
    

    def _getTemporalMask(self, X : np.ndarray) -> np.ndarray:
        return np.random.binomial(1, 1 - self._latentDropout, size = X.shape[: -1]).astype(np.bool);
    

    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        Z, = self._fnn.forward(X);

        if self.context.isTrainingMode:
            # timestamp masking
            temporalMask = ~self._getTemporalMask(Z);
            Z[temporalMask] = 0;

            self._temporalMask = temporalMask;

        Z = np.swapaxes(Z, -2, -1);
        Y, = self._extractor.forward(Z);
        Y = np.swapaxes(Y, -2, -1);

        return self._dropout.forward(Y);

        
    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dY, = self._dropout.backward(dout[0]);
        dY = np.swapaxes(dY, -2, -1);

        dZ, = self._extractor.backward(dY);
        dZ = np.swapaxes(dZ, -2, -1);
        dZ[self._temporalMask] = 0; # type: ignore

        dX, = self._fnn.backward(dZ);

        return dX, ;


class TS2VecLoss(NetLossBase):
    def __init__(self, instanceLossWeight : float = 0.5, minTemporalUnit : int = 0):
        super().__init__();

        self._instanceLossWeight = max(0.0, min(1.0, float(instanceLossWeight)));
        self._temporalLossWeight = 1- self._instanceLossWeight;
        self._minTemporalUnit : int = max(0, int(minTemporalUnit));

        self._d : int = 0;
        self._contrastiveLosses : List[_ContrastiveLoss] = [];
        self._maxPooling1DLayers : List[MaxPooling1DLayer] = [];
    

    @property
    def name(self) -> str:
        return "Hierarchical Contrastive Loss";


    def forward(self, *data: np.ndarray) -> float:
        Z1, Z2 = data[: 2];
        loss = defaultDType(0);

        d = int(math.log(Z1.shape[-2], 2));
        if self._d - 1 != d:
            self._d = d + 1;

            self._contrastiveLosses = [_ContrastiveLoss(self._instanceLossWeight, self._temporalLossWeight if i >= self._minTemporalUnit else 0.0) for i in range(d)];
            self._contrastiveLosses.append(_ContrastiveLoss(self._instanceLossWeight, 0.0));

            self._maxPooling1DLayers = [MaxPooling1DLayer(2) for _ in range(d)];
            for item in self._maxPooling1DLayers:
                item.context.isTrainingMode = True;
        
        i = 0;
        while Z1.shape[-2] > 1:
            loss += self._contrastiveLosses[i].forward(Z1, Z2);

            Z, = self._maxPooling1DLayers[i].forward(np.swapaxes(np.concatenate((Z1, Z2), axis = 0), -2, -1));
            Z1, Z2 = np.split(np.swapaxes(Z, -2, -1), 2, axis = 0);

            i += 1;
        loss += self._contrastiveLosses[i].forward(Z1, Z2);

        self._loss = float(loss / self._d);
        return self._loss;
        

    def backward(self) -> Tuple[np.ndarray, ...]:
        dy = defaultDType(1.0 / self._d);
        dZ1, dZ2 = self._contrastiveLosses[-1].backward(dy);

        for maxPooling1D, contrastiveLoss in zip(reversed(self._maxPooling1DLayers), reversed(self._contrastiveLosses[: -1])):
            dZ, = maxPooling1D.backward(np.swapaxes(np.concatenate((dZ1, dZ2), axis = 0), -2, -1));
            dZ1, dZ2 = np.split(np.swapaxes(dZ, -2, -1), 2, axis = 0);

            dZ1_, dZ2_ = contrastiveLoss.backward(dy);
            dZ1 += dZ1_;
            dZ2 += dZ2_;
        
        return dZ1, dZ2;


class TS2VecModel(NetModelBase):
    def __init__(self, intputSize : int, outputSize : int, hiddenSize : int, kernelSize : int = 3, blockNum : int = 10, latentDropout : float = 0.5, representationDropout : float = 0.1, minTemporalUnit : int = 0):
        self._minTemporalUnit = max(0, minTemporalUnit);
        self._encoder1 = _Encoder(intputSize, outputSize, hiddenSize, kernelSize = kernelSize, blockNum = blockNum, latentDropout = latentDropout, representationDropout = representationDropout);
        self._encoder2 = _Encoder(intputSize, outputSize, hiddenSize, kernelSize = kernelSize, blockNum = blockNum, latentDropout = latentDropout, representationDropout = representationDropout);
        self._encoder2.params = self._encoder1.params;

        super().__init__(self._encoder1, self._encoder2);

        self._croppingLength = 0;
        self._X, self._Z1, self._Z2 = np.empty(0), np.empty(0), np.empty(0);
        self._params = self._encoder1.params;
        self._name = f"TS2VecModel {intputSize}*({hiddenSize}*{blockNum})*{outputSize}";
    

    def _setParams(self, params: List[INetParamDefinition]):
        self._encoder1.params = params;
        self._encoder2.params = params;
    

    def _getSubsequence(self, X: np.ndarray, startIndex: np.ndarray, sequenceLength: int) -> np.ndarray:
        return X[np.arange(startIndex.shape[0])[:, None], startIndex[:, None] + np.arange(sequenceLength)];
    

    def getFinalTag(self, T: np.ndarray) -> Optional[np.ndarray]:
        return None;
    

    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        batchSize, sequenceLength = X.shape[: 2];

        # random cropping
        croppingLength = np.random.randint(2 ** (self._minTemporalUnit + 1), sequenceLength + 1, dtype = np.int32);
        a2 = np.random.randint(0, sequenceLength - croppingLength + 1, dtype = np.int32);
        b1 = a2 + croppingLength;
        a1 = np.random.randint(a2 + 1, dtype = np.int32);
        b2 = np.random.randint(b1, sequenceLength + 1, dtype = np.int32);
        offset = np.random.randint(-a1, sequenceLength - b2 + 1, size = batchSize, dtype = np.int32);
        
        V1 = self._getSubsequence(X, offset + a1, b1 - a1);
        V2 = self._getSubsequence(X, offset + a2, b2 - a2);

        Z1, = self._encoder1.forward(V1);
        Z2, = self._encoder2.forward(V2);

        self._croppingLength = croppingLength;
        self._X, self._Z1, self._Z2 = X, Z1, Z2;

        return Z1[:, -croppingLength:], Z2[:, : croppingLength];
    

    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dZ1_, dZ2_ = dout;

        dZ1, dZ2 = np.zeros_like(self._Z1), np.zeros_like(self._Z2);
        dZ1[:, -self._croppingLength:] = dZ1_;
        dZ2[:, : self._croppingLength] = dZ2_;

        self._encoder1.backward(dZ1);
        self._encoder2.backward(dZ2);
        
        return np.zeros_like(self._X), ;


    # pooling: max(MaxPooling1D), mean(AvgPooling1D)
    def encode(self, X : np.ndarray, pooling : Optional[str] = None) -> np.ndarray:
        self.context.isTrainingMode = False;

        Y, = self._encoder1.forward(X);

        if pooling is not None:
            poolingLayer : Optional[INetModule] = None;
            
            if pooling == "max":
                poolingLayer = MaxPooling1DLayer(Y.shape[-2]);
            elif pooling == "mean":
                poolingLayer = AvgPooling1DLayer(Y.shape[-2]);
            elif isinstance(pooling, INetModule):
                poolingLayer = pooling;
            else:
                raise ValueError(f"{pooling} is not a valid pooling method");

            if poolingLayer is not None:
                poolingLayer.context = self.context;

                Y, = poolingLayer.forward(np.swapaxes(Y, -2, -1));
                Y = np.squeeze(Y, axis = -1);

        return Y;
