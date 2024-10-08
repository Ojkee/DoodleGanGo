package optimizers

import (
	"slices"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
	"DoodleGan/layers"
)

type SGD struct {
	learningRate float64

	momentumMechanism
	lastConvOutputSize conv.MatSize
}

func NewSGD(learningRate, momentum float64) SGD {
	checkValidLearningRate(&learningRate, "NewSDG")
	checkValidMomentum(&momentum, "NewSGD")
	return SGD{
		learningRate:      learningRate,
		momentumMechanism: newMomentumMechanism(momentum),
	}
}

func (opt *SGD) PreTrainInit(
	lastConvOutputSize [2]int,
	convLayers *[]conv.ConvLayer,
	denseLayers *[]layers.Layer,
) {
	opt.lastConvOutputSize = *conv.NewMatSize(
		lastConvOutputSize[0],
		lastConvOutputSize[1],
	)
	if opt.momentum != 0.0 {
		opt.initMomentumMechanizm(convLayers, denseLayers)
	}
}

func (opt *SGD) BackwardDenseLayers(denses *[]layers.Layer, loss *mat.VecDense) *mat.VecDense {
	grads := *loss
	for i, denseLayer := range slices.Backward(*denses) {
		grads = *denseLayer.Backward(&grads)
		if trainableLayer, ok := denseLayer.(layers.LayerTrainable); ok {
			if opt.momentum == 0.0 {
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					trainableLayer.GetOutWeightsGrads(),
					trainableLayer.GetOutBiasGrads(),
				)
			} else { // Non Nesterov
				weightGrads := trainableLayer.GetOutWeightsGrads()
				biasGrads := trainableLayer.GetOutBiasGrads()
				opt.momentumUpdateDense(i, weightGrads, biasGrads)
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					opt.getMomentumDenseWeights(i),
					opt.getMomentumDenseBias(i),
				)
			}
		}
	}
	return &grads
}

func (opt *SGD) BackwardConv2DLayers(convs2D *[]conv.ConvLayer, denseGrads *mat.VecDense) {
	gradsMat := functools.VecToMatSlice(
		denseGrads,
		opt.lastConvOutputSize.Height(),
		opt.lastConvOutputSize.Width(),
	)
	for i := len(*convs2D) - 1; i >= 0; i-- {
		gradsMat = *(*convs2D)[i].Backward(&gradsMat)
		if trainableLayer, ok := (*convs2D)[i].(conv.ConvLayerTrainable); ok {
			if opt.momentum == 0.0 {
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					trainableLayer.GetFilterGrads(),
					trainableLayer.GetBiasGrads(),
				)
			} else { // Non Nesterov
				filterGrads := trainableLayer.GetFilterGrads()
				biasGrads := trainableLayer.GetBiasGrads()
				opt.momentumUpdateConv(i, filterGrads, biasGrads)
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					opt.getMomentumConvWeigts(i),
					opt.getMomentumConvBias(i),
				)
			}
		}
	}
}
