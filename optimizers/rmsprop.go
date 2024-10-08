package optimizers

import (
	"slices"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
	"DoodleGan/layers"
)

type RMSProp struct {
	learningRate float64

	rhoSquareMechanism
	lastConvOutputSize conv.MatSize
}

func NewRMSProp(learningRate, rho, eps float64) RMSProp {
	checkValidLearningRate(&learningRate, "NewRMSProp")
	checkValidRho(&rho, "NewRMSProp")
	checkValidEps(&eps, "NewRMSProp")
	return RMSProp{
		learningRate:       learningRate,
		rhoSquareMechanism: newRhoSquareMechanism(rho, eps),
	}
}

func (opt *RMSProp) PreTrainInit(
	lastConvOutputSize [2]int,
	convLayers *[]conv.ConvLayer,
	denseLayers *[]layers.Layer,
) {
	opt.lastConvOutputSize = *conv.NewMatSize(
		lastConvOutputSize[0],
		lastConvOutputSize[1],
	)
	if opt.rho != 0.0 {
		opt.initRhoMechanizm(convLayers, denseLayers)
	}
}

func (opt *RMSProp) BackwardDenseLayers(denses *[]layers.Layer, loss *mat.VecDense) *mat.VecDense {
	grads := *loss
	for i, denseLayer := range slices.Backward(*denses) {
		grads = *denseLayer.Backward(&grads)
		if trainableLayer, ok := denseLayer.(layers.LayerTrainable); ok {
			if opt.rho == 0.0 {
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					opt.zeroRhoActivateDense(trainableLayer.GetOutWeightsGrads()),
					opt.zeroRhoActivateVec(trainableLayer.GetOutBiasGrads()),
				)
			} else {
				dw2, db2 := opt.squareDenseLayerGrads(
					trainableLayer.GetOutWeightsGrads(),
					trainableLayer.GetOutBiasGrads(),
				)
				opt.rhoUpdateDense(i, &dw2, &db2)
				scaledGrads := opt.gradsScaleSquaredDense(
					trainableLayer.GetOutWeightsGrads(),
					opt.getRhoSquareDenseWeights(i),
				)
				scaledBiasGrads := opt.gradsScaleSquaredVec(
					trainableLayer.GetOutBiasGrads(),
					opt.getRhoSquareDenseBias(i),
				)
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					&scaledGrads,
					&scaledBiasGrads,
				)
			}
		}
	}
	return &grads
}

func (opt *RMSProp) BackwardConv2DLayers(convs2D *[]conv.ConvLayer, denseGrads *mat.VecDense) {
	gradsMat := functools.VecToMatSlice(
		denseGrads,
		opt.lastConvOutputSize.Height(),
		opt.lastConvOutputSize.Width(),
	)
	for i, convLayer := range slices.Backward(*convs2D) {
		gradsMat = *convLayer.Backward(&gradsMat)
		if trainableLayer, ok := (*convs2D)[i].(conv.ConvLayerTrainable); ok {
			if opt.rho == 0.0 {
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					opt.zeroRhoActivateDenseSlice(trainableLayer.GetFilterGrads()),
					opt.zeroRhoActivateFloatSlice(trainableLayer.GetBiasGrads()),
				)
			} else {
				convGradsSquared, convBiasSquared := opt.squareConvLayerGrads(
					trainableLayer.GetFilterGrads(),
					trainableLayer.GetBiasGrads(),
				)
				opt.rhoUpdateConv(i, &convGradsSquared, &convBiasSquared)
				scaledGrads := opt.gradsScaleSquaredConv(
					trainableLayer.GetFilterGrads(),
					opt.getRhoSquareConvWeigts(i),
				)
				scaledBiasGrads := opt.gradsScaleSquaredFloatSlice(
					trainableLayer.GetBiasGrads(),
					opt.getRhoSquareConvBias(i),
				)
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					&scaledGrads,
					&scaledBiasGrads,
				)
			}
		}
	}
}
