package optimizers

import (
	"math"
	"slices"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
	"DoodleGan/layers"
)

type RMSProp struct {
	learningRate       float64
	momentum           float64
	momentumComplement float64
	rho                float64
	rhoComplement      float64
	eps                float64

	rootFunc func(i, j int, v float64) float64

	convSquared     map[int]*filterMomentum // key: idx of conv layer in passed architecture
	denseSquared    map[int]*denseMomentum  // key: idx of dense layer in passed architecture
	convVelocities  map[int]*filterMomentum // TODO
	denseVelocities map[int]*denseMomentum  // TODO

	lastConvOutputSize conv.MatSize
}

func NewRMSProp(learningRate, rho, momentum, eps float64) RMSProp {
	if learningRate <= 0 {
		panic("NewRMSProp fail:\n\tLearning Rate can't be less or equal 0")
	}
	if rho < 0 || rho > 1 {
		panic("NewRMSProp fail:\n\trho must be in range [0, 1]")
	}
	rootFunc_ := func(i, j int, v float64) float64 {
		if v == 0.0 {
			v += eps
		}
		return math.Sqrt(v)
	}

	return RMSProp{
		learningRate:       learningRate,
		rho:                rho,
		rhoComplement:      1.0 - rho,
		momentum:           momentum,
		momentumComplement: 1.0 - momentum,
		eps:                eps,
		rootFunc:           rootFunc_,
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
		opt.denseSquared = initDenseVelocities(denseLayers)
		opt.convSquared = initConvVelocities(convLayers)
	}
	if opt.momentum != 0.0 {
		opt.denseVelocities = initDenseVelocities(denseLayers)
		opt.convVelocities = initConvVelocities(convLayers)
	}
}

func (opt *RMSProp) BackwardDenseLayers(denses *[]layers.Layer, loss *mat.VecDense) *mat.VecDense {
	grads := *loss
	for i, denseLayer := range slices.Backward(*denses) {
		grads = *denseLayer.Backward(&grads)
		if trainableLayer, ok := denseLayer.(layers.LayerTrainable); ok {
			denseGradsSquared, vecBiasGradsSquared := squareDenseLayerGrads(
				trainableLayer.GetOutWeightsGrads(),
				trainableLayer.GetOutBiasGrads(),
			)
			if opt.rho == 0.0 {
				scaledGrads := opt.gradsScaleSquaredDense(
					trainableLayer.GetOutWeightsGrads(),
					&denseGradsSquared,
				)
				scaledBiasGrads := opt.gradsScaleSquaredVec(
					trainableLayer.GetOutBiasGrads(),
					&vecBiasGradsSquared,
				)
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					&scaledGrads,
					&scaledBiasGrads,
				)
			} else {
				opt.denseSquared[i].updateWeight(&denseGradsSquared, &opt.rho, &opt.rhoComplement)
				opt.denseSquared[i].updateBias(&vecBiasGradsSquared, &opt.rho, &opt.rhoComplement)
				scaledGrads := opt.gradsScaleSquaredDense(
					trainableLayer.GetOutWeightsGrads(),
					&opt.denseSquared[i].weightsVelocities,
				)
				scaledBiasGrads := opt.gradsScaleSquaredVec(
					trainableLayer.GetOutBiasGrads(),
					&opt.denseSquared[i].biasesVelocities,
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
	for i := len(*convs2D) - 1; i >= 0; i-- {
		gradsMat = *(*convs2D)[i].Backward(&gradsMat)
		if trainableLayer, ok := (*convs2D)[i].(conv.ConvLayerTrainable); ok {
			convGradsSquared, convBiasSquared := squareConvLayerGrads(
				trainableLayer.GetFilterGrads(),
				trainableLayer.GetBiasGrads(),
			)
			if opt.rho == 0.0 {
				scaledGrads := opt.gradsScaleSquaredConv(
					trainableLayer.GetFilterGrads(),
					&convGradsSquared,
				)
				scaledBiasGrads := opt.gradsScaleSquaredFloatSlice(
					trainableLayer.GetBiasGrads(),
					&convBiasSquared,
				)
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					&scaledGrads,
					&scaledBiasGrads,
				)
			} else {
				opt.convSquared[i].updateFilter(&convGradsSquared, &opt.rho, &opt.rhoComplement)
				opt.convSquared[i].updateBias(&convBiasSquared, &opt.rho, &opt.rhoComplement)
				scaledGrads := opt.gradsScaleSquaredConv(
					trainableLayer.GetFilterGrads(),
					&opt.convSquared[i].channelsVelocities,
				)
				scaledBiasGrads := opt.gradsScaleSquaredFloatSlice(
					trainableLayer.GetBiasGrads(),
					&opt.convSquared[i].biasesVelocities,
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

// TODO Preallocate space for squared grads instead of allocating every opt step
func squareDenseLayerGrads(
	denseGrads *mat.Dense,
	biasGrads *mat.VecDense,
) (mat.Dense, mat.VecDense) {
	var denseGradsRet mat.Dense
	denseGrads.MulElem(
		denseGrads,
		denseGrads,
	)
	var vecBiasGradsRet mat.VecDense
	vecBiasGradsRet.MulElemVec(
		biasGrads,
		biasGrads,
	)
	return denseGradsRet, vecBiasGradsRet
}

// TODO Preallocate space for squared grads instead of allocating every opt step
func squareConvLayerGrads(
	convGrads *[]mat.Dense,
	biasGrads *[]float64,
) ([]mat.Dense, []float64) {
	convGradsRet := make([]mat.Dense, len(*convGrads))
	convBiasRet := make([]float64, len(*biasGrads))
	for i, convGrad := range *convGrads {
		convGradsRet[i].MulElem(&convGrad, &convGrad)
	}
	for i, biasGrad := range *biasGrads {
		convBiasRet[i] = biasGrad * biasGrad
	}
	return convGradsRet, convBiasRet
}

func (opt *RMSProp) gradsScaleSquaredDense(grads, gradsS *mat.Dense) mat.Dense {
	var rootedS mat.Dense
	rootedS.Apply(opt.rootFunc, gradsS)
	var retVal mat.Dense
	retVal.DivElem(grads, &rootedS)
	return retVal
}

func (opt *RMSProp) gradsScaleSquaredVec(grads, gradsS *mat.VecDense) mat.VecDense {
	retVal := mat.NewVecDense(grads.Len(), nil)
	for i := range grads.Len() {
		toSquare := gradsS.AtVec(i)
		if toSquare == 0.0 {
			toSquare += opt.eps
		}
		retVal.SetVec(i, grads.AtVec(i)/math.Sqrt(toSquare))
	}
	return *retVal
}

func (opt *RMSProp) gradsScaleSquaredConv(grads, gradsS *[]mat.Dense) []mat.Dense {
	retVal := make([]mat.Dense, len(*grads))
	for i := range retVal {
		retVal[i] = opt.gradsScaleSquaredDense(&(*grads)[i], &(*gradsS)[i])
	}
	return retVal
}

func (opt *RMSProp) gradsScaleSquaredFloatSlice(grads, gradsS *[]float64) []float64 {
	retVal := make([]float64, len(*grads))
	for i := range retVal {
		toSquare := (*gradsS)[i]
		if toSquare == 0.0 {
			toSquare += opt.eps
		}
		retVal[i] = (*grads)[i] / math.Sqrt(toSquare)
	}
	return retVal
}
