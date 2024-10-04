package optimizers

// import (
// 	"slices"
//
// 	"gonum.org/v1/gonum/mat"
//
// 	"DoodleGan/conv"
// 	"DoodleGan/functools"
// 	"DoodleGan/layers"
// )
//
// type Adam struct {
// 	learningRate float64
//
// 	momentumMechanism
// 	rhoSquareMechanism
// 	lastConvOutputSize conv.MatSize
// }
//
// func NewAdam(learningRate, rho, momentum, eps float64) Adam {
// 	if learningRate <= 0 {
// 		panic("NewAdam fail:\n\tLearning Rate can't be less or equal 0")
// 	}
// 	if rho < 0 || rho > 1 {
// 		panic("NewAdam fail:\n\trho must be in range [0, 1]")
// 	}
// 	if momentum < 0 || momentum > 1 {
// 		panic("NewAdam fail:\n\tmomentum must be in range [0, 1]")
// 	}
// 	return Adam{
// 		learningRate:       learningRate,
// 		momentumMechanism:  newMomentumMechanism(momentum),
// 		rhoSquareMechanism: newRhoSquareMechanism(rho, eps),
// 	}
// }
//
// func (opt *Adam) PreTrainInit(
// 	lastConvOutputSize [2]int,
// 	convLayers *[]conv.ConvLayer,
// 	denseLayers *[]layers.Layer,
// ) {
// 	opt.lastConvOutputSize = *conv.NewMatSize(
// 		lastConvOutputSize[0],
// 		lastConvOutputSize[1],
// 	)
// 	if opt.rho != 0.0 {
// 		opt.initRhoMechanizm(convLayers, denseLayers)
// 	}
// 	if opt.momentum != 0.0 {
// 		opt.initMomentumMechanizm(convLayers, denseLayers)
// 	}
// }
//
// func (opt *Adam) BackwardDenseLayers(denses *[]layers.Layer, loss *mat.VecDense) *mat.VecDense {
// 	grads := *loss
// 	for i, denseLayer := range slices.Backward(*denses) {
// 		grads = *denseLayer.Backward(&grads)
// 		if trainableLayer, ok := denseLayer.(layers.LayerTrainable); ok {
// 		}
// 	}
// 	return &grads
// }
//
// func (opt *Adam) BackwardConv2DLayers(convs2D *[]conv.ConvLayer, denseGrads *mat.VecDense) {
// 	gradsMat := functools.VecToMatSlice(
// 		denseGrads,
// 		opt.lastConvOutputSize.Height(),
// 		opt.lastConvOutputSize.Width(),
// 	)
// 	for i, convLayer := range slices.Backward(*convs2D) {
// 		gradsMat = *convLayer.Backward(&gradsMat)
// 		if trainableLayer, ok := (*convs2D)[i].(conv.ConvLayerTrainable); ok {
// 		}
// 	}
// }
