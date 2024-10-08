package optimizers

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/layers"
)

type Optimizer interface {
	BackwardDenseLayers(denses *[]layers.Layer, loss *mat.VecDense) *mat.VecDense
	BackwardConv2DLayers(convs2D *[]conv.ConvLayer, denseGrads *mat.VecDense)
}

func checkValidLearningRate(learningRate *float64, funcName string) {
	if *learningRate <= 0.0 {
		panic(fmt.Sprintf("%s fail:\n\tLearning Rate can't be less or equal 0", funcName))
	}
}

func checkValidEps(eps *float64, funcName string) {
	if *eps < 0.0 {
		panic(fmt.Sprintf("%s fail:\n\teps can't be less or equal 0", funcName))
	}
}

func checkValidMomentum(momentum *float64, funcName string) {
	if *momentum < 0.0 || *momentum >= 1.0 {
		panic(fmt.Sprintf("%s fail:\n\tmomentum must be in range [0, 1)", funcName))
	}
}

func checkValidRho(rho *float64, funcName string) {
	if *rho < 0.0 || *rho >= 1.0 {
		panic(fmt.Sprintf("%s fail:\n\trho must be in range [0, 1)", funcName))
	}
}
