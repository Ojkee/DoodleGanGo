package optimizers

import (
	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/layers"
)

type Optimizer interface {
	BackwardDenseLayers(denses *[]layers.Layer, loss *mat.VecDense) *mat.VecDense
	BackwardConv2DLayers(convs2D *[]conv.ConvLayer, denseGrads *mat.VecDense)
}
