package models

import (
	"DoodleGan/conv"
	"DoodleGan/layers"
	"DoodleGan/optimizers"
)

type Model interface {
	AddConvLayer(layer *conv.ConvLayer)
	AddDenseLayer(layer *layers.Layer)
	SetOptimizer(opt *optimizers.Optimizer)
}
