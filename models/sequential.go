package models

import (
	"DoodleGan/conv"
	"DoodleGan/layers"
	"DoodleGan/optimizers"
)

type Sequential struct {
	denseLayers []layers.Layer
	convLayers  []conv.ConvLayer
	optimizer   optimizers.Optimizer
}
