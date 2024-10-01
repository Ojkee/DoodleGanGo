package models

import (
	"DoodleGan/conv"
	"DoodleGan/layers"
	"DoodleGan/optimizers"
)

type Sequential struct {
	denseLayers []*layers.Layer
	convLayers  []*conv.ConvLayer
	optimizer   optimizers.Optimizer
}

func NewSequential() Sequential {
	return Sequential{
		denseLayers: make([]*layers.Layer, 0),
		convLayers:  make([]*conv.ConvLayer, 0),
	}
}

func (model *Sequential) AddConvLayer(layer *conv.ConvLayer) {
	model.convLayers = append(model.convLayers, layer)
}

func (model *Sequential) AddDenseLayer(layer *layers.Layer) {
	model.denseLayers = append(model.denseLayers, layer)
}

func (model *Sequential) SetOptimizer(opt *optimizers.Optimizer) {
	model.optimizer = opt
}

func (model *Sequential) Train(X *[]float64, batchSize, epochs int) {
}
