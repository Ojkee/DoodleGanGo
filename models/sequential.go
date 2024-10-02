package models

import (
	"DoodleGan/conv"
	"DoodleGan/layers"
	"DoodleGan/losses"
	"DoodleGan/optimizers"
)

type Sequential struct {
	batchSize int
	epochs    int

	denseLayers  []*layers.Layer
	convLayers   []*conv.ConvLayer
	optimizer    *optimizers.Optimizer
	lossFunction *losses.Loss

	correctGuesses uint
	totalGuesses   uint
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

func (model *Sequential) SetLoss(lossFunction *losses.Loss) {
	model.lossFunction = lossFunction
}

func (model *Sequential) Train(X, y *[]float64) {
}
