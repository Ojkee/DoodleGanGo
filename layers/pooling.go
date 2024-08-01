package layers

type AvgPool struct {
	poolSize [2]int

	SavedDataMat
}

func NewAvgPool(poolSize []int)

func (layer *AvgPool) Forward() {
}
