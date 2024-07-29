package layers

type FloatType interface {
	~float32 | ~float64
}

type Layer interface {
	Forward(input interface{}) interface{}
}
