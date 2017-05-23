package ff

import (
	"math"
	"math/rand"
	"time"
)

func random(a, b float64) float64 {
	rand.Seed(time.Now().UnixNano())
	return rand.Float64()
	//return float64(1)
}

func sigmoid(x float64) float64 {
	// sigmoid
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}
func matrix(I, J int) [][]float64 {
	m := make([][]float64, I)
	for i := 0; i < I; i++ {
		m[i] = make([]float64, J)
	}
	return m
}

func vector(I int, fill float64) []float64 {
	v := make([]float64, I)
	for i := 0; i < I; i++ {
		v[i] = fill
	}
	return v
}
