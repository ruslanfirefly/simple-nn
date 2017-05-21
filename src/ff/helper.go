package ff

import (
	"math"
	"math/rand"
	"time"
)

func random(a, b float64) float64 {
	rand.Seed(time.Now().UnixNano())
	//return rand.Float64()
	return float64(1)
}

func activateFunction(x float64) float64 {
	// sigmoid
	return 1 / (1 + math.Exp(-x))
}

func dActivateFunction(y float64) float64 {
	// dsigmoid
	return y * (1 - y)
}
