package ff

import (
	"math"
	"math/rand"
	"time"
)

func random(a, b float64) float64 {
	rand.Seed(time.Now().UnixNano())
	res := float64(rand.Int31n(int32(b-a)))*rand.Float64() + a
	if res == -1 {
		return random(a, b)
	}
	return res
}

func activateFunction(x float64) float64 {
	// sigmoid
	return 1 / (1 + math.Exp(-x))
}

func dActivateFunction(y float64) float64 {
	// dsigmoid
	return y * (1 - y)
}
