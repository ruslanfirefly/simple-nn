package ff

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"log"
	"math"
	"math/rand"
)

type FeedForward struct {
	NInputs, NHiddens, NOutputs                            int
	Regression                                             bool
	InputActivations, HiddenActivations, OutputActivations []float64
	InputWeights, OutputWeights                            *mat64.Dense
	InputChanges, OutputChanges                            *mat64.Dense
	AllWeights                                             []float64
}

func (nn *FeedForward) MatrixsToVector() {
	nn.AllWeights = []float64{}
	nn.AllWeights = append(nn.AllWeights, nn.InputWeights.RawMatrix().Data...)
	nn.AllWeights = append(nn.AllWeights, nn.OutputWeights.RawMatrix().Data...)
}
func (nn *FeedForward) VectorToMatrix() {
	nn.InputWeights = mat64.NewDense(nn.NInputs, nn.NHiddens, nn.AllWeights[0:(nn.NInputs*nn.NHiddens)])
	nn.OutputWeights = mat64.NewDense(nn.NHiddens, nn.NOutputs, nn.AllWeights[(nn.NInputs*nn.NHiddens):((nn.NInputs*nn.NHiddens)+(nn.NHiddens*nn.NOutputs))])
}

func (nn *FeedForward) Init(inputs, hiddens, outputs int) {
	nn.NInputs = inputs + 1   // +1 for bias
	nn.NHiddens = hiddens + 1 // +1 for bias
	nn.NOutputs = outputs

	nn.InputActivations = vector(nn.NInputs, 1.0)
	nn.HiddenActivations = vector(nn.NHiddens, 1.0)
	nn.OutputActivations = vector(nn.NOutputs, 1.0)

	matrix1 := make([]float64, nn.NInputs*nn.NHiddens)

	for i := range matrix1 {
		matrix1[i] = rand.NormFloat64()
	}

	nn.InputWeights = mat64.NewDense(nn.NInputs, nn.NHiddens, matrix1)

	matrix2 := make([]float64, nn.NHiddens*nn.NOutputs)

	for i := range matrix2 {
		matrix2[i] = rand.NormFloat64()
	}

	nn.OutputWeights = mat64.NewDense(nn.NHiddens, nn.NOutputs, matrix2)

	nn.InputChanges = mat64.NewDense(nn.NInputs, nn.NHiddens, make([]float64, nn.NInputs*nn.NHiddens))
	nn.OutputChanges = mat64.NewDense(nn.NHiddens, nn.NOutputs, make([]float64, nn.NHiddens*nn.NOutputs))
}

func (nn *FeedForward) Predict(inputs []float64) []float64 {
	if len(inputs) != nn.NInputs-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	for i := 0; i < nn.NInputs-1; i++ {
		nn.InputActivations[i] = inputs[i]
	}

	for i := 0; i < nn.NHiddens-1; i++ {
		var sum float64

		for j := 0; j < nn.NInputs; j++ {
			sum += nn.InputActivations[j] * nn.InputWeights.At(j, i)
		}

		nn.HiddenActivations[i] = sigmoid(sum)
	}

	for i := 0; i < nn.NOutputs; i++ {
		var sum float64
		for j := 0; j < nn.NHiddens; j++ {
			sum += nn.HiddenActivations[j] * nn.OutputWeights.At(j, i)
		}

		nn.OutputActivations[i] = sigmoid(sum)
	}

	return nn.OutputActivations
}

/*
The BackPropagate method is used, when training the Neural Network,
to back propagate the errors from network activation.
*/
func (nn *FeedForward) BackPropagate(targets []float64, lRate, mFactor float64) (float64, []float64) {
	if len(targets) != nn.NOutputs {
		log.Fatal("Error: wrong number of target values")
	}

	outputDeltas := vector(nn.NOutputs, 0.0)
	for i := 0; i < nn.NOutputs; i++ {
		outputDeltas[i] = dsigmoid(nn.OutputActivations[i]) * (targets[i] - nn.OutputActivations[i])
	}

	hiddenDeltas := vector(nn.NHiddens, 0.0)

	for i := 0; i < nn.NHiddens; i++ {
		var e float64

		for j := 0; j < nn.NOutputs; j++ {
			e += outputDeltas[j] * nn.OutputWeights.At(i, j)
		}

		hiddenDeltas[i] = dsigmoid(nn.HiddenActivations[i]) * e
	}

	inputDeltas := vector(nn.NInputs-1, 0.0)

	for i := 0; i < nn.NInputs-1; i++ {
		var e float64

		for j := 0; j < nn.NHiddens; j++ {
			e += hiddenDeltas[j] * nn.InputWeights.At(i, j)
		}

		inputDeltas[i] = dsigmoid(nn.InputActivations[i]) * e
	}

	for i := 0; i < nn.NHiddens; i++ {
		for j := 0; j < nn.NOutputs; j++ {
			change := outputDeltas[j] * nn.HiddenActivations[i]
			nn.OutputWeights.Set(i, j, nn.OutputWeights.At(i, j)+lRate*change+mFactor*nn.OutputChanges.At(i, j))
			nn.OutputChanges.Set(i, j, change)
		}
	}

	for i := 0; i < nn.NInputs; i++ {
		for j := 0; j < nn.NHiddens; j++ {
			change := hiddenDeltas[j] * nn.InputActivations[i]
			nn.InputWeights.Set(i, j, nn.InputWeights.At(i, j)+lRate*change+mFactor*nn.InputChanges.At(i, j))
			nn.InputChanges.Set(i, j, change)
		}
	}

	var e float64

	for i := 0; i < len(targets); i++ {
		e += 0.5 * math.Pow(targets[i]-nn.OutputActivations[i], 2)
	}

	return e, inputDeltas
}

/*
This method is used to train the Network, it will run the training operation for 'iterations' times
and return the computed errors when training.
*/
func (nn *FeedForward) Train(patterns [][][]float64, iterations int, lRate, mFactor float64, debug bool) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64
		for _, p := range patterns {
			nn.Predict(p[0])

			tmp, _ := nn.BackPropagate(p[1], lRate, mFactor)
			e += tmp
		}

		errors[i] = e

		if debug {
			fmt.Printf("Iteraton: %d, Avg. Error: %e \n", i, e)
		}
	}

	return errors
}

func (nn *FeedForward) Test(patterns [][][]float64) {
	correct := 0
	err := 0
	for _, p := range patterns {
		key := 0
		res := float64(0)
		key1 := 0
		res1 := float64(0)
		for k, v := range nn.Predict(p[0]) {
			if v > float64(res) {
				res = v
				key = k
			}
		}
		for k, v := range p[1] {
			if v >= float64(res1) {
				res1 = v
				key1 = k
			}
		}
		if key == key1 {
			correct++
		} else {
			err++
		}
		fmt.Println(nn.Predict(p[0]))
		fmt.Println(p[1])
	}
	fmt.Printf("Found correctly: %d, Found wrong: %d \n", correct, err)
}
