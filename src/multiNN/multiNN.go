package multiNN

import (
	"ff"
	"fmt"
)

type MultiNN struct {
	NN                []*ff.FeedForward
	AllWeightsNetwork []float64
}

func (ff *MultiNN) GetAllWeightsNetwork() {
	ff.AllWeightsNetwork = []float64{}
	for _, v := range ff.NN {
		v.MatrixsToVector()
		ff.AllWeightsNetwork = append(ff.AllWeightsNetwork, v.AllWeights...)
	}
}

func (ff *MultiNN) SetAllWeightsNetwork() {
	start := 0
	for _, v := range ff.NN {
		end := v.NInputs*v.NHiddens + v.NHiddens*v.NOutputs
		v.AllWeights = []float64{}
		v.AllWeights = append(v.AllWeights, ff.AllWeightsNetwork[start:start+end]...)
		v.VectorToMatrix()
		start = start + end
	}
}

func (Mnn *MultiNN) Init(layers [][]int) {
	for _, v := range layers {
		ff := ff.FeedForward{}
		ff.Init(v[0], v[1], v[2])
		Mnn.NN = append(Mnn.NN, &ff)
	}
}

func (ff *MultiNN) Update(patern []float64) []float64 {
	res := patern
	for _, v := range ff.NN {
		res = v.Predict(res)
	}
	return res
}

func (ff *MultiNN) BackProp(target []float64, lRate, mFactor float64) float64 {
	t := target
	sum := float64(0)
	for i := len(ff.NN) - 1; i >= 0; i-- {
		err, deltas := ff.NN[i].BackPropagate(t, lRate, mFactor)
		if i > 0 {
			t = nil
			for k, v := range deltas {
				t = append(t, v+ff.NN[i-1].OutputActivations[k])
			}
		}
		if i == len(ff.NN)-1 {
			sum += err
		} else {
			sum += 0
		}
	}
	//e,_:=ff.nn[len(ff.nn)-1].BackPropagate(t, lRate, mFactor)
	//sum += e
	return sum
}

func (nn *MultiNN) Train(patterns [][][]float64, iterations int, lRate, mFactor float64, debug bool) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64
		for _, p := range patterns {
			nn.Update(p[0])

			tmp := nn.BackProp(p[1], lRate, mFactor)
			e += tmp
		}

		errors[i] = e

		if debug && i%10 == 0 {
			fmt.Printf("Iteraton: %d, Avg. Error: %f \n", i, e)
		}
	}

	return errors
}

func (nn *MultiNN) Test(patterns [][][]float64) {
	correct := 0
	err := 0
	for _, p := range patterns {
		key := 0
		res := float64(0)
		key1 := 0
		res1 := float64(0)
		for k, v := range nn.Update(p[0]) {
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

	}
	fmt.Printf("Found correctly: %d, Found wrong: %d \n", correct, err)
}
