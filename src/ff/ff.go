package ff

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
)

type FeedForward struct {
	InputNodes   int
	OutputNodes  int
	LayerNumbers int
	Layers       []int
	Weights      []float64
	wMatrix      []mat64.Dense
	LRate        float64
}

func countError(s mat64.Dense, d []float64) []float64 {
	s.Apply(func(i, j int, v float64) float64 {
		return v * d[j]
	}, &s)
	c, _ := s.Caps()
	res := make([]float64, c)
	for i := 0; i < c; i++ {
		col := s.RowView(i).RawVector().Data
		for _, value := range col {
			res[i] += value
		}
	}
	return res
}

func applySigm(i, j int, v float64) float64 {
	return sigmoid(v)
}

func (ff *FeedForward) Init(inputNodes, OutputNodes int, layers []int, lRate float64) {
	ff.InputNodes = inputNodes
	ff.OutputNodes = OutputNodes
	fl := ff.InputNodes
	ff.Layers = layers
	ff.LRate = lRate
	sum := 0
	for _, v := range ff.Layers {
		sum = sum + (fl * v)
		fl = v
	}
	sum = sum + (fl * ff.OutputNodes)
	ff.Weights = make([]float64, sum)
	for i := 0; i < len(ff.Weights); i++ {
		ff.Weights[i] = random(0, 1)
	}
}
func (ff *FeedForward) Predict(pattern []float64) []float64 {
	var result_itog mat64.Dense
	updatedPattern := pattern
	initMat := mat64.NewDense(1, len(updatedPattern), updatedPattern)
	fl := ff.InputNodes
	split := 0
	for _, v := range ff.Layers {
		var result mat64.Dense
		secondMatrix := mat64.NewDense(fl, v, ff.Weights[split:(split+(fl*v))])
		result.Mul(initMat, secondMatrix)
		result.Apply(applySigm, &result)
		*initMat = result
		fl = v
		split += fl * v
	}
	secondMatrix := mat64.NewDense(fl, ff.OutputNodes, ff.Weights[split:(split+(fl*ff.OutputNodes))])
	result_itog.Mul(initMat, secondMatrix)
	result_itog.Apply(applySigm, &result_itog)
	return result_itog.RawRowView(0)
}

func (ff *FeedForward) predict_for_train(pattern []float64) {
	var result_itog mat64.Dense
	updatedPattern := pattern
	initMat := mat64.NewDense(1, len(updatedPattern), updatedPattern)
	ff.wMatrix = append(ff.wMatrix, *initMat)
	fl := ff.InputNodes
	split := 0
	for _, v := range ff.Layers {
		var result mat64.Dense
		secondMatrix := mat64.NewDense(fl, v, ff.Weights[split:(split+(fl*v))])
		result.Mul(initMat, secondMatrix)
		result.Apply(applySigm, &result)
		*initMat = result
		fl = v
		split += fl * v
		ff.wMatrix = append(ff.wMatrix, *secondMatrix)
		ff.wMatrix = append(ff.wMatrix, result)
	}
	secondMatrix := mat64.NewDense(fl, ff.OutputNodes, ff.Weights[split:(split+(fl*ff.OutputNodes))])
	result_itog.Mul(initMat, secondMatrix)
	result_itog.Apply(applySigm, &result_itog)

	ff.wMatrix = append(ff.wMatrix, *secondMatrix)
	ff.wMatrix = append(ff.wMatrix, result_itog)

}

func (ff *FeedForward) backPropagation(targets []float64) {
	var delta_w []float64
	for cnt := len(ff.wMatrix) - 2; cnt >= 1; cnt -= 2 {
		var err_lay []float64
		res := ff.wMatrix[cnt+1].RawRowView(0)

		if len(ff.wMatrix)-2 == cnt {
			for k := 0; k < len(res); k++ {
				err_lay = append(err_lay, targets[k]-res[k])
			}
		} else {
			fmt.Println(len(delta_w))
			fmt.Println(ff.wMatrix[cnt+2])
			err_lay = countError(ff.wMatrix[cnt+2], delta_w)
		}

		delta_w = nil
		_, col := ff.wMatrix[cnt].Caps()
		delta_w = make([]float64, col)

		ff.wMatrix[cnt].Apply(
			func(i, j int, v float64) float64 {
				gradient_lay := dsigmoid(res[j])
				delta_w[j] = err_lay[j] * gradient_lay
				return v - (res[j] * delta_w[j] * ff.LRate)
			}, &ff.wMatrix[cnt])
	}

	var newW []float64
	for i := 1; i < len(ff.wMatrix); i += 2 {
		newW = append(newW, ff.wMatrix[i].RawMatrix().Data...)
	}

	copy(ff.Weights, newW)

}

//func (ff *FeedForward) backPropagation2(targets []float64) {
//	targetM := mat64.NewDense(1, len(targets), targets)
//	var tempMatrix *mat64.Dense
//	for cnt := len(ff.wMatrix) - 2; cnt >= 1; cnt -= 2 {
//		var (
//			error_layer         mat64.Dense
//			gradient_layer      mat64.Dense
//			weights_delta_layer mat64.Dense
//		)
//		res := ff.wMatrix[cnt+1]
//		if cnt == len(ff.wMatrix)-2 {
//			error_layer.Sub(&res, targetM)
//		} else {
//			error_layer.Mul(tempMatrix, ff.wMatrix[cnt+2].T())
//		}
//
//		gradient_layer.Apply(func(i, j int, v float64) float64 {
//			return dActivateFunction(v)
//		}, &ff.wMatrix[cnt+1])
//
//		if cnt == len(ff.wMatrix)-2 {
//			weights_delta_layer.Mul(error_layer.T(), &gradient_layer)
//		} else {
//			weights_delta_layer.Mul(&gradient_layer, error_layer.T())
//		}
//		weights_delta_layer.Apply(func(i, j int, v float64) float64 {
//			return v * ff.LRate
//		}, &weights_delta_layer)
//
//		if cnt == len(ff.wMatrix)-2 {
//			ff.wMatrix[cnt].Sub(&ff.wMatrix[cnt], weights_delta_layer.T())
//		} else {
//			fmt.Println("##########")
//			fmt.Println(ff.wMatrix[cnt])
//			fmt.Println(weights_delta_layer)
//			fmt.Println("##########")
//			ff.wMatrix[cnt].Sub(&ff.wMatrix[cnt], weights_delta_layer.T())
//		}
//		tempMatrix = &weights_delta_layer
//
//	}
//
//	//var newW []float64
//	//for i := 1; i < len(ff.wMatrix); i += 2 {
//	//	newW = append(newW, ff.wMatrix[i].RawMatrix().Data...)
//	//}
//	//copy(ff.Weights, newW)
//}

func (ff *FeedForward) Train(patterns [][][]float64, epohe int, debug bool) {
	for i := 0; i < epohe; i++ {
		sum := float64(0)
		for _, v := range patterns {
			ff.predict_for_train(v[0])
			ff.backPropagation(v[1])

			res := ff.wMatrix[len(ff.wMatrix)-1].RawRowView(0)
			for key, value := range v[1] {
				sum += 0.5 * math.Pow((res[key]-value), 2)
			}

			ff.wMatrix = nil
		}
		if debug {
			fmt.Println("Middle Error after ", i, " epohe", sum/float64(len(patterns)))
		}
	}

}

func (ff *FeedForward) Test(patterns [][][]float64) {
	correct := 0
	err := 0
	for _, p := range patterns {
		key := 0
		res := float64(0)
		key1 := 0
		res1 := float64(0)
		for k, v := range ff.Predict(p[0]) {
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
		fmt.Println(ff.Predict(p[0]))
		fmt.Println(p[1])
	}

	fmt.Println("correct: ", correct, "error: ", err)
}
