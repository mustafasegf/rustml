use super::matrix::Matrix;

#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    size: usize,
    weight: Vec<Matrix>,
    bias: Vec<Matrix>,
    activation: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(data: &[usize]) -> Self {
        let size = data.len();
        assert!(size > 0);

        Self::from_iter(data, std::iter::repeat_with(|| 0.0))
    }

    pub fn from_iter<I>(data: &[usize], iter: I) -> Self
    where
        I: IntoIterator<Item = f32> + Clone,
    {
        let size = data.len();
        assert!(size > 0);

        let mut nn = NeuralNetwork {
            size: size - 1,
            weight: Vec::with_capacity(size),
            bias: Vec::with_capacity(size),
            activation: Vec::with_capacity(size),
        };

        nn.activation.push(Matrix::from_iter(
            1,
            data[0],
            std::iter::repeat_with(|| 0.0),
        ));

        for i in 1..size {
            nn.weight
                .push(Matrix::from_iter(data[i - 1], data[i], iter.clone()));
            nn.bias.push(Matrix::from_iter(1, data[i], iter.clone()));
            nn.activation
                .push(Matrix::from_iter(1, data[i], iter.clone()));
        }

        nn
    }

    pub fn set_input_take(&mut self, input: Matrix) {
        assert_eq!(self.activation[0].rows(), input.rows());
        assert_eq!(self.activation[0].cols(), input.cols());
        self.activation[0] = input;
    }

    pub fn set_input(&mut self, input: &Matrix) {
        assert_eq!(self.activation[0].rows(), input.rows());
        assert_eq!(self.activation[0].cols(), input.cols());

        for row in 0..input.rows() {
            for col in 0..input.cols() {
                self.activation[0].set(row, col, *input.get(row, col).unwrap());
            }
        }
    }

    pub fn get_output(&self) -> &Matrix {
        &self.activation[self.size]
    }

    pub fn forward(&mut self) -> &Matrix {
        for i in 0..self.size {
            let (prev_layer, next_layer) = self.activation.split_at_mut(i + 1);
            let next_activation = &mut next_layer[0];
            let activation = &prev_layer[i];
            let weight = &self.weight[i];
            let bias = &self.bias[i];

            next_activation.dot_from(activation, weight);
            next_activation.add_from(bias);
            next_activation.sigmoid();
        }

        &self.activation.last().unwrap()
    }

    pub fn cost(&mut self, input: &Matrix, output: &Matrix) -> f32 {
        assert!(input.rows() == output.rows());
        assert!(output.cols() == self.activation[self.size].cols());

        let mut result = 0.0;
        let n = input.rows();

        for training in 0..input.rows() {
            let x = input.get_row_matrix(training).unwrap();
            let truth_ouput = output.get_row_matrix(training).unwrap();

            self.set_input_take(x);
            let model_output = self.forward();

            for col in 0..output.cols() {
                let d = model_output.get(0, col).unwrap() - truth_ouput.get(0, col).unwrap();
                result += d * d;
            }
        }

        result / (n as f32)
    }

    pub fn finite_diff(&mut self, gradient: &mut Self, eps: &f32, input: &Matrix, output: &Matrix) {
        let c = self.cost(input, output);

        for i in 0..self.weight.len() {
            for row in 0..self.weight[i].rows() {
                for col in 0..self.weight[i].cols() {
                    let temp = *self.weight[i].get(row, col).unwrap();
                    self.weight[i].set(row, col, temp + eps);

                    gradient.weight[i].set(row, col, (self.cost(input, output) - c) / eps);
                    self.weight[i].set(row, col, temp);
                }
            }

            for row in 0..self.bias[i].rows() {
                for col in 0..self.bias[i].cols() {
                    let temp = *self.bias[i].get(row, col).unwrap();
                    self.bias[i].set(row, col, temp + eps);
                    gradient.bias[i].set(row, col, (self.cost(input, output) - c) / eps);
                    self.bias[i].set(row, col, temp);
                }
            }
        }
    }

    pub fn learn(&mut self, gradient: &mut Self, rate: &f32) {
        for i in 0..self.size {
            for row in 0..self.weight[i].rows() {
                for col in 0..self.weight[i].cols() {
                    *self.weight[i].get_mut(row, col).unwrap() -=
                        rate * gradient.weight[i].get(row, col).unwrap();
                }
            }

            for row in 0..self.bias[i].rows() {
                for col in 0..self.bias[i].cols() {
                    *self.bias[i].get_mut(row, col).unwrap() -=
                        rate * gradient.bias[i].get(row, col).unwrap();
                }
            }
        }
    }

    pub fn test(&mut self, input: &Matrix) -> Matrix {
        self.activation[0] = input.clone();
        self.forward();
        self.activation[self.size].clone()
    }
}

impl std::fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "size: {}", self.size)?;
        for i in 0..self.size {
            writeln!(f, "weight[{}]:\n{}", i, self.weight[i])?;
            writeln!(f, "bias[{}]:\n{}", i, self.bias[i])?;
            writeln!(f, "activation[{}]:\n{}", i, self.activation[i])?;
        }
        writeln!(
            f,
            "activation[{}]:\n{}",
            self.size, self.activation[self.size]
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let nn = NeuralNetwork::new(&[2, 3, 1]);
        assert_eq!(nn.size, 2);
        assert_eq!(nn.weight.len(), 2);
        assert_eq!(nn.bias.len(), 2);
        assert_eq!(nn.activation.len(), 3);
        assert_eq!(nn.activation[0], Matrix::new(1, 2));
        assert_eq!(nn.activation[1], Matrix::new(1, 3));
        assert_eq!(nn.activation[2], Matrix::new(1, 1));
        assert_eq!(nn.weight[0], Matrix::new(2, 3));
        assert_eq!(nn.weight[1], Matrix::new(3, 1));
        assert_eq!(nn.bias[0], Matrix::new(1, 3));
        assert_eq!(nn.bias[1], Matrix::new(1, 1));
    }

    #[test]
    fn test_new_random() {
        let nn = NeuralNetwork::from_iter(&[2, 3, 1], std::iter::repeat(1.1));
        assert_eq!(nn.size, 2);
        assert_eq!(nn.weight.len(), 2);
        assert_eq!(nn.bias.len(), 2);
        assert_eq!(nn.activation.len(), 3);
        assert_eq!(nn.activation[1].rows(), 1);
        assert_eq!(nn.activation[1].cols(), 3);
        assert_eq!(nn.activation[2].rows(), 1);
        assert_eq!(nn.activation[2].cols(), 1);

        assert!(nn.weight[0].iter().all(|&x| x == 1.0));
        assert!(nn.weight[1].iter().all(|&x| x == 1.0));
        assert!(nn.bias[0].iter().all(|&x| x == 1.0));
        assert!(nn.bias[1].iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_forward() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1]);
        nn.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);

        nn.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);
        nn.activation[0] = Matrix::from_iter(1, 2, vec![0.5, 0.6]);

        nn.forward();

        assert_eq!(
            nn.activation[1],
            Matrix::from_iter(
                1,
                3,
                vec![0.5962826992967879, 0.6456563062257954, 0.6921095043017882]
            )
        );
        assert_eq!(
            nn.activation[2],
            Matrix::from_iter(1, 1, vec![0.8761885526812198])
        );
    }

    #[test]
    fn test_cost() {
        let mut nn = NeuralNetwork::new(&[2, 2, 1]);
        nn.weight[0] = Matrix::from_iter(2, 2, vec![0.840188, 0.394383, 0.783099, 0.798440]);
        nn.weight[1] = Matrix::from_iter(2, 1, vec![0.335223, 0.768230]);
        nn.bias[0] = Matrix::from_iter(1, 2, vec![0.911647, 0.197551]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.277775]);

        let input = Matrix::from_iter(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        let output = Matrix::from_iter(4, 1, vec![0.0, 1.0, 1.0, 0.0]);

        let cost = nn.cost(&input, &output);

        // assert_eq!(cost, 0.308771);
        assert_eq!(cost, 0.3087706);
    }

    #[test]
    fn test_finite_diff() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1]);
        nn.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);
        nn.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);

        let input = Matrix::from_iter(2, 2, vec![0.1, 0.2, 0.3, 0.4]);
        let output = Matrix::from_iter(2, 1, vec![0.5, 0.6]);

        let mut gradient = NeuralNetwork::new(&[2, 3, 1]);
        let eps = 0.0001;
        nn.finite_diff(&mut gradient, &eps, &input, &output);

        assert_eq!(
            gradient.weight[0],
            Matrix::from_iter(
                2,
                3,
                vec![
                    0.002338431382142847,
                    0.0026023290392029885,
                    0.0028172072125132175,
                    0.0036226984256870765,
                    0.004037243530596868,
                    0.004379096906204083
                ]
            )
        );
        assert_eq!(
            gradient.weight[1],
            Matrix::from_iter(
                3,
                1,
                vec![
                    0.04148976830056772,
                    0.044146948279810694,
                    0.046728225587544525
                ]
            )
        );
        assert_eq!(
            gradient.bias[0],
            Matrix::from_iter(
                1,
                3,
                vec![
                    0.012842612617247617,
                    0.014349051058182294,
                    0.01561876383046612
                ]
            )
        );
        assert_eq!(
            gradient.bias[1],
            Matrix::from_iter(1, 1, vec![0.07441508547284537])
        );
    }

    #[test]
    fn test_learn() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1]);
        nn.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);
        nn.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);
        nn.activation[0] = Matrix::from_iter(1, 2, vec![0.5, 0.6]);

        let mut gradient = NeuralNetwork::new(&[2, 3, 1]);
        gradient.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        gradient.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);
        gradient.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        gradient.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);

        nn.learn(&mut gradient, &0.1);

        assert_eq!(
            nn.weight[0],
            Matrix::from_iter(2, 3, vec![0.090, 0.180, 0.270, 0.360, 0.450, 0.540])
        );
        assert_eq!(
            nn.weight[1],
            Matrix::from_iter(3, 1, vec![0.630, 0.720, 0.810])
        );
        assert_eq!(nn.bias[0], Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]));
        assert_eq!(nn.bias[1], Matrix::from_iter(1, 1, vec![0.4]));
    }

    #[test]
    fn test_nn_test() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1]);
        nn.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);
        nn.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);

        let input = Matrix::from_iter(1, 2, vec![0.5, 0.6]);
        let output = nn.test(&input);

        assert_eq!(output, Matrix::from_iter(1, 1, vec![0.8761885526812198]));
    }
}
