use rustml::{Matrix, NeuralNetwork};

fn main() {
    let input = Matrix::from_iter(4, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let output = Matrix::from_iter(4, 1, vec![0.0, 1.0, 1.0, 0.0]);

    println!("input = {input}");
    println!("output = {output}");

    let eps = 1e-1;
    let rate = 1e-1;

    let mut nn = NeuralNetwork::new_random(&[2.0, 2.0, 1.0]);
    let mut gradient = NeuralNetwork::new_random(&[2.0, 2.0, 1.0]);

    // println!("nn = {nn}");

    println!("cost = {}", nn.cost(&input, &output));
    for i in 0..10_000 {
        nn.finite_diff(&mut gradient, &eps, &input, &output);
        nn.learn(&mut gradient, &rate);
        println!("i: {i} cost: {}", nn.cost(&input, &output));
    }

    println!("nn after training = {nn}");
    println!("cost: {}", nn.cost(&input, &output));

    for row in 0..2 {
        for col in 0..2 {
            let res = nn.test(&Matrix::from_iter(1, 2, vec![row as f64, col as f64]));
            println!("{row} ^ {col} = {}", res.get(0, 0).unwrap(),);
        }
    }
}
