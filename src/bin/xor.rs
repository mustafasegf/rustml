use rand::prelude::*;
use rand::rngs::StdRng;
use rustml::{Matrix, NeuralNetwork};

fn main() {
    let input = Matrix::from_iter(
        4,
        2,
        vec![
            0.0, 0.0, //
            0.0, 1.0, //
            1.0, 0.0, //
            1.0, 1.0, //
        ],
    );
    let output = Matrix::from_iter(4, 1, vec![0.0, 1.0, 1.0, 0.0]);

    println!("input = \n{input}");
    println!("output = \n{output}");

    let eps = 1e-1;
    let rate = 1e-1;

    let rng = StdRng::seed_from_u64(1);
    // let rng = rand::thread_rng();

    let mut nn = NeuralNetwork::from_iter(
        &[2, 2, 1],
        std::iter::repeat_with(|| rng.clone().gen_range(0.0..1.0)),
    );

    let mut gradient = NeuralNetwork::new(&[2, 2, 1]);

    println!("nn = {nn}");
    println!("cost = {:.32}", nn.cost(&input, &output));

    let mut count = 0;
    for i in 0..100_000 {
        nn.finite_diff(&mut gradient, &eps, &input, &output);
        nn.learn(&mut gradient, &rate);
        count += 1;
        if count % 100 == 0 {
            println!("i: {i} cost: {:.32}", nn.cost(&input, &output));
        }
    }

    println!("nn after training = {nn}");
    println!("cost: {:.32}", nn.cost(&input, &output));

    for row in 0..input.rows() {
        let res = nn.test(&input.get_row_matrix(row).unwrap());
        println!(
            "{row} ^ {col} = {res}",
            row = input.get(row, 0).unwrap(),
            col = input.get(row, 1).unwrap(),
            res = res.get(0, 0).unwrap()
        );
    }
}
