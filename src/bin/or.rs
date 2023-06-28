use rand::prelude::*;

fn main() {
    // let mut rng = rand::thread_rng();
    let mut rng = StdRng::seed_from_u64(1);

    // let mut arr1 = Matrix::from_iter(3, 3, std::iter::repeat_with(|| rng.gen_range(0.0..1.0)));
    // let arr2 = Matrix::from_iter(3, 3, std::iter::repeat_with(|| rng.gen_range(0.0..1.0)));

    let data = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];

    let eps = 1e-3;
    let rate = 1e-3;

    let mut w1 = rng.gen_range(0.0..1.0);
    let mut w2 = rng.gen_range(0.0..1.0);
    let mut b = rng.gen_range(0.0..1.0);

    let mut cost_res = cost(&w1, &w2, &b, &data);
    println!("w1: {w1}, w2: {w2}, b:{b} cost: {cost_res}");

    // let mut i = 0;
    for _ in 0..1000_000 {
        let dw1 = (cost(&(w1 + eps), &w2, &b, &data) - cost_res) / eps;
        let dw2 = (cost(&w1, &(w2 + eps), &b, &data) - cost_res) / eps;
        let db = (cost(&w1, &w2, &(b + eps), &data) - cost_res) / eps;

        w1 -= dw1 * rate;
        w2 -= dw2 * rate;
        b -= db * rate;

        cost_res = cost(&w1, &w2, &b, &data);

        // i += 1;
        // if i % 10 == 0 {
        //     println!("w1: {w1}, w2: {w2}, b:{b} cost: {cost_res}");
        // }
        // println!("w1: {w1}, w2: {w2}, b:{b} cost: {cost_res}");
    }

    for d in data {
        let x1 = d[0];
        let x2 = d[1];
        let y = d[2];
        let pred = sigmoid(x1 * w1 + x2 * w2 + b);
        println!("x1: {x1}, x2: {x2}, y: {y}, pred: {pred}");
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn cost(w1: &f64, w2: &f64, b: &f64, data: &Vec<Vec<f64>>) -> f64 {
    let mut result = 0.0;

    for i in 0..data.len() {
        let x1 = data[i][0];
        let x2 = data[i][1];

        let y = sigmoid(x1 * w1 + x2 * w2 + b);
        let d = y - data[i][2];
        result += d * d;
    }
    result /= data.len() as f64;

    result
}
