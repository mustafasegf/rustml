use rand::prelude::*;

fn main() {
    // let mut rng = rand::thread_rng();
    let mut rng = StdRng::seed_from_u64(1);

    // let mut arr1 = Matrix::from_iter(3, 3, std::iter::repeat_with(|| rng.gen_range(0.0..1.0)));
    // let arr2 = Matrix::from_iter(3, 3, std::iter::repeat_with(|| rng.gen_range(0.0..1.0)));

    let data = vec![
        vec![0.0, 0.0],
        vec![1.0, 2.0],
        vec![2.0, 4.0],
        vec![3.0, 6.0],
        vec![4.0, 8.0],
        vec![5.0, 10.0],
        vec![6.0, 12.0],
    ];

    let eps = 1e-3;
    let rate = 1e-3;

    let mut w = rng.gen_range(0.0..10.0);
    let mut b = rng.gen_range(0.0..5.0);

    let mut cost_res = cost(&w, &b, &data);
    println!("w: {w}, b: {b}, cost: {cost_res}");

    for _ in 0..100_000 {
        let dw = (cost(&(w + eps), &b, &data) - cost_res) / eps;
        let db = (cost(&w, &(b + eps), &data) - cost_res) / eps;

        w -= dw * rate;
        b -= db * rate;

        cost_res = cost(&w, &b, &data);

        println!("w: {w}, b: {b}, cost: {cost_res}");
    }
}

fn cost(w: &f64, b: &f64, data: &Vec<Vec<f64>>) -> f64 {
    let mut result = 0.0;

    for i in 0..data.len() {
        let x = data[i][0];

        let y = x * w + b;
        let d = y - data[i][1];
        result += d * d;
    }
    result /= data.len() as f64;

    result
}
