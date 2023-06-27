use rand::prelude::*;
use rustml::Matrix;

fn main() {
    // let mut rng = rand::thread_rng();
    let mut rng = StdRng::seed_from_u64(1);

    let mut arr1 = Matrix::from_iter(3, 3, std::iter::repeat_with(|| rng.gen_range(0.0..1.0)));
    let arr2 = Matrix::from_iter(3, 3, std::iter::repeat_with(|| rng.gen_range(0.0..1.0)));

    let data =  vec![
        vec![0, 0, 0],
        vec![0, 1, 1],
        vec![1, 0, 1],
        vec![1, 1, 1],
    ];

    
    let eps = 1e-1;
    let rate = 1e-1;

    for _ in (0..1_000) {

    }

}
