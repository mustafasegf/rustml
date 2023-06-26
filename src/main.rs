use rustml::Matrix;

fn main() {
    let mut arr1 = Matrix::from_iter(3, 3, 0..9);
    let arr2 = Matrix::from_iter(3, 3, 10..);

    let arr3 = &arr1 * &arr2;
    arr1 *= &arr2;

    println!("{}", arr1);
    println!("{}", arr3);
}
