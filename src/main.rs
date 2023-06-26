use rustml::Matrix;

fn main() {
    let arr1 = Matrix::from_iter(3, 3, 0..9);
    let arr2 = Matrix::from_iter(3, 3, 10..);
    println!("{}", arr1);
    let arr3 = &arr1 + &arr2;
    println!("{}", arr3);
}