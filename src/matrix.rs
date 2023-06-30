mod ops;
use std::ops::Deref;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}
impl Matrix {
    pub fn new(row: usize, col: usize) -> Self {
        Self::from_iter(row, col, std::iter::repeat(0.0))
    }

    pub fn from_iter<I>(row: usize, col: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = f32>,
    {
        let data = iter.into_iter().take(col * row).collect::<Vec<_>>();
        assert_eq!(data.len(), row * col);
        Self {
            data,
            rows: row,
            cols: col,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&f32> {
        self.data.get(col + row * self.cols)
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f32> {
        self.data.get_mut(col + row * self.cols)
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) -> Option<f32> {
        let old = self.data.get(col + row * self.cols).map(|x| *x);
        self.data[col + row * self.cols] = value;
        old
    }

    pub fn get_row(&self, row: usize) -> Option<impl Iterator<Item = &f32>> {
        match row < self.rows {
            true => Some(self.data[row * self.cols..(row + 1) * self.cols].iter()),
            false => None,
        }
    }

    pub fn get_col(&self, col: usize) -> Option<impl Iterator<Item = &f32>> {
        match col < self.cols {
            true => Some((0..self.rows).map(move |row| self.get(row, col).unwrap())),
            false => None,
        }
    }

    pub fn get_row_matrix(&self, row: usize) -> Option<Self> {
        match self.get_row(row) {
            None => None,
            Some(data) => Some(Self {
                rows: 1,
                cols: self.cols,
                data: data.map(|x| *x).collect(),
            }),
        }
    }

    pub fn get_col_matrix(&self, row: usize) -> Option<Self> {
        match self.get_col(row) {
            None => None,
            Some(data) => Some(Self {
                rows: self.rows,
                cols: 1,
                data: data.map(|x| *x).collect(),
            }),
        }
    }

    pub fn sigmoid(&mut self) {
        for i in 0..self.data.len() {
            self.data[i] = 1.0 / (1.0 + (-self.data[i]).exp());
        }
    }

    pub fn relu(&mut self) {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i].max(0.0);
        }
    }

    pub fn fill(&mut self, value: f32) {
        for i in 0..self.data.len() {
            self.data[i] = value;
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        self.data.copy_from_slice(&other.data);
    }

    pub fn add_from(&mut self, other: &Self) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }

    pub fn dot_from(&mut self, a: &Self, b: &Self) {
        assert_eq!(self.rows, a.rows);
        assert_eq!(self.cols, b.cols);
        assert_eq!(a.cols, b.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                let mut sum = 0.0;
                for i in 0..a.cols {
                    sum += a.data[row * a.cols + i] * b.data[i * b.cols + col];
                }
                self.data[row * self.cols + col] = sum;
            }
        }
    }
}

impl Deref for Matrix {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{:.6} ", self.data[row * self.cols + col])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let m = Matrix::new(2, 2);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.get(0, 0), Some(&0.0));
        assert_eq!(m.get(0, 1), Some(&0.0));
        assert_eq!(m.get(1, 0), Some(&0.0));
        assert_eq!(m.get(1, 1), Some(&0.0));
    }

    #[test]
    fn test_from_iter() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let m = Matrix::from_iter(2, 2, data);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.get(0, 0), Some(&1.0));
        assert_eq!(m.get(0, 1), Some(&2.0));
        assert_eq!(m.get(1, 0), Some(&3.0));
        assert_eq!(m.get(1, 1), Some(&4.0));
    }

    #[test]
    fn test_get_set() {
        let mut m = Matrix::new(2, 2);
        assert_eq!(m.set(0, 0, 1.0), Some(0.0));
        assert_eq!(m.get(0, 0), Some(&1.0));
        assert_eq!(m.set(1, 1, 2.0), Some(0.0));
        assert_eq!(m.get(1, 1), Some(&2.0));
    }

    #[test]
    fn test_get_row() {
        let m = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            m.get_row(0).map(|row| row.collect::<Vec<_>>()),
            Some(vec![&1.0, &2.0])
        );
        assert_eq!(
            m.get_row(1).map(|row| row.collect::<Vec<_>>()),
            Some(vec![&3.0, &4.0])
        );
        assert_eq!(m.get_row(2).map(|row| row.collect::<Vec<_>>()), None);
    }

    #[test]
    fn test_get_col() {
        let m = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            m.get_col(0).map(|row| row.collect::<Vec<_>>()),
            Some(vec![&1.0, &3.0])
        );
        assert_eq!(
            m.get_col(1).map(|row| row.collect::<Vec<_>>()),
            Some(vec![&2.0, &4.0])
        );
        assert_eq!(m.get_col(2).map(|row| row.collect::<Vec<_>>()), None);
    }

    #[test]
    fn test_get_row_matrix() {
        let m = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            m.get_row_matrix(0),
            Some(Matrix::from_iter(1, 2, vec![1.0, 2.0]))
        );
        assert_eq!(
            m.get_row_matrix(1),
            Some(Matrix::from_iter(1, 2, vec![3.0, 4.0]))
        );
        assert_eq!(m.get_row_matrix(2), None);
    }

    #[test]
    fn test_get_col_matrix() {
        let m = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            m.get_col_matrix(0),
            Some(Matrix::from_iter(2, 1, vec![1.0, 3.0]))
        );
        assert_eq!(
            m.get_col_matrix(1),
            Some(Matrix::from_iter(2, 1, vec![2.0, 4.0]))
        );
        assert_eq!(m.get_col_matrix(2), None);
    }

    #[test]
    fn test_sigmoid() {
        let mut m = Matrix::from_iter(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
        m.sigmoid();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.get(0, 0), Some(&0.5));
        assert_eq!(m.get(0, 1), Some(&0.7310585786300049));
        assert_eq!(m.get(1, 0), Some(&0.8807970779778823));
        assert_eq!(m.get(1, 1), Some(&0.9525741268224334));
    }
}
