use ark_std::{mem::ManuallyDrop, vec::Vec};

pub fn into_raw_parts<T>(vec: Vec<T>) -> (*mut T, usize, usize) {
    let mut me = ManuallyDrop::new(vec);
    (me.as_mut_ptr(), me.len(), me.capacity())
}
