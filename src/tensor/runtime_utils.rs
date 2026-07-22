// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Types and utilities to interact with the tensor dialect from Rust

/// Represents a tensor descriptor in Rust.
/// Provides conversion to/from the IR tensor descriptor.
/// and retrieve their types and descriptors for use in IR generation.
#[derive(Debug)]
pub struct TensorDesciptor {
    allocated_ptr: *const u8,
    aligned_ptr: *const u8,
    offset: usize,
    sizes: Vec<usize>,
    strides: Vec<usize>,
    /// Size of each element, not part of IR descriptor.
    elem_size: usize,
}

impl TensorDesciptor {
    /// Create a new tensor descriptor with the inputs.
    /// The allocated memory will be uninitialized, so the caller must ensure that it is properly initialized before use.
    pub fn new(dims: Vec<usize>, elem_size: usize, elems_ptr: *const u8) -> Self {
        let mut strides = vec![0; dims.len()];
        strides[dims.len() - 1] = 1;
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        Self {
            allocated_ptr: elems_ptr,
            aligned_ptr: elems_ptr,
            offset: 0,
            sizes: dims,
            strides,
            elem_size,
        }
    }

    /// Get the allocated pointer.
    pub fn allocated_ptr(&self) -> *const u8 {
        self.allocated_ptr
    }

    /// Get the aligned pointer.
    pub fn aligned_ptr(&self) -> *const u8 {
        self.aligned_ptr
    }

    /// Get the offset (in elements) of the tensor.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the sizes of the tensor.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Get the strides of the tensor.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the total number of elements in the tensor.
    /// This is calculated as the product of the sizes of all dimensions.
    pub fn num_elements(&self) -> usize {
        self.sizes
            .iter()
            .cloned()
            .reduce(|total_size, dim_size| total_size * dim_size)
            .unwrap()
    }

    /// Get the element size of the tensor.
    pub fn elem_size(&self) -> usize {
        self.elem_size
    }

    /// Get the total size of the tensor in bytes.
    /// This is calculated as the number of elements multiplied by the size of the element type.
    pub fn total_size_in_bytes(&self) -> usize {
        self.num_elements() * self.elem_size()
    }

    /// Get a tensor's IR descriptor.
    pub fn build_ir_descriptor(&self) -> Vec<u8> {
        // The descriptor will contain the allocated pointer, aligned pointer, offset, sizes, and strides.
        let mut descriptor = Vec::new();
        descriptor.extend_from_slice(&(self.allocated_ptr as usize).to_ne_bytes());
        descriptor.extend_from_slice(&(self.aligned_ptr as usize).to_ne_bytes());
        descriptor.extend_from_slice(&self.offset.to_ne_bytes());
        for &size in &self.sizes {
            descriptor.extend_from_slice(&size.to_ne_bytes());
        }
        for &stride in &self.strides {
            descriptor.extend_from_slice(&stride.to_ne_bytes());
        }
        descriptor
    }

    /// Create a tensor descriptor from its IR equivalent.
    /// # Safety
    /// The caller must ensure that `descriptor` is correctly formatted and that
    /// the rank and element size are accurate. No additional validation is performed.
    pub unsafe fn from_ir_descriptor(descriptor: *const u8, rank: usize, elem_size: usize) -> Self {
        unsafe {
            let allocated_ptr = std::ptr::read_unaligned(descriptor as *const usize) as *mut u8;
            let aligned_ptr = std::ptr::read_unaligned(
                descriptor.add(std::mem::size_of::<usize>()) as *const usize
            ) as *mut u8;
            let offset = std::ptr::read_unaligned(
                descriptor.add(2 * std::mem::size_of::<usize>()) as *const usize
            );
            let mut sizes = Vec::with_capacity(rank);
            let mut strides = Vec::with_capacity(rank);
            for i in 0..rank {
                sizes.push(std::ptr::read_unaligned(
                    descriptor.add((3 + i) * std::mem::size_of::<usize>()) as *const usize,
                ));
            }
            for i in 0..rank {
                strides.push(std::ptr::read_unaligned(
                    descriptor.add((3 + rank + i) * std::mem::size_of::<usize>()) as *const usize,
                ));
            }
            Self {
                allocated_ptr,
                aligned_ptr,
                offset,
                sizes,
                strides,
                elem_size,
            }
        }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        self.sizes.len()
    }

    fn delinearize_index(&self, mut linear_idx: usize) -> Vec<usize> {
        let rank = self.rank();
        let mut idxs = vec![0; rank];
        for d in (0..rank).rev() {
            let dim = self.sizes[d];
            idxs[d] = linear_idx % dim;
            linear_idx /= dim;
        }
        idxs
    }

    fn strided_offset(&self, idxs: &[usize]) -> usize {
        self.offset
            + idxs
                .iter()
                .zip(self.strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>()
    }

    /// Copy all tensor elements into `dst` in row-major logical order.
    ///
    /// This handles arbitrary strides in the descriptor. `dst` is cleared and
    /// resized to `self.num_elements()`.
    ///
    /// # Safety
    /// The descriptor pointers must reference valid memory for reading all
    /// elements addressed by `sizes`, `strides`, and `offset`.
    pub unsafe fn copy_to_vec<T: Copy>(&self, dst: &mut Vec<T>) {
        assert_eq!(std::mem::size_of::<T>(), self.elem_size);
        let total = self.num_elements();
        dst.clear();
        dst.reserve(total);

        // Note: this allocates an index Vec per element; correct but not optimised.
        // Rewrite iteratively if this is ever on a hot path.
        let base_ptr = self.aligned_ptr as *const T;
        for lin in 0..total {
            let idxs = self.delinearize_index(lin);
            let off = self.strided_offset(&idxs);
            // SAFETY: Caller guarantees descriptor memory validity.
            dst.push(unsafe { *base_ptr.add(off) });
        }
    }

    /// Copy tensor elements from `src` (row-major logical order) into the
    /// descriptor-backed memory, honoring descriptor strides.
    ///
    /// # Safety
    /// The descriptor pointers must reference valid writable memory for all
    /// elements addressed by `sizes`, `strides`, and `offset`.
    pub unsafe fn copy_from_slice<T: Copy>(&self, src: &[T]) {
        assert_eq!(std::mem::size_of::<T>(), self.elem_size);
        assert_eq!(src.len(), self.num_elements());

        let base_ptr = self.aligned_ptr as *mut T;
        for (lin, value) in src.iter().copied().enumerate() {
            let idxs = self.delinearize_index(lin);
            let off = self.strided_offset(&idxs);
            // SAFETY: Caller guarantees descriptor memory validity.
            unsafe { *base_ptr.add(off) = value };
        }
    }
}
