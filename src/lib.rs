// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor and memref related functionality, including types and ops.

/// Implement [SideEffects](pliron::opts::dce::SideEffects) with
/// [has_self_side_effects](pliron::opts::dce::SideEffects::has_self_side_effects)
/// returning `false`.
macro_rules! impl_no_self_side_effects {
    ($($op:ty),+ $(,)?) => {
        $(
            #[pliron::derive::op_interface_impl]
            impl pliron::opts::dce::SideEffects for $op {
                fn has_self_side_effects(&self, _ctx: &pliron::context::Context) -> bool {
                    false
                }
            }
        )+
    };
}

pub mod memref;
pub mod tensor;
