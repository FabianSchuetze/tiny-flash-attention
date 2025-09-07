#pragma once

#include <cutlass/numeric_types.h>

#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"

using namespace cute;

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
          typename elem_type = cutlass::half_t>
struct Flash_kernel_traits {
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;

    using ElementAccum = float;
    using index_t = uint32_t;

    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
          typename elem_type = cutlass::half_t,
          typename Base = Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_,
                                              kNWarps_, elem_type>>
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;

    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = (kHeadDim % 128 == 0) ? 128 : kBlockKSmem;

    using TiledMma =
        TiledMMA<typename Base::MMA_Atom_Arch,
                 Layout<Shape<Int<kNWarps>, _1, _1>>,  // 4x1x1 or 8x1x1 thread
                 Tile<Int<16 * kNWarps>, _16, _16>>;

    using SmemLayoutAtomQO =
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>;

    using SmemLayoutQO = decltype(tile_to_shape(
        SmemLayoutAtomQO{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQO{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomVtransposed =
        Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>,
               Stride<_1, Int<kBlockKSmem>>>;
    using SmemLayoutVtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    //using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQOCount = size(SmemLayoutQO{});
    static constexpr int kSmemKVCount = size(SmemLayoutKV{}) * 2;
    static constexpr int kSmemQOSize = kSmemQOCount * sizeof(Element);
    static constexpr int kSmemKVSize = kSmemKVCount * sizeof(Element);
    // TODO:
    static constexpr int kSmemSize = kSmemQOSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad =
        sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0,
                  "kHeadDim must be a multiple of kGmemElemsPerLoad");

    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0,
                  "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<
        Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;

    using GmemTiledCopyQKVO = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{}, GmemLayoutAtom{},
        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 vals
                                                        // per read
};
