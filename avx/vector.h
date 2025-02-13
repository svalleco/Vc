/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_AVX_VECTOR_H_
#define VC_AVX_VECTOR_H_

#include "intrinsics.h"
#include "casts.h"
#include "../sse/vector.h"
#include "shuffle.h"
#include "vectorhelper.h"
#include "mask.h"
#include <algorithm>
#include <cmath>
#include "../common/aliasingentryhelper.h"
#include "../common/memoryfwd.h"
#include "../common/where.h"
#include "macros.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
template <typename T, typename Abi> struct VectorTraits
{
    using mask_type = Vc::Mask<T, Abi>;
    using vector_type = Vc::Vector<T, Abi>;
    using writemasked_vector_type = Common::WriteMaskedVector<vector_type, mask_type>;
    using intrinsic_type = typename AVX::VectorTypeHelper<T>::Type;
};
}  // namespace Detail

#define Vc_CURRENT_CLASS_NAME Vector
template <typename T> class Vector<T, VectorAbi::Avx>
{
public:
    using abi = VectorAbi::Avx;

private:
    using traits_type = Detail::VectorTraits<T, abi>;
    static_assert(
        std::is_arithmetic<T>::value,
        "Vector<T> only accepts arithmetic builtin types as template parameter T.");

    using WriteMaskedVector = typename traits_type::writemasked_vector_type;

public:
    using VectorType = typename traits_type::intrinsic_type;
    using vector_type = VectorType;

    using mask_type = typename traits_type::mask_type;
    using Mask = mask_type;
    using MaskType = mask_type;
    using MaskArg Vc_DEPRECATED("Use MaskArgument instead.") = typename Mask::AsArg;
    using MaskArgument = typename Mask::AsArg;

    Vc_FREE_STORE_OPERATORS_ALIGNED(alignof(VectorType))

    Vc_ALIGNED_TYPEDEF(sizeof(T), T, EntryType);
        using value_type = EntryType;
        typedef EntryType VectorEntryType;
        static constexpr size_t Size = sizeof(VectorType) / sizeof(EntryType);
        static constexpr size_t MemoryAlignment = alignof(VectorType);
        enum Constants {
            HasVectorDivision = AVX::HasVectorDivisionHelper<T>::Value
        };
#ifdef Vc_IMPL_AVX2
        typedef typename std::conditional<
            (Size >= 8), SimdArray<int, Size, AVX2::int_v, 8>,
            typename std::conditional<(Size >= 4), SimdArray<int, Size, SSE::int_v, 4>,
                                      SimdArray<int, Size, Scalar::int_v, 1>>::type>::type
            IndexType;
#else
        typedef typename std::conditional<(Size >= 4),
                                          SimdArray<int, Size, SSE::int_v, 4>,
                                          SimdArray<int, Size, Scalar::int_v, 1>>::type IndexType;
#endif
#ifdef Vc_PASSING_VECTOR_BY_VALUE_IS_BROKEN
        typedef const Vector<T, abi> &AsArg;
        typedef const VectorType &VectorTypeArg;
#else
        typedef Vector<T, abi> AsArg;
        typedef VectorType VectorTypeArg;
#endif

    protected:
        template <typename U> using V = Vector<U, abi>;

        // helper that specializes on VectorType
        typedef AVX::VectorHelper<VectorType> HV;

        // helper that specializes on T
        typedef AVX::VectorHelper<T> HT;

        // cast any m256/m128 to VectorType
        template <typename V>
        static Vc_INTRINSIC VectorType _cast(Vc_ALIGNED_PARAMETER(V) v)
        {
            return AVX::avx_cast<VectorType>(v);
        }

        typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
        StorageType d;

        using WidthT = Common::WidthT<VectorType>;
        // ICC can't compile this:
        // static constexpr WidthT Width = WidthT();

    public:
#include "../common/generalinterface.h"

        static Vc_ALWAYS_INLINE_L Vector Random() Vc_ALWAYS_INLINE_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal: required to enable returning objects of VectorType
        Vc_ALWAYS_INLINE Vector(VectorTypeArg x) : d(x) {}

        // implict conversion from compatible Vector<U, abi>
        template <typename U>
        Vc_INTRINSIC Vector(
            Vc_ALIGNED_PARAMETER(V<U>) x,
            typename std::enable_if<Traits::is_implicit_cast_allowed<U, T>::value,
                                    void *>::type = nullptr)
            : d(AVX::convert<U, T>(x.data()))
        {
        }

        // static_cast from the remaining Vector<U, abi>
        template <typename U>
        Vc_INTRINSIC explicit Vector(
            Vc_ALIGNED_PARAMETER(V<U>) x,
            typename std::enable_if<!Traits::is_implicit_cast_allowed<U, T>::value,
                                    void *>::type = nullptr)
            : d(Detail::zeroExtendIfNeeded(AVX::convert<U, T>(x.data())))
        {
        }

        // static_cast from other types, implemented via the non-member simd_cast function in
        // simd_cast_caller.tcc
        template <typename U,
                  typename = enable_if<Traits::is_simd_vector<U>::value &&
                                       !std::is_same<Vector, Traits::decay<U>>::value>>
        Vc_INTRINSIC_L explicit Vector(U &&x) Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        Vc_INTRINSIC Vector(EntryType a) : d(Detail::avx_broadcast(a)) {}
        template <typename U>
        Vc_INTRINSIC Vector(U a,
                            typename std::enable_if<std::is_same<U, int>::value &&
                                                        !std::is_same<U, EntryType>::value,
                                                    void *>::type = nullptr)
            : Vector(static_cast<EntryType>(a))
        {
        }

        //template<typename U>
        explicit Vector(std::initializer_list<EntryType>)
        {
            static_assert(std::is_same<EntryType, void>::value,
                          "A SIMD vector object cannot be initialized from an initializer list "
                          "because the number of entries in the vector is target-dependent.");
        }

#include "../common/loadinterface.h"
#include "../common/storeinterface.h"

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        Vc_INTRINSIC_L void setZero() Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setZero(const Mask &k) Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setZeroInverted(const Mask &k) Vc_INTRINSIC_R;

        Vc_INTRINSIC_L void setQnan() Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setQnan(MaskArg k) Vc_INTRINSIC_R;

#include "../common/gatherinterface.h"
#include "../common/scatterinterface.h"

        ///////////////////////////////////////////////////////////////////////////////////////////
        //prefix
        Vc_ALWAYS_INLINE Vector &operator++() { data() = Detail::add(data(), Detail::one(T()), T()); return *this; }
        Vc_ALWAYS_INLINE Vector &operator--() { data() = Detail::sub(data(), Detail::one(T()), T()); return *this; }
        //postfix
        Vc_ALWAYS_INLINE Vector operator++(int) { const Vector r = *this; data() = Detail::add(data(), Detail::one(T()), T()); return r; }
        Vc_ALWAYS_INLINE Vector operator--(int) { const Vector r = *this; data() = Detail::sub(data(), Detail::one(T()), T()); return r; }

        Vc_INTRINSIC decltype(d.ref(0)) operator[](size_t index) { return d.ref(index); }
        Vc_ALWAYS_INLINE EntryType operator[](size_t index) const {
            return d.m(index);
        }

        Vc_INTRINSIC_L Vc_PURE_L Vector operator[](Permutation::ReversedTag) const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L Vector operator[](const IndexType &perm) const Vc_INTRINSIC_R Vc_PURE_R;

        Vc_INTRINSIC Vc_PURE Mask operator!() const
        {
            return *this == Zero();
        }
        Vc_ALWAYS_INLINE Vector operator~() const
        {
#ifndef Vc_ENABLE_FLOAT_BIT_OPERATORS
            static_assert(std::is_integral<T>::value,
                          "bit-complement can only be used with Vectors of integral type");
#endif
            return Detail::andnot_(data(), Detail::allone<VectorType>());
        }
        Vc_ALWAYS_INLINE_L Vc_PURE_L Vector operator-() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
        Vc_INTRINSIC Vc_PURE Vector operator+() const { return *this; }

        // shifts
#define Vc_OP_VEC(op)                                                                    \
    Vc_INTRINSIC Vector &operator op##=(AsArg x)                                         \
    {                                                                                    \
        static_assert(                                                                   \
            std::is_integral<T>::value,                                                  \
            "bitwise-operators can only be used with Vectors of integral type");         \
    }                                                                                    \
    Vc_INTRINSIC Vc_PURE Vector operator op(AsArg x) const                               \
    {                                                                                    \
        static_assert(                                                                   \
            std::is_integral<T>::value,                                                  \
            "bitwise-operators can only be used with Vectors of integral type");         \
    }
    Vc_ALL_SHIFTS(Vc_OP_VEC)
#undef Vc_OP_VEC

        Vc_ALWAYS_INLINE_L Vector &operator>>=(int x) Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Vector &operator<<=(int x) Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Vector operator>>(int x) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Vector operator<<(int x) const Vc_ALWAYS_INLINE_R;

        Vc_INTRINSIC Vc_PURE Vc_DEPRECATED("use isnegative(x) instead") Mask
            isNegative() const
        {
            return Vc::isnegative(*this);
        }

        Vc_ALWAYS_INLINE void assign( const Vector &v, const Mask &mask ) {
            const VectorType k = _cast(mask.data());
            data() = Detail::blend(data(), v.data(), k);
        }

        template <typename V2>
        Vc_ALWAYS_INLINE Vc_DEPRECATED("Use simd_cast instead of Vector::staticCast") V2
            staticCast() const
        {
            return V2(*this);
        }
        template<typename V2> Vc_ALWAYS_INLINE V2 reinterpretCast() const { return AVX::avx_cast<typename V2::VectorType>(data()); }

        Vc_ALWAYS_INLINE WriteMaskedVector operator()(const Mask &k)
        {
            return {this, k};
        }

        Vc_ALWAYS_INLINE VectorType &data() { return d.v(); }
        Vc_ALWAYS_INLINE const VectorType &data() const { return d.v(); }

        template<int Index>
        Vc_INTRINSIC_L Vector broadcast() const Vc_INTRINSIC_R;

        Vc_INTRINSIC_L std::pair<Vector, int> minIndex() const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L std::pair<Vector, int> maxIndex() const Vc_INTRINSIC_R;

        Vc_ALWAYS_INLINE EntryType min() const { return Detail::min(data(), T()); }
        Vc_ALWAYS_INLINE EntryType max() const { return Detail::max(data(), T()); }
        Vc_ALWAYS_INLINE EntryType product() const { return Detail::mul(data(), T()); }
        Vc_ALWAYS_INLINE EntryType sum() const { return Detail::add(data(), T()); }
        Vc_ALWAYS_INLINE_L Vector partialSum() const Vc_ALWAYS_INLINE_R;
        //template<typename BinaryOperation> Vc_ALWAYS_INLINE_L Vector partialSum(BinaryOperation op) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType min(MaskArg m) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType max(MaskArg m) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType product(MaskArg m) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType sum(MaskArg m) const Vc_ALWAYS_INLINE_R;

        Vc_INTRINSIC_L Vector shifted(int amount, Vector shiftIn) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector shifted(int amount) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector rotated(int amount) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vc_PURE_L Vector reversed() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_ALWAYS_INLINE_L Vc_PURE_L Vector sorted() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        template <typename F> void callWithValuesSorted(F &&f)
        {
            EntryType value = d.m(0);
            f(value);
            for (size_t i = 1; i < Size; ++i) {
                if (d.m(i) != value) {
                    value = d.m(i);
                    f(value);
                }
            }
        }

        template <typename F> Vc_INTRINSIC void call(F &&f) const
        {
            Common::for_all_vector_entries<Size>([&](size_t i) { f(EntryType(d.m(i))); });
        }

        template <typename F> Vc_INTRINSIC void call(F &&f, const Mask &mask) const
        {
            for (size_t i : where(mask)) {
                f(EntryType(d.m(i)));
            }
        }

        template <typename F> Vc_INTRINSIC Vector apply(F &&f) const
        {
            Vector r;
            Common::for_all_vector_entries<Size>(
                [&](size_t i) { r.d.set(i, f(EntryType(d.m(i)))); });
            return r;
        }

        template <typename F> Vc_INTRINSIC Vector apply(F &&f, const Mask &mask) const
        {
            Vector r(*this);
            for (size_t i : where(mask)) {
                r.d.set(i, f(EntryType(r.d.m(i))));
            }
            return r;
        }

        template<typename IndexT> Vc_INTRINSIC void fill(EntryType (&f)(IndexT)) {
            Common::for_all_vector_entries<Size>([&](size_t i) { d.set(i, f(i)); });
        }
        Vc_INTRINSIC void fill(EntryType (&f)()) {
            Common::for_all_vector_entries<Size>([&](size_t i) { d.set(i, f()); });
        }

        template <typename G> static Vc_INTRINSIC_L Vector generate(G gen) Vc_INTRINSIC_R;

        Vc_INTRINSIC Vc_DEPRECATED("use copysign(x, y) instead") Vector
            copySign(AsArg reference) const
        {
            return Vc::copysign(*this, reference);
        }

        Vc_INTRINSIC Vc_DEPRECATED("use exponent(x) instead") Vector exponent() const
        {
            Vc::exponent(*this);
        }

        Vc_INTRINSIC_L Vector interleaveLow(Vector x) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector interleaveHigh(Vector x) const Vc_INTRINSIC_R;
};
#undef Vc_CURRENT_CLASS_NAME
template <typename T> constexpr size_t Vector<T, VectorAbi::Avx>::Size;
template <typename T> constexpr size_t Vector<T, VectorAbi::Avx>::MemoryAlignment;

static_assert(Traits::is_simd_vector<AVX2::double_v>::value, "is_simd_vector<double_v>::value");
static_assert(Traits::is_simd_vector<AVX2:: float_v>::value, "is_simd_vector< float_v>::value");
static_assert(Traits::is_simd_vector<AVX2::   int_v>::value, "is_simd_vector<   int_v>::value");
static_assert(Traits::is_simd_vector<AVX2::  uint_v>::value, "is_simd_vector<  uint_v>::value");
static_assert(Traits::is_simd_vector<AVX2:: short_v>::value, "is_simd_vector< short_v>::value");
static_assert(Traits::is_simd_vector<AVX2::ushort_v>::value, "is_simd_vector<ushort_v>::value");
static_assert(Traits::is_simd_mask  <AVX2::double_m>::value, "is_simd_mask  <double_m>::value");
static_assert(Traits::is_simd_mask  <AVX2:: float_m>::value, "is_simd_mask  < float_m>::value");
static_assert(Traits::is_simd_mask  <AVX2::   int_m>::value, "is_simd_mask  <   int_m>::value");
static_assert(Traits::is_simd_mask  <AVX2::  uint_m>::value, "is_simd_mask  <  uint_m>::value");
static_assert(Traits::is_simd_mask  <AVX2:: short_m>::value, "is_simd_mask  < short_m>::value");
static_assert(Traits::is_simd_mask  <AVX2::ushort_m>::value, "is_simd_mask  <ushort_m>::value");

#ifdef Vc_IMPL_AVX2
static_assert(!std::is_convertible<float *, AVX2::short_v>::value, "A float* should never implicitly convert to short_v. Something is broken.");
static_assert(!std::is_convertible<int *  , AVX2::short_v>::value, "An int* should never implicitly convert to short_v. Something is broken.");
static_assert(!std::is_convertible<short *, AVX2::short_v>::value, "A short* should never implicitly convert to short_v. Something is broken.");
#endif

#define Vc_CONDITIONAL_ASSIGN(name_, op_)                                                \
    template <Operator O, typename T, typename M, typename U>                            \
    Vc_INTRINSIC enable_if<O == Operator::name_, void> conditional_assign(               \
        AVX2::Vector<T> &lhs, M &&mask, U &&rhs)                                         \
    {                                                                                    \
        lhs(mask) op_ rhs;                                                               \
    }
Vc_CONDITIONAL_ASSIGN(          Assign,  =)
Vc_CONDITIONAL_ASSIGN(      PlusAssign, +=)
Vc_CONDITIONAL_ASSIGN(     MinusAssign, -=)
Vc_CONDITIONAL_ASSIGN(  MultiplyAssign, *=)
Vc_CONDITIONAL_ASSIGN(    DivideAssign, /=)
Vc_CONDITIONAL_ASSIGN( RemainderAssign, %=)
Vc_CONDITIONAL_ASSIGN(       XorAssign, ^=)
Vc_CONDITIONAL_ASSIGN(       AndAssign, &=)
Vc_CONDITIONAL_ASSIGN(        OrAssign, |=)
Vc_CONDITIONAL_ASSIGN( LeftShiftAssign,<<=)
Vc_CONDITIONAL_ASSIGN(RightShiftAssign,>>=)
#undef Vc_CONDITIONAL_ASSIGN

#define Vc_CONDITIONAL_ASSIGN(name_, expr_)                                              \
    template <Operator O, typename T, typename M>                                        \
    Vc_INTRINSIC enable_if<O == Operator::name_, AVX2::Vector<T>> conditional_assign(    \
        AVX2::Vector<T> &lhs, M &&mask)                                                  \
    {                                                                                    \
        return expr_;                                                                    \
    }
Vc_CONDITIONAL_ASSIGN(PostIncrement, lhs(mask)++)
Vc_CONDITIONAL_ASSIGN( PreIncrement, ++lhs(mask))
Vc_CONDITIONAL_ASSIGN(PostDecrement, lhs(mask)--)
Vc_CONDITIONAL_ASSIGN( PreDecrement, --lhs(mask))
#undef Vc_CONDITIONAL_ASSIGN

}  // namespace Vc

#include "vector.tcc"
#include "simd_cast.h"

#endif // VC_AVX_VECTOR_H_
