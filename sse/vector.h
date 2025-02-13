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

#ifndef VC_SSE_VECTOR_H_
#define VC_SSE_VECTOR_H_

#include "../scalar/vector.h"
#include "intrinsics.h"
#include "types.h"
#include "vectorhelper.h"
#include "mask.h"
#include "../common/writemaskedvector.h"
#include "../common/aliasingentryhelper.h"
#include "../common/memoryfwd.h"
#include "../common/loadstoreflags.h"
#include <algorithm>
#include <cmath>
#include "detail.h"

#include "macros.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

namespace Vc_VERSIONED_NAMESPACE
{

#define Vc_CURRENT_CLASS_NAME Vector
template <typename T> class Vector<T, VectorAbi::Sse>
{
    static_assert(std::is_arithmetic<T>::value,
                  "Vector<T> only accepts arithmetic builtin types as template parameter T.");

    protected:
#ifdef Vc_COMPILE_BENCHMARKS
    public:
#endif
        typedef typename SSE::VectorTraits<T>::StorageType StorageType;
        StorageType d;
        typedef typename SSE::VectorTraits<T>::GatherMaskType GatherMask;
        typedef SSE::VectorHelper<typename SSE::VectorTraits<T>::VectorType> HV;
        typedef SSE::VectorHelper<T> HT;
    public:
        Vc_FREE_STORE_OPERATORS_ALIGNED(16)

        typedef typename SSE::VectorTraits<T>::VectorType VectorType;
        using vector_type = VectorType;
        static constexpr size_t Size = SSE::VectorTraits<T>::Size;
        static constexpr size_t MemoryAlignment = alignof(VectorType);
        typedef typename SSE::VectorTraits<T>::EntryType EntryType;
        using value_type = EntryType;
        using VectorEntryType = EntryType;
        typedef typename std::conditional<(Size >= 4),
                                          SimdArray<int, Size, SSE::int_v, 4>,
                                          SimdArray<int, Size, Scalar::int_v, 1>>::type IndexType;
        typedef typename SSE::VectorTraits<T>::MaskType Mask;
        using MaskType = Mask;
        using mask_type = Mask;
        typedef typename Mask::Argument MaskArg;
        typedef typename Mask::Argument MaskArgument;
#ifdef Vc_PASSING_VECTOR_BY_VALUE_IS_BROKEN
        typedef const Vector &AsArg;
#else
        typedef const Vector AsArg;
#endif
        using abi = VectorAbi::Sse;
        using WriteMaskedVector = Common::WriteMaskedVector<Vector, Mask>;
        template <typename U> using V = Vector<U, abi>;

#include "../common/generalinterface.h"

        static Vc_INTRINSIC_L Vector Random() Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal: required to enable returning objects of VectorType
        Vc_ALWAYS_INLINE Vector(VectorType x) : d(x) {}

        // implict conversion from compatible Vector<U>
        template <typename U>
        Vc_INTRINSIC Vector(
            Vc_ALIGNED_PARAMETER(V<U>) x,
            typename std::enable_if<Traits::is_implicit_cast_allowed<U, T>::value,
                                    void *>::type = nullptr)
            : d(SSE::convert<U, T>(x.data()))
        {
        }

        // static_cast from the remaining Vector<U>
        template <typename U>
        Vc_INTRINSIC explicit Vector(
            Vc_ALIGNED_PARAMETER(V<U>) x,
            typename std::enable_if<!Traits::is_implicit_cast_allowed<U, T>::value,
                                    void *>::type = nullptr)
            : d(SSE::convert<U, T>(x.data()))
        {
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        Vc_INTRINSIC Vector(EntryType a) : d(HT::set(a)) {}
        template <typename U>
        Vc_INTRINSIC Vector(U a,
                            typename std::enable_if<std::is_same<U, int>::value &&
                                                        !std::is_same<U, EntryType>::value,
                                                    void *>::type = nullptr)
            : Vector(static_cast<EntryType>(a))
        {
        }

#include "../common/loadinterface.h"
#include "../common/storeinterface.h"

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        Vc_INTRINSIC_L void setZero() Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setZero(const Mask &k) Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setZeroInverted(const Mask &k) Vc_INTRINSIC_R;

        Vc_INTRINSIC_L void setQnan() Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setQnan(const Mask &k) Vc_INTRINSIC_R;

#include "../common/gatherinterface.h"
#include "../common/scatterinterface.h"

        //prefix
        Vc_INTRINSIC Vector &operator++() { data() = HT::add(data(), HT::one()); return *this; }
        Vc_INTRINSIC Vector &operator--() { data() = HT::sub(data(), HT::one()); return *this; }
        //postfix
        Vc_INTRINSIC Vector operator++(int) { const Vector r = *this; data() = HT::add(data(), HT::one()); return r; }
        Vc_INTRINSIC Vector operator--(int) { const Vector r = *this; data() = HT::sub(data(), HT::one()); return r; }

        Vc_INTRINSIC decltype(d.ref(0)) operator[](size_t index) { return d.ref(index); }
        Vc_INTRINSIC_L EntryType operator[](size_t index) const Vc_PURE Vc_INTRINSIC_R;

        Vc_INTRINSIC_L Vector operator[](SSE::int_v perm) const Vc_INTRINSIC_R;

        Vc_INTRINSIC Vc_PURE Mask operator!() const
        {
            return *this == Zero();
        }
        Vc_INTRINSIC Vc_PURE Vector operator~() const
        {
#ifndef Vc_ENABLE_FLOAT_BIT_OPERATORS
            static_assert(std::is_integral<T>::value,
                          "bit-complement can only be used with Vectors of integral type");
#endif
            return Detail::andnot_(data(), HV::allone());
        }
        Vc_ALWAYS_INLINE_L Vc_PURE_L Vector operator-() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
        Vc_INTRINSIC Vc_PURE Vector operator+() const { return *this; }

        Vc_INTRINSIC_L Vector &operator<<=(AsArg shift)       Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector  operator<< (AsArg shift) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector &operator<<=(  int shift)       Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector  operator<< (  int shift) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector &operator>>=(AsArg shift)       Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector  operator>> (AsArg shift) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector &operator>>=(  int shift)       Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector  operator>> (  int shift) const Vc_INTRINSIC_R;

        Vc_INTRINSIC Vc_PURE Vc_DEPRECATED("use isnegative(x) instead") Mask
            isNegative() const
        {
            return Vc::isnegative(*this);
        }

        Vc_ALWAYS_INLINE void assign(const Vector &v, const Mask &mask)
        {
            const VectorType k = SSE::sse_cast<VectorType>(mask.data());
            data() = HV::blend(data(), v.data(), k);
        }

        template <typename V2>
        Vc_ALWAYS_INLINE Vc_PURE
            Vc_DEPRECATED("Use simd_cast instead of Vector::staticCast") V2
            staticCast() const
        {
            return SSE::convert<T, typename V2::EntryType>(data());
        }
        template<typename V2> Vc_ALWAYS_INLINE Vc_PURE V2 reinterpretCast() const { return SSE::sse_cast<typename V2::VectorType>(data()); }

        Vc_INTRINSIC WriteMaskedVector operator()(const Mask &k) { return {this, k}; }

        Vc_ALWAYS_INLINE Vc_PURE VectorType &data() { return d.v(); }
        Vc_ALWAYS_INLINE Vc_PURE const VectorType &data() const { return d.v(); }

        template<int Index>
        Vc_INTRINSIC_L Vector broadcast() const Vc_INTRINSIC_R;

        Vc_INTRINSIC EntryType min() const { return HT::min(data()); }
        Vc_INTRINSIC EntryType max() const { return HT::max(data()); }
        Vc_INTRINSIC EntryType product() const { return HT::mul(data()); }
        Vc_INTRINSIC EntryType sum() const { return HT::add(data()); }
        Vc_INTRINSIC_L Vector partialSum() const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L EntryType min(MaskArg m) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L EntryType max(MaskArg m) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L EntryType product(MaskArg m) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L EntryType sum(MaskArg m) const Vc_INTRINSIC_R;

        Vc_INTRINSIC_L Vector shifted(int amount, Vector shiftIn) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector shifted(int amount) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector rotated(int amount) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vc_PURE_L Vector reversed() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_ALWAYS_INLINE_L Vc_PURE_L Vector sorted() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        template <typename F> void callWithValuesSorted(F &&f)
        {
            EntryType value = d.m(0);
            f(value);
            for (std::size_t i = 1; i < Size; ++i) {
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
            for(size_t i : where(mask)) {
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
            return Vc::exponent(*this);
        }

        Vc_INTRINSIC_L Vector interleaveLow(Vector x) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector interleaveHigh(Vector x) const Vc_INTRINSIC_R;
};
#undef Vc_CURRENT_CLASS_NAME
template <typename T> constexpr size_t Vector<T, VectorAbi::Sse>::Size;
template <typename T> constexpr size_t Vector<T, VectorAbi::Sse>::MemoryAlignment;

static Vc_ALWAYS_INLINE Vc_PURE SSE::int_v    min(const SSE::int_v    &x, const SSE::int_v    &y) { return SSE::min_epi32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::uint_v   min(const SSE::uint_v   &x, const SSE::uint_v   &y) { return SSE::min_epu32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::short_v  min(const SSE::short_v  &x, const SSE::short_v  &y) { return _mm_min_epi16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::ushort_v min(const SSE::ushort_v &x, const SSE::ushort_v &y) { return SSE::min_epu16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::float_v  min(const SSE::float_v  &x, const SSE::float_v  &y) { return _mm_min_ps(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::double_v min(const SSE::double_v &x, const SSE::double_v &y) { return _mm_min_pd(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::int_v    max(const SSE::int_v    &x, const SSE::int_v    &y) { return SSE::max_epi32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::uint_v   max(const SSE::uint_v   &x, const SSE::uint_v   &y) { return SSE::max_epu32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::short_v  max(const SSE::short_v  &x, const SSE::short_v  &y) { return _mm_max_epi16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::ushort_v max(const SSE::ushort_v &x, const SSE::ushort_v &y) { return SSE::max_epu16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::float_v  max(const SSE::float_v  &x, const SSE::float_v  &y) { return _mm_max_ps(x.data(), y.data()); }
static Vc_ALWAYS_INLINE Vc_PURE SSE::double_v max(const SSE::double_v &x, const SSE::double_v &y) { return _mm_max_pd(x.data(), y.data()); }

template <typename T,
          typename = enable_if<std::is_same<T, double>::value || std::is_same<T, float>::value ||
                               std::is_same<T, short>::value ||
                               std::is_same<T, int>::value>>
Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Sse> abs(Vector<T, VectorAbi::Sse> x)
{
    return SSE::VectorHelper<T>::abs(x.data());
}

  template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Sse> sqrt (const Vector<T, VectorAbi::Sse> &x) { return SSE::VectorHelper<T>::sqrt(x.data()); }
  template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Sse> rsqrt(const Vector<T, VectorAbi::Sse> &x) { return SSE::VectorHelper<T>::rsqrt(x.data()); }
  template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Sse> reciprocal(const Vector<T, VectorAbi::Sse> &x) { return SSE::VectorHelper<T>::reciprocal(x.data()); }
  template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Sse> round(const Vector<T, VectorAbi::Sse> &x) { return SSE::VectorHelper<T>::round(x.data()); }

  template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T, VectorAbi::Sse>::Mask isfinite(const Vector<T, VectorAbi::Sse> &x) { return SSE::VectorHelper<T>::isFinite(x.data()); }
  template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T, VectorAbi::Sse>::Mask isinf(const Vector<T, VectorAbi::Sse> &x) { return SSE::VectorHelper<T>::isInfinite(x.data()); }
  template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T, VectorAbi::Sse>::Mask isnan(const Vector<T, VectorAbi::Sse> &x) { return SSE::VectorHelper<T>::isNaN(x.data()); }

#define Vc_CONDITIONAL_ASSIGN(name_, op_)                                                \
    template <Operator O, typename T, typename M, typename U>                            \
    Vc_INTRINSIC enable_if<O == Operator::name_, void> conditional_assign(               \
        Vector<T, VectorAbi::Sse> &lhs, M &&mask, U &&rhs)                               \
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
    Vc_INTRINSIC enable_if<O == Operator::name_, Vector<T, VectorAbi::Sse>>              \
    conditional_assign(Vector<T, VectorAbi::Sse> &lhs, M &&mask)                         \
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

#endif // VC_SSE_VECTOR_H_
