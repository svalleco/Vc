/*  This file is part of the Vc library. {{{
Copyright © 2015 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_SCALAR_OPERATORS_H_
#define VC_SCALAR_OPERATORS_H_

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
// compare operators {{{1
#define Vc_OP(op_)                                                                       \
    template <typename T>                                                                \
    Vc_INTRINSIC Scalar::Mask<T> operator op_(Scalar::Vector<T> a, Scalar::Vector<T> b)  \
    {                                                                                    \
        return Scalar::Mask<T>(a.data() op_ b.data());                                   \
    }
Vc_ALL_COMPARES(Vc_OP)
#undef Vc_OP

// bitwise operators {{{1
#define Vc_OP(symbol)                                                                    \
    template <typename T>                                                                \
    Vc_INTRINSIC enable_if<std::is_integral<T>::value, Scalar::Vector<T>>                \
    operator symbol(Scalar::Vector<T> a, Scalar::Vector<T> b)                            \
    {                                                                                    \
        return a.data() symbol b.data();                                                 \
    }                                                                                    \
    template <typename T>                                                                \
    Vc_INTRINSIC enable_if<std::is_floating_point<T>::value, Scalar::Vector<T>>          \
    operator symbol(Scalar::Vector<T> &lhs, Scalar::Vector<T> rhs)                       \
    {                                                                                    \
        using uinta =                                                                    \
            MayAlias<typename std::conditional<sizeof(T) == sizeof(int), unsigned int,   \
                                               unsigned long long>::type>;               \
        uinta *left = reinterpret_cast<uinta *>(&lhs.data());                            \
        const uinta *right = reinterpret_cast<const uinta *>(&rhs.data());               \
        *left symbol## = *right;                                                         \
        return lhs;                                                                      \
    }
Vc_ALL_BINARY(Vc_OP)
#undef Vc_OP

// arithmetic operators {{{1
template <typename T>
Vc_INTRINSIC Scalar::Vector<T> operator+(Scalar::Vector<T> a, Scalar::Vector<T> b)
{
    return a.data() + b.data();
}
template <typename T>
Vc_INTRINSIC Scalar::Vector<T> operator-(Scalar::Vector<T> a, Scalar::Vector<T> b)
{
    return a.data() - b.data();
}
template <typename T>
Vc_INTRINSIC Scalar::Vector<T> operator*(Scalar::Vector<T> a, Scalar::Vector<T> b)
{
    return a.data() * b.data();
}
template <typename T>
Vc_INTRINSIC Scalar::Vector<T> operator/(Scalar::Vector<T> a, Scalar::Vector<T> b)
{
    return a.data() / b.data();
}
template <typename T>
Vc_INTRINSIC Scalar::Vector<T> operator%(Scalar::Vector<T> a, Scalar::Vector<T> b)
{
    return a.data() % b.data();
}
// }}}1
}  // namespace Detail
}  // namespace Vc

#endif  // VC_SCALAR_OPERATORS_H_

// vim: foldmethod=marker
