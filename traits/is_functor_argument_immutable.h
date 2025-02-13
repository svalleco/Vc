/*  This file is part of the Vc library. {{{
Copyright © 2014-2016 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_TRAITS_IS_FUNCTOR_ARGUMENT_IMMUTABLE_H_
#define VC_TRAITS_IS_FUNCTOR_ARGUMENT_IMMUTABLE_H_

namespace Vc_VERSIONED_NAMESPACE
{
namespace Traits
{
namespace is_functor_argument_immutable_impl
{
template <typename F, typename A> std::false_type test(void (F::*)(A &));
template <typename F, typename A> std::false_type test(void (F::*)(A &) const);
template <typename F, typename A> std:: true_type test(void (F::*)(const A &));
template <typename F, typename A> std:: true_type test(void (F::*)(const A &) const);
template <typename F, typename A> std:: true_type test(void (F::*)(const A &&));
template <typename F, typename A> std:: true_type test(void (F::*)(const A &&) const);
template <typename F, typename A> std:: true_type test(void (F::*)(const A));
template <typename F, typename A> std:: true_type test(void (F::*)(const A) const);
template <typename F, typename A> std:: true_type test(void (F::*)(A));
template <typename F, typename A> std:: true_type test(void (F::*)(A) const);

// This function is defined with a forwarding reference. Therefore it can also bind as T
// &, in which case the argument is mutable.
template <typename F, typename A> std::false_type test(void (F::*)(A &&));
template <typename F, typename A> std::false_type test(void (F::*)(A &&) const);

struct dummy {};

// generate a true_type for template operator() members in F that are callable with a
// 'const A &' argument even if the template parameter to operator() is fixed to 'A'.
template <
    typename F, typename A
#ifdef Vc_ICC
    // this ensures that F is a generic lambda. We can be pretty sure that noone wrote a
    // lambda with Vc::Traits::is_functor_argument_immutable_impl::dummy parameter
    // type. In theory, this is not needed because the return type fails with a
    // substitution failure in that case. Only ICC generates and error instead of doing
    // SFINAE.
    ,
    typename = decltype(std::declval<F &>()(std::declval<dummy &>()))
#endif
    >
decltype(is_functor_argument_immutable_impl::test(
    std::declval<decltype(&F::template operator() < A > )>()))
test2(int);

// generate a true_type for non-template operator() members in F that are callable with a
// 'const A &' argument.
template <typename F, typename A>
decltype(
    is_functor_argument_immutable_impl::test(std::declval<decltype(&F::operator())>()))
test2(float);

template <typename A> std::false_type test3(void (*)(A &));
template <typename A> std:: true_type test3(void (*)(const A &));
template <typename A> std:: true_type test3(void (*)(const A));
template <typename A> std:: true_type test3(void (*)(A));
template <typename A> std:: true_type test3(void (*)(A &&));

}  // namespace is_functor_argument_immutable_impl

template <typename F, typename A, bool = std::is_function<F>::value>
struct is_functor_argument_immutable;
template <typename F, typename A>
struct is_functor_argument_immutable<F, A, false>
    : public decltype(is_functor_argument_immutable_impl::test2<
                      typename std::remove_reference<F>::type, A>(int())) {
};
template <typename F, typename A>
struct is_functor_argument_immutable<F, A, true>
    : public decltype(is_functor_argument_immutable_impl::test3(std::declval<F>())) {
};

}  // namespace Traits
}  // namespace Vc

#endif  // VC_TRAITS_IS_FUNCTOR_ARGUMENT_IMMUTABLE_H_

// vim: foldmethod=marker
