/*
    drjit/vcall.h -- Vectorized method call support, via horiz. reduction

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)


template <size_t I, size_t N, typename T, typename UInt32>
DRJIT_INLINE decltype(auto) gather_helper(const T& value, const UInt32 &perm) {
    if constexpr (is_mask_v<T> && I == N - 1) {
        return true;
    } else if constexpr (is_jit_v<T>) {
        return gather<T, true>(value, perm);
    } else if constexpr (is_drjit_struct_v<T>) {
        T result = value;
        struct_support_t<T>::apply_1(
            result, [&perm](auto &x) { x = gather_helper<1, 1>(x, perm); });
        return result;
    } else {
        DRJIT_MARK_USED(perm);
        return value;
    }
}

// Helper template to check if a type is a pointer to a template type
template <template <typename...> class Template, typename T>
struct is_pointer_to_template : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct is_pointer_to_template<Template, Template<Args...> const * __ptr64> : std::true_type {
};

template <typename Result, typename Func, typename Self, size_t... Is,
          typename... Args>
Result vcall_jit_reduce_impl(Func func, const Self &self_,
                             std::index_sequence<Is...>, const Args &...args) {
    using UInt32 = uint32_array_t<Self>;
    using Class = scalar_t<Self>;
    using Mask = mask_t<UInt32>;
    static constexpr JitBackend Backend = detached_t<Mask>::Backend;
    constexpr size_t N = sizeof...(Args);
    DRJIT_MARK_USED(N);

    schedule(args...);

    size_t self_size = self_.size();
    if (self_size == 1) {
        auto self = self_.entry(0);
        if (self)
            return func(self, args...);
        else
            return zeros<Result>();
    }

    Mask mask = extract_mask<Mask>(args...);
    size_t mask_size = mask.size();

    // Apply mask stack
    mask = Mask::steal(jit_var_mask_apply(mask.index(),
        (uint32_t) (self_size > mask_size ? self_size : mask_size)
    ));

    struct SetSelfHelper {
        void set(uint32_t value, uint32_t index) {
            jit_vcall_set_self(detached_t<Mask>::Backend, value, index);
        }

        ~SetSelfHelper() {
            jit_vcall_set_self(detached_t<Mask>::Backend, 0, 0);
        }
    };

    Self self = self_ & mask;
    auto [buckets, n_inst] = self.vcall_();

//#define DEBUG_PRINT
#if defined(DEBUG_PRINT)
    if constexpr (is_pointer_to_template<mitsuba::BSDF, Class>{}) {
        if (!strcmp(typeid(Func).name(),
                    "class <lambda_a3166cd1409662f7932530c96e6ad55e>")) {
            fprintf(stderr, "In vcall_jit_reduce.h:\n");
            fprintf(stderr, "Result = %s\n", typeid(Result).name());
            fprintf(stderr, "Func = %s\n", typeid(Func).name());
            fprintf(stderr, "Self = %s\n", typeid(Self).name());
            fprintf(stderr, "UInt32 = %s\n", typeid(UInt32).name());
            fprintf(stderr, "Class = %s\n", typeid(Class).name());
            fprintf(stderr, "Mask = %s\n", typeid(Mask).name());
            fprintf(stderr, "self_size = %d\n", (int) self_size);
            fprintf(stderr, "mask_size = %d\n", (int) mask_size);
            fprintf(stderr, "n_inst = %d\n", n_inst);
            fprintf(stderr, "N = sizeof...(Args) = %d\n", (int) N);
            fprintf(stderr, "Args = ");
            (fprintf(stderr, "%s, \n", typeid(Args).name()), ...);
            fprintf(stderr, "\n");
            printf_async(Mask(true), "self = %u\n", self);
        }
    }
#endif

    Result result;
    SetSelfHelper self_helper;
    if (n_inst > 0 && self_size > 0) {
        result = empty<Result>(self_size);
        size_t last_size = 0;

        for (size_t i = 0; i < n_inst ; ++i) {
            UInt32 perm = UInt32::borrow(buckets[i].index);
            size_t wavefront_size = perm.size();

            MaskScope<Mask> scope(Mask::steal(
                jit_var_mask_default(Backend, (uint32_t) wavefront_size)));

            UInt32 instance_id = gather<UInt32>(self, perm);

            // Avoid merging multiple vcall launches if size repeats..
            if (wavefront_size != last_size)
                last_size = wavefront_size;
            else
                eval(result);

            if (buckets[i].ptr) {
                self_helper.set(buckets[i].id, instance_id.index());

                if constexpr (!std::is_same_v<Result, std::nullptr_t>) {
#if defined(DEBUG_PRINT)
                    if constexpr (is_pointer_to_template<mitsuba::BSDF, Class>{}) {
                        if (!strcmp(
                                typeid(Func).name(),
                                "class "
                                "<lambda_a3166cd1409662f7932530c96e6ad55e>")) {
                            fprintf(stderr, "buckets[%zu].id = %u\n", i,
                                    buckets[i].id);
                            fprintf(stderr, "buckets[%zu].index = %u\n", i,
                                    buckets[i].index);
                            fprintf(stderr,
                                    "wavefront_size = perm.size() = %zu\n",
                                    wavefront_size);
                            printf_async(Mask(true), "perm = %u\n", perm);
                            printf_async(Mask(true), "instance_id = %u\n",
                                         instance_id);
                        }
                    }
#endif
                    using OrigResult = decltype(func((Class) nullptr, args...));
                    scatter<true>(
                        result,
                        ref_cast_t<OrigResult, Result>(func(
                            (Class) buckets[i].ptr,
                            gather_helper<Is, N>(args, perm)...)),
                        perm);
                } else {
                    func((Class) buckets[i].ptr, gather_helper<Is, N>(args, perm)...);
                }
            } else {
                if constexpr (!std::is_same_v<Result, std::nullptr_t>)
                    scatter<true>(result, zeros<Result>(), perm);
            }
        }
        schedule(result);
//#undef DEBUG_PRINT
#if defined(DEBUG_PRINT)
        fprintf(stderr, "info of result:\n");
        fprintf(
            stderr, "Class = %s, is_pointer_to_template<mitsuba::BSDF, Class>{} = %d\n",
            typeid(Class).name(),
            (int) is_pointer_to_template<mitsuba::BSDF, Class>{});
        if constexpr (is_pointer_to_template<mitsuba::BSDF, Class>{}) {
            if constexpr (is_jit_v<Result> && array_depth_v < Result >> 1) {
                if (!strcmp(typeid(Func).name(),
                            "class "
                            "<lambda_a3166cd1409662f7932530c96e6ad55e>")) {
                    fprintf(stderr, "type of result = %s\n",
                            typeid(result).name());
                    fprintf(stderr, "type of result.derived() = %s\n",
                            typeid(result.derived()).name());
                    fprintf(stderr, "result.derived().size() = %zu\n",
                            result.derived().size());
                    for (size_t i = 0; i < result.derived().size(); ++i) {
                        fprintf(stderr,
                                "type of result.derived().entry(%zu)) = %s\n",
                                i, typeid(result.derived().entry(i)).name());
                        fprintf(stderr, "result.derived().entry(%zu).index() = %u\n", i,
                                result.derived().entry(i).index());
                    }
                }
            }
        }
#endif
#undef DEBUG_PRINT
    } else {
        result = zeros<Result>(self_size);
    }

    return result;
}

template <typename Result, typename Func, typename Self, size_t... Is,
          typename... Args>
Result vcall_jit_reduce_perm_impl(Func func, const Self &self_, int id,
                                  uint32_array_t<Self> &perm_,
                                  int &size_valid,
                                  std::index_sequence<Is...>,
                                  const Args &...args) {
    using UInt32                        = uint32_array_t<Self>;
    using Class                         = scalar_t<Self>;
    using Mask                          = mask_t<UInt32>;
    static constexpr JitBackend Backend = detached_t<Mask>::Backend;
    constexpr size_t N                  = sizeof...(Args);
    DRJIT_MARK_USED(N);

    schedule(args...);

    size_t self_size = self_.size();
    if (self_size == 1) {
        auto self = self_.entry(0);
        if (self)
            return func(self, args...);
        else
            return zeros<Result>();
    }

    Mask mask        = extract_mask<Mask>(args...);
    size_t mask_size = mask.size();

    // Apply mask stack
    mask = Mask::steal(jit_var_mask_apply(
        mask.index(),
        (uint32_t) (self_size > mask_size ? self_size : mask_size)));

    struct SetSelfHelper {
        void set(uint32_t value, uint32_t index) {
            jit_vcall_set_self(detached_t<Mask>::Backend, value, index);
        }

        ~SetSelfHelper() {
            jit_vcall_set_self(detached_t<Mask>::Backend, 0, 0);
        }
    };

    Self self              = self_ & mask;
    auto [buckets, n_inst] = self.vcall_();

//#define DEBUG_PRINT
#if defined(DEBUG_PRINT)
    if constexpr (is_pointer_to_template<mitsuba::BSDF, Class>{}) {
        fprintf(stderr, "In vcall_jit_reduce.h:\n");
        fprintf(stderr, "Result = %s\n", typeid(Result).name());
        fprintf(stderr, "Func = %s\n", typeid(Func).name());
        fprintf(stderr, "Self = %s\n", typeid(Self).name());
        fprintf(stderr, "UInt32 = %s\n", typeid(UInt32).name());
        fprintf(stderr, "Class = %s\n", typeid(Class).name());
        fprintf(stderr, "Mask = %s\n", typeid(Mask).name());
        fprintf(stderr, "self_size = %d\n", (int) self_size);
        fprintf(stderr, "mask_size = %d\n", (int) mask_size);
        fprintf(stderr, "n_inst = %d\n", n_inst);
        fprintf(stderr, "N = sizeof...(Args) = %d\n", (int) N);
        fprintf(stderr, "Args = ");
        (fprintf(stderr, "%s, \n", typeid(Args).name()), ...);
        fprintf(stderr, "\n");
    }
#endif

    Result result;
    SetSelfHelper self_helper;
    if (n_inst > 0 && self_size > 0) {
        result           = empty<Result>(self_size);
        size_t last_size = 0;

        size_valid = 0;

        for (size_t i = 0; i < n_inst; ++i) {
            UInt32 perm           = UInt32::borrow(buckets[i].index);
            size_t wavefront_size = perm.size();

            // Need this or not?
            MaskScope<Mask> scope(Mask::steal(
                jit_var_mask_default(Backend, (uint32_t) wavefront_size)));

            UInt32 instance_id = gather<UInt32>(self, perm);

            if (buckets[i].ptr) {
                // Calc size_valid with extra loop
                if (instance_id == id)
                    size_valid += wavefront_size;
            }
        }

        // TODO: Do not need to return size_valid with reference anymore
        perm_ = empty<UInt32>(size_valid);
        size_valid = 0;

        for (size_t i = 0; i < n_inst; ++i) {
            UInt32 perm           = UInt32::borrow(buckets[i].index);
            size_t wavefront_size = perm.size();

            MaskScope<Mask> scope(Mask::steal(
                jit_var_mask_default(Backend, (uint32_t) wavefront_size)));

            UInt32 instance_id = gather<UInt32>(self, perm);

            // Avoid merging multiple vcall launches if size repeats..
            if (wavefront_size != last_size)
                last_size = wavefront_size;
            else
                eval(result);

#if defined(DEBUG_PRINT)
            if constexpr (is_pointer_to_template<mitsuba::BSDF, Class>{}) {
                fprintf(stderr, "buckets[%zu].ptr = %llu\n", i,
                        (unsigned long long) buckets[i].ptr);
                fprintf(stderr, "buckets[%zu].id = %u\n", i, buckets[i].id);
                fprintf(stderr, "buckets[%zu].index = %u\n", i,
                        buckets[i].index);
                fprintf(stderr, "wavefront_size = perm.size() = %zu\n",
                        wavefront_size);
                fprintf(stderr, "perm = ");
                for (int j = 0; j < wavefront_size; j++) {
                    fprintf(stderr, "%u%s", perm[j],
                            j == wavefront_size - 1 ? "\n" : ", ");
                }
                fprintf(stderr, "instance_id = ");
                for (int j = 0; j < wavefront_size; j++) {
                    fprintf(stderr, "%u%s", instance_id[j],
                            j == wavefront_size - 1 ? "\n" : ", ");
                }
            }
#endif

            if (buckets[i].ptr) {
                self_helper.set(buckets[i].id, instance_id.index());

                if constexpr (!std::is_same_v<Result, std::nullptr_t>) {
                    using OrigResult = decltype(func((Class) nullptr, args...));
                    scatter<true>(result,
                                  ref_cast_t<OrigResult, Result>(func(
                                      (Class) buckets[i].ptr,
                                      gather_helper<Is, N>(args, perm)...)),
                                  perm);
                } else {
                    func((Class) buckets[i].ptr,
                         gather_helper<Is, N>(args, perm)...);
                }

                // concat perm into perm_
                // Hints: use arange to be the index
                // What is Permute in scatter? Try true first

                // Assume that the UInt32 array instance_id contains the same number,
                // i.e. the bucket have the same instance
                if (instance_id == id) {
                    scatter<true>(perm_, perm,
                                  arange<UInt32>(size_valid,
                                                 size_valid + wavefront_size));
                    size_valid += wavefront_size;
                }
            } else {
                if constexpr (!std::is_same_v<Result, std::nullptr_t>)
                    scatter<true>(result, zeros<Result>(), perm);
            }
        }
        schedule(result);
#if defined(DEBUG_PRINT)
        fprintf(stderr, "info of result:\n");
        fprintf(
            stderr,
            "Class = %s, is_pointer_to_template<mitsuba::BSDF, Class>{} = %d\n",
            typeid(Class).name(),
            (int) is_pointer_to_template<mitsuba::BSDF, Class>{});
        if constexpr (is_pointer_to_template<mitsuba::BSDF, Class>{}) {
            if constexpr (is_jit_v<Result> && array_depth_v < Result >> 1) {
                fprintf(stderr, "type of result = %s\n",
                        typeid(result).name());
                fprintf(stderr, "type of result.derived() = %s\n",
                        typeid(result.derived()).name());
                fprintf(stderr, "result.derived().size() = %zu\n",
                        result.derived().size());
                for (size_t i = 0; i < result.derived().size(); ++i) {
                    fprintf(stderr,
                            "type of result.derived().entry(%zu)) = %s\n",
                            i, typeid(result.derived().entry(i)).name());
                    fprintf(stderr,
                            "result.derived().entry(%zu).index() = %u\n", i,
                            result.derived().entry(i).index());
                }
            }
        }
#endif
#undef DEBUG_PRINT
    } else {
        result = zeros<Result>(self_size);
    }

    return result;
}

template <typename Result, typename Func, typename Self, typename... Args>
Result vcall_jit_reduce(const Func &func, const Self &self,
                        const Args &... args) {
    return vcall_jit_reduce_impl<Result>(
        func, self, std::make_index_sequence<sizeof...(Args)>(), args...);
}

template <typename Result, typename Func, typename Self, typename... Args>
Result vcall_jit_reduce_perm(const Func &func, const Self &self, int id,
                             uint32_array_t<Self> &perm,
                             int &size_valid, const Args &...args) {
    return vcall_jit_reduce_perm_impl<Result>(
        func, self, id, perm,
        size_valid, std::make_index_sequence<sizeof...(Args)>(),
        args...);
}

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
