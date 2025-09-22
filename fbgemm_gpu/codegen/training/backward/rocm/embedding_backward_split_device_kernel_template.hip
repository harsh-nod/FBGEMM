/*******************************************************************************
 * Copyright (c) 2016 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "fbgemm_gpu/rocm/split_embeddings_common.h"

namespace fbgemm_gpu::rocm {
template <typename cache_t, typename emb_t, int32_t embedding_dim,
          int32_t weight_decay_mode>
struct rowwise_adagrad_optimizer_t {
  __device__
  rowwise_adagrad_optimizer_t(const rowwise_adagrad_kernel_arg_t &karg_)
      : karg(karg_) {}

  template <int32_t thread_length, int32_t segment_split>
  __device__ void update(cache_t *acc, emb_t *weight, uint32_t row_index) {
    if constexpr (segment_split == 0) {
      cache_t *p_momentum = reinterpret_cast<cache_t *>(karg.p_momentum);
      cache_t momentum = p_momentum[row_index]; // should be s_load
      // compute per row square sum
      cache_t local_sum_squre = .0f;
      if constexpr (weight_decay_mode == 1) {
#pragma unroll
        for (auto i = 0; i < thread_length; i++) {
          cache_t w = static_cast<cache_t>(weight[i]);
          cache_t a = acc[i] + w * karg.weight_decay;
          local_sum_squre += a * a;
        }
      } else {
#pragma unroll
        for (auto i = 0; i < thread_length; i++) {
          cache_t a = acc[i];
          local_sum_squre += a * a;
        }
      }

      cache_t avg_square =
          wave_reduce<reduce_op_sum_t<cache_t>, cache_t, AMDGCN_WAVE_SIZE>(
              local_sum_squre) /
          embedding_dim;

      cache_t momentum_new = momentum + avg_square;

      cache_t multiplier =
          karg.learning_rate / (sqrtf(momentum_new) + karg.eps);
      cache_t correction;

      if constexpr (weight_decay_mode == 1) {
        correction = 1.0 - multiplier * karg.weight_decay;
      } else if constexpr (weight_decay_mode == 2) {
        correction = 1.0 - karg.learning_rate * karg.weight_decay;
      } else {
        correction = 1.0;
      }

// update new weight value
#pragma unroll
      for (auto i = 0; i < thread_length; i++) {
        cache_t w = static_cast<cache_t>(weight[i]);
        cache_t a = acc[i];
        w = correction * w - multiplier * a;
        weight[i] = static_cast<emb_t>(w);
      }

      p_momentum[row_index] = momentum_new;
    }
  }

  rowwise_adagrad_kernel_arg_t karg;
};

template <typename optimizer_t, typename optimizer_karg_t, typename emb_t,
          typename cache_t, typename grad_t, typename index_t,
          int32_t block_size, int32_t embedding_dim,
          int32_t segment_prefetch, // 2
          int32_t segment_unroll,   // 8
          int32_t segment_split,    // 0-warp per row, 1-cta per row, 2-atomic
          bool weighted>
__device__ void split_tbe_backward_hip_kernel_{{kdesc}}(
    const grad_t *p_output_grad, emb_t *p_emb_table,
    const int64_t *p_hash_size_cumsum,
    const index_t *p_sorted_linear_indices_run,
    const int32_t *p_sorted_linear_indices_cumulative_run_lengths,
    const int32_t *p_sorted_linear_indices_num_runs,
    const int32_t info_B_num_bits, const uint32_t info_B_mask,
    const int32_t *p_sorted_infos,
    uint32_t max_segment_length_per_warp, uint32_t emb_dim,
    uint32_t num_tables, optimizer_karg_t opt_karg,
    const float *p_sorted_indice_weights = nullptr) {
  constexpr uint32_t dword_per_row =
      (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
  constexpr uint32_t waves_per_block = block_size / AMDGCN_WAVE_SIZE;
  constexpr uint32_t length_mask = ~(segment_unroll - 1);
  const uint32_t wave_id =
      __builtin_amdgcn_readfirstlane(threadIdx.x / AMDGCN_WAVE_SIZE);
  const uint32_t lane_id = threadIdx.x % AMDGCN_WAVE_SIZE;
  const uint32_t run_id = wave_id + blockIdx.x * waves_per_block;

  if (run_id >= p_sorted_linear_indices_num_runs[0]) {
    return;
  }

  const auto linear_index = p_sorted_linear_indices_run[run_id];

  const int32_t segment_start =
      p_sorted_linear_indices_cumulative_run_lengths[run_id];
  const int32_t segment_end =
      p_sorted_linear_indices_cumulative_run_lengths[run_id + 1];

  const int32_t segment_length = segment_end - segment_start;
  if (segment_length >= max_segment_length_per_warp)
    return;

  const int32_t segment_length_mod = segment_length & length_mask;

  const auto info_0 =
      reinterpret_cast<const uint32_t *>(&p_sorted_infos[0])[segment_start];
  const auto t_0 = info_0 >> info_B_num_bits;
  int64_t hash_size = p_hash_size_cumsum[t_0];

  const int64_t emb_idx = linear_index - hash_size;

  p_emb_table += hash_size * emb_dim;
  opt_karg.p_momentum = reinterpret_cast<void *>(
      reinterpret_cast<cache_t *>(opt_karg.p_momentum) + hash_size);

  uint32_t infos[segment_unroll];
  grad_t grad_data[dword_per_row * segment_prefetch];
  emb_t emb_data[dword_per_row];
  float indice_weights[segment_unroll];

  alignas(16) cache_t grad_acc[dword_per_row] = {};

  int itr = 0;
  if (segment_length_mod == 0)
    goto L_tail_grad_acc;

  if constexpr (!weighted) {
#pragma unroll
    for (int i = 0; i < segment_unroll; i++) {
      infos[i] = p_sorted_infos[segment_start + i];
    }
  } else {
    for (int i = 0; i < segment_unroll; i++) {
      infos[i] = p_sorted_infos[segment_start + i];
      indice_weights[i] = p_sorted_indice_weights[segment_start + i];
    }
  }

  itr += segment_unroll;
  p_sorted_infos += segment_unroll;

  if constexpr (weighted) {
    p_sorted_indice_weights += segment_unroll;
  }

  uint32_t bag_index;
  uint32_t table_index;

  // LOOP
  /*
  -- shorter live ranges
  -- used __restrict__ thinking it might help in better schedulig... need to
  investigate;
  -- need to see vpr/sgpr cnts
  -- accumualte and load pipelining
  */
  for (; itr < segment_length_mod; itr += segment_unroll) {

    const uint32_t Bbits = info_B_num_bits;
    const uint32_t Bmask = info_B_mask;
    const uint32_t stride = embedding_dim;
    const uint32_t tables = num_tables;

    cache_t *__restrict__ acc_base = &grad_acc[0];
    grad_t *__restrict__ gbuf0 = &grad_data[0];
    grad_t *__restrict__ gbuf1 = &grad_data[dword_per_row];

    {
      const uint32_t i0 = infos[0];
      const uint32_t t0 = (i0 >> Bbits);
      const uint32_t b0 = (i0 & Bmask);
      load_row_per_warp<grad_t, embedding_dim, index_t>::run(
          gbuf0, b0 * tables, p_output_grad + (uint64_t)t0 * stride, lane_id);

      const uint32_t i1 = infos[1];
      const uint32_t t1 = (i1 >> Bbits);
      const uint32_t b1 = (i1 & Bmask);
      load_row_per_warp<grad_t, embedding_dim, index_t>::run(
          gbuf1, b1 * tables, p_output_grad + (uint64_t)t1 * stride, lane_id);
    }

#pragma unroll
    for (int j = 2; j < segment_unroll; j += 2) {
      if constexpr (!weighted) {
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                /*weighted=*/false>::run(acc_base, gbuf0,
                                                         lane_id);
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                /*weighted=*/false>::run(acc_base, gbuf1,
                                                         lane_id);
      } else {
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                /*weighted=*/true>::run(acc_base, gbuf0,
                                                        lane_id,
                                                        indice_weights[j - 2]);
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                /*weighted=*/true>::run(acc_base, gbuf1,
                                                        lane_id,
                                                        indice_weights[j - 1]);
      }

      const uint32_t ij = infos[j];
      const uint32_t tj = (ij >> Bbits);
      const uint32_t bj = (ij & Bmask);
      load_row_per_warp<grad_t, embedding_dim, index_t>::run(
          gbuf0, bj * tables, p_output_grad + (uint64_t)tj * stride, lane_id);

      const uint32_t ij1 = infos[j + 1];
      const uint32_t tj1 = (ij1 >> Bbits);
      const uint32_t bj1 = (ij1 & Bmask);
      load_row_per_warp<grad_t, embedding_dim, index_t>::run(
          gbuf1, bj1 * tables, p_output_grad + (uint64_t)tj1 * stride, lane_id);
    }

    // accumulate the last pair
    if constexpr (!weighted) {
      accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                              /*weighted=*/false>::run(acc_base, gbuf0,
                                                       lane_id);
      accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                              /*weighted=*/false>::run(acc_base, gbuf1,
                                                       lane_id);
    } else {
      accumulate_row_per_warp<
          grad_t, embedding_dim, cache_t,
          /*weighted=*/true>::run(acc_base, gbuf0, lane_id,
                                  indice_weights[segment_unroll - 2]);
      accumulate_row_per_warp<
          grad_t, embedding_dim, cache_t,
          /*weighted=*/true>::run(acc_base, gbuf1, lane_id,
                                  indice_weights[segment_unroll - 1]);
    }

#pragma unroll
    for (int i = 0; i < segment_unroll; ++i) {
      infos[i] = p_sorted_infos[i];
    }
    p_sorted_infos += segment_unroll;

    if constexpr (weighted) {
#pragma unroll
      for (int i = 0; i < segment_unroll; ++i) {
        indice_weights[i] = p_sorted_indice_weights[i];
      }
      p_sorted_indice_weights += segment_unroll;
    }
  }

  // LAST
  {
    const uint32_t tail = segment_length - segment_length_mod;
    if (tail) {
      const uint32_t Bbits = info_B_num_bits;
      const uint32_t Bmask = info_B_mask;
      const uint32_t stride = embedding_dim;
      const uint32_t tables = num_tables;

      cache_t *__restrict__ acc_base = &grad_acc[0];
      grad_t *__restrict__ gbuf0 = &grad_data[0];
      grad_t *__restrict__ gbuf1 = &grad_data[dword_per_row];

      {
        const uint32_t i0 = infos[0];
        const uint32_t t0 = (i0 >> Bbits);
        const uint32_t b0 = (i0 & Bmask);
        load_row_per_warp<grad_t, embedding_dim, index_t>::run(
            gbuf0, b0 * tables, p_output_grad + (uint64_t)t0 * stride, lane_id);

        if (tail >= 2) {
          const uint32_t i1 = infos[1];
          const uint32_t t1 = (i1 >> Bbits);
          const uint32_t b1 = (i1 & Bmask);
          load_row_per_warp<grad_t, embedding_dim, index_t>::run(
              gbuf1, b1 * tables, p_output_grad + (uint64_t)t1 * stride,
              lane_id);
        }
      }

      uint32_t j = (tail >= 2) ? 2u : 1u;

      for (; j + 1u < tail; j += 2u) {
        if constexpr (!weighted) {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/false>::run(acc_base, gbuf0,
                                                           lane_id);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/false>::run(acc_base, gbuf1,
                                                           lane_id);
        } else {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/true>::run(acc_base, gbuf0,
                                                          lane_id,
                                                          indice_weights[j -
                                                                         2]);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/true>::run(acc_base, gbuf1,
                                                          lane_id,
                                                          indice_weights[j -
                                                                         1]);
        }

        const uint32_t ij = infos[j];
        const uint32_t tj = (ij >> Bbits);
        const uint32_t bj = (ij & Bmask);
        load_row_per_warp<grad_t, embedding_dim, index_t>::run(
            gbuf0, bj * tables, p_output_grad + (uint64_t)tj * stride, lane_id);

        const uint32_t ij1 = infos[j + 1];
        const uint32_t tj1 = (ij1 >> Bbits);
        const uint32_t bj1 = (ij1 & Bmask);
        load_row_per_warp<grad_t, embedding_dim, index_t>::run(
            gbuf1, bj1 * tables, p_output_grad + (uint64_t)tj1 * stride,
            lane_id);
      }

      if (tail >= 2) {
        if constexpr (!weighted) {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/false>::run(acc_base, gbuf0,
                                                           lane_id);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/false>::run(acc_base, gbuf1,
                                                           lane_id);
        } else {
          // last pair corresponds to indicess (tail-2, tail-1)
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/true>::run(acc_base, gbuf0,
                                                          lane_id,
                                                          indice_weights[tail -
                                                                         2]);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/true>::run(acc_base, gbuf1,
                                                          lane_id,
                                                          indice_weights[tail -
                                                                         1]);
        }
      } else {
        // tail == 1:
        if constexpr (!weighted) {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/false>::run(acc_base, gbuf0,
                                                           lane_id);
        } else {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t,
                                  /*weighted=*/true>::run(acc_base, gbuf0,
                                                          lane_id,
                                                          indice_weights[0]);
        }
      }
    }
  }

L_tail_grad_acc : {
  const uint32_t tail = segment_length - itr;
  if (tail) {
    const uint32_t Bbits = info_B_num_bits;
    const uint32_t Bmask = info_B_mask;
    const uint32_t stride = embedding_dim;
    const uint32_t tables = num_tables;

    const int32_t *__restrict__ infos_ptr =
        p_sorted_infos + segment_start + itr;
    const float *__restrict__ w_ptr =
        (weighted ? (p_sorted_indice_weights + segment_start + itr) : nullptr);

    grad_t *__restrict__ gbuf0 = &grad_data[0];
    grad_t *__restrict__ gbuf1 = &grad_data[dword_per_row];
    cache_t *__restrict__ acc_base = &grad_acc[0];

    uint32_t i0u = static_cast<uint32_t>(*infos_ptr++);
    uint32_t t0 = (i0u >> Bbits);
    uint32_t b0 = (i0u & Bmask);
    load_row_per_warp<grad_t, embedding_dim, index_t>::run(
        gbuf0, b0 * tables, p_output_grad + (uint64_t)t0 * stride, lane_id);
    float w0 = 0.0f;
    if constexpr (weighted) {
      w0 = *w_ptr++;
    }

    if (tail == 1) {
      if constexpr (!weighted) {
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, false>::run(
            acc_base, gbuf0, lane_id);
      } else {
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, true>::run(
            acc_base, gbuf0, lane_id, w0);
      }
    } else {

      uint32_t i1u = static_cast<uint32_t>(*infos_ptr++);
      uint32_t t1 = (i1u >> Bbits);
      uint32_t b1 = (i1u & Bmask);
      load_row_per_warp<grad_t, embedding_dim, index_t>::run(
          gbuf1, b1 * tables, p_output_grad + (uint64_t)t1 * stride, lane_id);
      float w1 = 0.0f;
      if constexpr (weighted) {
        w1 = *w_ptr++;
      }

      uint32_t done = 2;
      
      while (done + 1u < tail) {
        if constexpr (!weighted) {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, false>::run(
              acc_base, gbuf0, lane_id);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, false>::run(
              acc_base, gbuf1, lane_id);
        } else {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, true>::run(
              acc_base, gbuf0, lane_id, w0);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, true>::run(
              acc_base, gbuf1, lane_id, w1);
        }

        uint32_t iwu = static_cast<uint32_t>(*infos_ptr++);
        uint32_t tw = (iwu >> Bbits);
        uint32_t bw = (iwu & Bmask);
        load_row_per_warp<grad_t, embedding_dim, index_t>::run(
            gbuf0, bw * tables, p_output_grad + (uint64_t)tw * stride, lane_id);
        float ww = 0.0f;
        if constexpr (weighted) {
          ww = *w_ptr++;
        }

        uint32_t iw1u = static_cast<uint32_t>(*infos_ptr++);
        uint32_t tw1 = (iw1u >> Bbits);
        uint32_t bw1 = (iw1u & Bmask);
        load_row_per_warp<grad_t, embedding_dim, index_t>::run(
            gbuf1, bw1 * tables, p_output_grad + (uint64_t)tw1 * stride,
            lane_id);
        float ww1 = 0.0f;
        if constexpr (weighted) {
          ww1 = *w_ptr++;
        }

        if constexpr (weighted) {
          w0 = ww;
          w1 = ww1;
        }
        done += 2;
      }

      // pairwise handling of tail
      if (done == tail) {
        if constexpr (!weighted) {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, false>::run(
              acc_base, gbuf0, lane_id);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, false>::run(
              acc_base, gbuf1, lane_id);
        } else {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, true>::run(
              acc_base, gbuf0, lane_id, w0);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, true>::run(
              acc_base, gbuf1, lane_id, w1);
        }
      } else { // last one remaining
        if constexpr (!weighted) {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, false>::run(
              acc_base, gbuf0, lane_id);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, false>::run(
              acc_base, gbuf1, lane_id);
        } else {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, true>::run(
              acc_base, gbuf0, lane_id, w0);
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, true>::run(
              acc_base, gbuf1, lane_id, w1);
        }

        uint32_t iku = static_cast<uint32_t>(*infos_ptr++);
        uint32_t tk = (iku >> Bbits);
        uint32_t bk = (iku & Bmask);
        load_row_per_warp<grad_t, embedding_dim, index_t>::run(
            gbuf0, bk * tables, p_output_grad + (uint64_t)tk * stride, lane_id);
        float wk = 0.0f;
        if constexpr (weighted) {
          wk = *w_ptr++;
        }

        if constexpr (!weighted) {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, false>::run(
              acc_base, gbuf0, lane_id);
        } else {
          accumulate_row_per_warp<grad_t, embedding_dim, cache_t, true>::run(
              acc_base, gbuf0, lane_id, wk);
        }
      }

      itr = segment_length;
      p_sorted_infos = infos_ptr;
      if constexpr (weighted)
        p_sorted_indice_weights = w_ptr;
    }
  }

  load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], emb_idx,
                                                        p_emb_table, lane_id);
  optimizer_t optimizer(opt_karg);
  optimizer.template update<dword_per_row, segment_split>(grad_acc, emb_data,
                                                          emb_idx);

  store_row_per_warp<emb_t, embedding_dim, emb_t>::run(
      &emb_data[0], p_emb_table + (uint64_t)emb_idx * embedding_dim, lane_id);
}
}
} // namespace fbgemm_gpu::rocm
