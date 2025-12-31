/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Stub implementations for CUDA kernels when CUDA is not available.
 * These stubs return error code 999 to indicate CUDA functionality is unavailable.
 * This allows the library to build and load without CUDA, with graceful runtime errors.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Complex number types for compatibility */
typedef struct {
    double x;  // Real part
    double y;  // Imaginary part
} CuDoubleComplex;

typedef struct {
    float x;  // Real part
    float y;  // Imaginary part
} CuComplex;

/* Stub: Launch amplitude encoding kernel */
int launch_amplitude_encode(
    const double *input_d,
    void *state_d,
    size_t input_len,
    size_t state_len,
    double inv_norm,
    void *stream)
{
    (void)input_d;
    (void)state_d;
    (void)input_len;
    (void)state_len;
    (void)inv_norm;
    (void)stream;
    return 999;  // Error: CUDA unavailable
}

/* Stub: Launch amplitude encoding kernel (float32) */
int launch_amplitude_encode_f32(
    const float *input_d,
    void *state_d,
    size_t input_len,
    size_t state_len,
    float inv_norm,
    void *stream)
{
    (void)input_d;
    (void)state_d;
    (void)input_len;
    (void)state_len;
    (void)inv_norm;
    (void)stream;
    return 999;
}

/* Stub: Launch batch amplitude encoding kernel */
int launch_amplitude_encode_batch(
    const double *input_batch_d,
    void *state_batch_d,
    const double *inv_norms_d,
    size_t num_samples,
    size_t input_len,
    size_t state_len,
    void *stream)
{
    (void)input_batch_d;
    (void)state_batch_d;
    (void)inv_norms_d;
    (void)num_samples;
    (void)input_len;
    (void)state_len;
    (void)stream;
    return 999;
}

/* Stub: Launch L2 norm reduction */
int launch_l2_norm(
    const double *input_d,
    size_t input_len,
    double *inv_norm_out_d,
    void *stream)
{
    (void)input_d;
    (void)input_len;
    (void)inv_norm_out_d;
    (void)stream;
    return 999;
}

/* Stub: Launch batched L2 norm reduction */
int launch_l2_norm_batch(
    const double *input_batch_d,
    size_t num_samples,
    size_t sample_len,
    double *inv_norms_out_d,
    void *stream)
{
    (void)input_batch_d;
    (void)num_samples;
    (void)sample_len;
    (void)inv_norms_out_d;
    (void)stream;
    return 999;
}

/* Stub: Convert complex128 state vector to complex64 */
int convert_state_to_float(
    const CuDoubleComplex *input_state_d,
    CuComplex *output_state_d,
    size_t len,
    void *stream)
{
    (void)input_state_d;
    (void)output_state_d;
    (void)len;
    (void)stream;
    return 999;
}

#ifdef __cplusplus
}
#endif
