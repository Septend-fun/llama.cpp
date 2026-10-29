#pragma once

#include "common.cuh"

bool ggml_cuda_pipeline_begin(ggml_backend_cuda_context & target, ggml_backend_cuda_context & draft);
bool ggml_cuda_pipeline_active(const ggml_backend_cuda_context * ctx);
bool ggml_cuda_pipeline_graph(ggml_backend_cuda_context * ctx, cudaGraph_t graph, uint64_t id);
bool ggml_cuda_pipeline_copy(ggml_backend_cuda_context * src, ggml_backend_cuda_context * dst,
        const void * source, void * destination, size_t bytes);
bool ggml_cuda_pipeline_event(ggml_backend_cuda_context * ctx, cudaEvent_t event);
void ggml_cuda_pipeline_sync(ggml_backend_cuda_context * ctx = nullptr);
void ggml_cuda_pipeline_event_sync(cudaEvent_t event);
void ggml_cuda_pipeline_cancel(ggml_backend_cuda_context * ctx);
void ggml_cuda_pipeline_finish(ggml_backend_cuda_context * ctx);
uint64_t ggml_cuda_pipeline_graph_id();
