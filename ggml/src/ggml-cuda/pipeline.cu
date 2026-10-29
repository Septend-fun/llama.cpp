#include "pipeline.cuh"

#include <atomic>
#include <cstring>
#include <cstdio>
#include <map>

uint64_t ggml_cuda_pipeline_graph_id() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

#ifdef USE_CUDA_GRAPH
struct pipeline_op {
    enum type_t { GRAPH = 1, COPY = 2, EVENT = 3 } type;
    cudaGraph_t graph = nullptr;
    uint64_t id = 0;
    const void * source = nullptr;
    void * destination = nullptr;
    size_t bytes = 0;
    cudaEvent_t event = nullptr;
    ggml_backend_cuda_context * owner = nullptr;
};

struct pipeline_graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~pipeline_graph() {
        if (executable) { CUDA_CHECK(cudaGraphExecDestroy(executable)); }
        if (graph) { CUDA_CHECK(cudaGraphDestroy(graph)); }
    }
};

struct pipeline_state {
    ggml_backend_cuda_context * target = nullptr;
    ggml_backend_cuda_context * draft = nullptr;
    bool active = false;
    std::vector<pipeline_op> ops;
    std::map<std::vector<uint64_t>, std::unique_ptr<pipeline_graph>> cache;
    size_t launches = 0;
    size_t reused = 0;
    size_t aborted = 0;
    cudaEvent_t input_ready = nullptr;
    int device = -1;

    bool owns(const ggml_backend_cuda_context * ctx) const {
        return !ctx || ctx == target || ctx == draft;
    }

    ~pipeline_state() {
        if (device >= 0) { ggml_cuda_set_device(device); }
        cache.clear();
        if (input_ready) { CUDA_CHECK(cudaEventDestroy(input_ready)); }
        if (std::getenv("DSPARK_CUDA_PIPELINE_DEBUG")) {
            std::fprintf(stderr, "CUDA pipeline: launches=%zu, reused=%zu, aborted=%zu\n", launches, reused, aborted);
        }
    }

    void flush() {
        if (ops.empty()) {
            active = false;
            return;
        }
        std::vector<uint64_t> key;
        for (const auto & op : ops) {
            key.push_back(op.type);
            if (op.type == pipeline_op::GRAPH) {
                key.push_back(op.id);
            } else if (op.type == pipeline_op::COPY) {
                key.insert(key.end(), {(uintptr_t) op.source, (uintptr_t) op.destination, op.bytes});
            } else {
                key.push_back((uintptr_t) op.event);
            }
        }
        auto it = cache.find(key);
        if (it == cache.end()) {
            if (cache.size() >= 8) {
                CUDA_CHECK(cudaStreamSynchronize(target->stream()));
                cache.clear();
            }
            auto entry = std::make_unique<pipeline_graph>();
            CUDA_CHECK(cudaGraphCreate(&entry->graph, 0));
            cudaGraphNode_t previous = nullptr;
            for (const auto & op : ops) {
                cudaGraphNode_t node = nullptr;
                const auto * deps = previous ? &previous : nullptr;
                const size_t n_deps = previous ? 1 : 0;
                if (op.type == pipeline_op::GRAPH) {
                    CUDA_CHECK(cudaGraphAddChildGraphNode(&node, entry->graph, deps, n_deps, op.graph));
                } else if (op.type == pipeline_op::COPY) {
                    CUDA_CHECK(cudaGraphAddMemcpyNode1D(&node, entry->graph, deps, n_deps,
                            op.destination, op.source, op.bytes, cudaMemcpyDeviceToDevice));
                } else {
                    CUDA_CHECK(cudaGraphAddEventRecordNode(&node, entry->graph, deps, n_deps, op.event));
                }
                previous = node;
            }
            CUDA_CHECK(cudaGraphInstantiate(&entry->executable, entry->graph, nullptr, nullptr, 0));
            it = cache.emplace(std::move(key), std::move(entry)).first;
        } else {
            ++reused;
        }
        CUDA_CHECK(cudaEventRecord(input_ready, draft->stream()));
        CUDA_CHECK(cudaStreamWaitEvent(target->stream(), input_ready, 0));
        CUDA_CHECK(cudaGraphLaunch(it->second->executable, target->stream()));
        ++launches;
        ops.clear();
        active = false;
    }
};

static thread_local pipeline_state pipeline;
#endif

bool ggml_cuda_pipeline_begin(ggml_backend_cuda_context & target, ggml_backend_cuda_context & draft) {
#ifdef USE_CUDA_GRAPH
    const char * enabled = std::getenv("DSPARK_CUDA_PIPELINE");
    if (!enabled || std::strcmp(enabled, "1") != 0 || std::getenv("GGML_CUDA_DISABLE_GRAPHS") || target.device != draft.device) {
        return false;
    }
    pipeline.flush();
    if (pipeline.target != &target || pipeline.draft != &draft || pipeline.device != target.device) {
        if (pipeline.device >= 0) { ggml_cuda_set_device(pipeline.device); }
        pipeline.cache.clear();
        if (pipeline.input_ready) {
            CUDA_CHECK(cudaEventDestroy(pipeline.input_ready));
            pipeline.input_ready = nullptr;
        }
        pipeline.device = target.device;
        ggml_cuda_set_device(target.device);
        pipeline.target = &target;
        pipeline.draft = &draft;
    }
    pipeline.active = true;
    if (!pipeline.input_ready) {
        CUDA_CHECK(cudaEventCreateWithFlags(&pipeline.input_ready, cudaEventDisableTiming));
    }
    return true;
#else
    GGML_UNUSED(target); GGML_UNUSED(draft);
    return false;
#endif
}

bool ggml_cuda_pipeline_active(const ggml_backend_cuda_context * ctx) {
#ifdef USE_CUDA_GRAPH
    return pipeline.active && pipeline.owns(ctx);
#else
    GGML_UNUSED(ctx);
    return false;
#endif
}

bool ggml_cuda_pipeline_graph(ggml_backend_cuda_context * ctx, cudaGraph_t graph, uint64_t id) {
#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_pipeline_active(ctx)) {
        pipeline_op op{pipeline_op::GRAPH}; op.graph = graph; op.id = id; op.owner = ctx;
        pipeline.ops.push_back(op);
        return true;
    }
#else
    GGML_UNUSED(ctx); GGML_UNUSED(graph); GGML_UNUSED(id);
#endif
    return false;
}

bool ggml_cuda_pipeline_copy(ggml_backend_cuda_context * src, ggml_backend_cuda_context * dst,
        const void * source, void * destination, size_t bytes) {
#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_pipeline_active(src) && ggml_cuda_pipeline_active(dst)) {
        pipeline_op op{pipeline_op::COPY}; op.source = source; op.destination = destination; op.bytes = bytes;
        pipeline.ops.push_back(op);
        return true;
    }
#else
    GGML_UNUSED(src); GGML_UNUSED(dst); GGML_UNUSED(source); GGML_UNUSED(destination); GGML_UNUSED(bytes);
#endif
    return false;
}

bool ggml_cuda_pipeline_event(ggml_backend_cuda_context * ctx, cudaEvent_t event) {
#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_pipeline_active(ctx)) {
        pipeline_op op{pipeline_op::EVENT}; op.event = event;
        pipeline.ops.push_back(op);
        return true;
    }
#else
    GGML_UNUSED(ctx); GGML_UNUSED(event);
#endif
    return false;
}

void ggml_cuda_pipeline_sync(ggml_backend_cuda_context * ctx) {
#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_pipeline_active(ctx) && !pipeline.ops.empty()) {
        if (ctx) {
            bool computes = false;
            for (const auto & op : pipeline.ops) {
                computes |= op.type == pipeline_op::GRAPH && op.owner == ctx;
            }
            if (!computes) { return; }
        }
        ++pipeline.aborted;
        pipeline.flush();
    }
#else
    GGML_UNUSED(ctx);
#endif
}

void ggml_cuda_pipeline_cancel(ggml_backend_cuda_context * ctx) {
#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_pipeline_active(ctx)) {
        ++pipeline.aborted;
        pipeline.flush();
    }
#else
    GGML_UNUSED(ctx);
#endif
}

void ggml_cuda_pipeline_event_sync(cudaEvent_t event) {
#ifdef USE_CUDA_GRAPH
    if (pipeline.active) {
        for (const auto & op : pipeline.ops) {
            if (op.type == pipeline_op::EVENT && op.event == event) {
                ggml_cuda_pipeline_sync();
                break;
            }
        }
    }
#else
    GGML_UNUSED(event);
#endif
}

void ggml_cuda_pipeline_finish(ggml_backend_cuda_context * ctx) {
#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_pipeline_active(ctx)) { pipeline.flush(); }
#else
    GGML_UNUSED(ctx);
#endif
}
