//
// Created by xiaoi on 12/29/23.
//
#include "chatllm.h"
#include <fcntl.h>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>
#include <sys/stat.h>
#include <iostream>
#include <unordered_set>

using namespace chatllm;

#define MAX_NODES 8192

// ===== streamer =====

auto StreamerGroup::put(const std::vector<int_least32_t> &output_ids) -> void {
    for (auto &streamer : streamers_) {
        streamer->put(output_ids);
    }
}

auto StreamerGroup::end() -> void {
    for (auto &streamer : streamers_) {
        streamer->end();
    }
}

auto TextStreamer::put(const std::vector<int> &output_ids) -> void {
    if (is_prompt_) {
        is_prompt_ = false;
        return;
    }

    static const std::vector<char> puncts{',', '!', ':', ';', '?'};

    token_cache_.insert(token_cache_.end(), output_ids.begin(), output_ids.end());
    std::string text = tokenizer_->tokenizer.DecodeIds(token_cache_);
    if (text.empty()) {
        return;
    }

    std::string printable_text;
    if (text.back() == '\n') {
        // flush the cache after newline
        printable_text = text.substr(print_len_);
        token_cache_.clear();
        print_len_ = 0;
    } else if (std::find(puncts.begin(), puncts.end(), text.back()) != puncts.end()) {
        // last symbol is a punctuation, hold on
    } else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
        // ends with an incomplete token, hold on
    } else {
        printable_text = text.substr(print_len_);
        print_len_ = text.size();
    }

    os_ << printable_text << std::flush;
}

auto TextStreamer::end() -> void {
    std::string text = tokenizer_->tokenizer.DecodeIds(token_cache_);
    os_ << text.substr(print_len_) << std::endl;
    is_prompt_ = true;
    token_cache_.clear();
    print_len_ = 0;
}

auto PerfStreamer::put(const std::vector<int> &output_ids) -> void {
    if (num_prompt_tokens_ == 0) {
        // before prompt eval
        start_us_ = ggml_time_us();
        num_prompt_tokens_ = output_ids.size();
    } else {
        if (num_output_tokens_ == 0) {
            // first new token
            prompt_us_ = ggml_time_us();
        }
        num_output_tokens_ += output_ids.size();
    }
}

auto PerfStreamer::reset() -> void {
    start_us_ = prompt_us_ = end_us_ = 0;
    num_prompt_tokens_ = num_output_tokens_ = 0;
}

auto PerfStreamer::to_string() -> std::string const {
    std::ostringstream oss;
    oss << "prompt time: " << prompt_total_time_us() / 1000.f << " ms / " << num_prompt_tokens() << " tokens ("
        << prompt_token_time_us() / 1000.f << " ms/token)\n"
        << "output time: " << output_total_time_us() / 1000.f << " ms / " << num_output_tokens() << " tokens ("
        << output_token_time_us() / 1000.f << " ms/token)\n"
        << "total time: " << (prompt_total_time_us() + output_total_time_us()) / 1000.f << " ms";
    return oss.str();
}




// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp
auto chatllm::ggml_graph_compute_helper(std::vector<uninitialized_char> &buf, ggml_cgraph *graph, int n_threads) -> void {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = (uint8_t *)buf.data();
    }
//    int* x = (int*)graph->nodes[0]->src[1]->data;
//    for (int i = 0; i < 7; i++){
//        std::cout << x[i] << std::endl;
//    }
    ggml_graph_compute(graph, &plan);
}

auto Embedding::forward(ModelContext *ctx, ggml_tensor *input) const -> ggml_tensor * {

    ggml_tensor *output = ggml_get_rows(ctx->ctx_b.get(), weight, input);

    return output;
}

auto Linear::forward(ModelContext *ctx, ggml_tensor *input) const -> ggml_tensor * {
    // input: [seqlen, in_features]
    ggml_tensor *output = ggml_mul_mat(ctx->ctx_b.get(), weight, input); // [seqlen, out_features]

    return output;
}

auto RMSNorm::forward(ModelContext *ctx, ggml_tensor *input, float eps) const -> ggml_tensor * {
    ggml_context *gctx = ctx->ctx_b.get();
    auto ggml_rms_norm_fn = inplace ? ggml_rms_norm_inplace : ggml_rms_norm;
    ggml_tensor *output = ggml_rms_norm_fn(gctx, input, eps);
    output = ggml_mul_inplace(gctx, output, weight);
    return output;
}

FileMmap::FileMmap(std::string &model_path) {
    int file = open(model_path.c_str(), O_RDONLY);
    struct stat s;
    fstat(file, &s);
    size = s.st_size;

    data = (char *)mmap(nullptr, size, PROT_READ, MAP_SHARED, file, 0);
    close(file);
}
FileMmap::~FileMmap() {munmap(data, size);}

auto ModelLoader::seek(int64_t offset, int whence) -> void {
    if (whence == SEEK_SET) {
        ptr = data + offset;
    } else if (whence == SEEK_CUR) {
        ptr += offset;
    } else if (whence == SEEK_END) {
        ptr = data + size + offset;
    } else {
        std::cout << "invalid seek mode " << whence;
    }
}

auto ModelLoader::read_string(size_t length) -> std::string {
    std::string s(ptr, ptr + length);
    ptr += length;
    return s;
}

auto ModelLoader::read_tensor(const std::string &name, ggml_tensor *tensor) -> void {
    // read and check tensor name
    {
        int name_size = read_basic<int>();
        if(name_size != (int)name.size()){
            std::cout << "tensor " << name << " name size mismatch: expect " << name.size() << " but got " << name_size;
        }

        std::string weight_name = read_string(name_size);
        if(weight_name != name){
            std::cout << "tensor name mismatch: expect " << name << " but got " << weight_name;
        }

    }
    // read and check tensor shape
    {
        int ndim = read_basic<int>();
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_size = read_basic<int>();
        }
    }
    // read and check tensor dtype
    {
        ggml_type dtype = (ggml_type)read_basic<int>();
    }
    // map tensor data
    {
        constexpr int64_t MEM_ALIGNED = 16;
        const int64_t data_offset = (tell() + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
//        const int64_t data_offset = (tell() + (MEM_ALIGNED - 1)) / MEM_ALIGNED * MEM_ALIGNED;
        tensor->data = const_cast<char *const>(data) + data_offset;
        seek(data_offset + ggml_nbytes(tensor), SEEK_SET);
    }

}

ChatllmAttention::ChatllmAttention(ModelContext *ctx, int hidden_size, int num_attention_heads, int max_length)
        : num_attention_heads(num_attention_heads),
          c_attn(ctx, hidden_size, 3 * hidden_size), c_proj(ctx, hidden_size, hidden_size),
          k_cache(ggml_new_tensor_3d(ctx->ctx_kv.get(), GGML_TYPE_F16, hidden_size / num_attention_heads, max_length,
                                     num_attention_heads)),
          v_cache(ggml_new_tensor_3d(ctx->ctx_kv.get(), GGML_TYPE_F16,max_length, hidden_size / num_attention_heads,
                                     num_attention_heads)) {}

auto ChatllmAttention::forward(chatllm::ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos,
                               int n_ctx) const -> ggml_tensor * {
    ggml_context *gctx = ctx->ctx_b.get();
    const int hidden_size = hidden_states->ne[0];
    const int qlen = hidden_states->ne[1];
    const int head_size = hidden_size / num_attention_heads;
    const int n_past = static_cast<int *>(KQ_pos->data)[0];

    ggml_tensor *qkv = c_attn.forward(ctx, hidden_states);
    ggml_tensor *query_layer =
            ggml_view_3d(gctx, qkv, head_size, num_attention_heads, qlen, head_size * ggml_element_size(qkv), qkv->nb[1],
                         0); // [qlen, heads, head_size]
    if (!ggml_is_contiguous(query_layer)) {
        query_layer = ggml_cont(gctx, query_layer);
    }
    query_layer =  ggml_permute(gctx, query_layer, 0, 2, 1, 3);

    ggml_tensor *key_layer =
            ggml_view_3d(gctx, qkv, head_size, num_attention_heads, qlen, head_size * ggml_element_size(qkv), qkv->nb[1],
                         hidden_size * ggml_element_size(qkv)); // [qlen, kv_heads, head_size]

    key_layer =ggml_permute(gctx, key_layer, 0, 2, 1, 3); // [kv_heads, qlen, head_size]

    ggml_tensor *value_layer =
            ggml_view_3d(gctx, qkv, head_size, num_attention_heads, qlen, head_size * ggml_element_size(qkv), qkv->nb[1],
                         (hidden_size + head_size * num_attention_heads) * ggml_element_size(qkv)); // [qlen, kv_heads, head_size]

    value_layer = ggml_permute( gctx, value_layer, 1, 2, 0, 3); // [kv_heads, qlen, head_size]

    // store key & value to cache
    ggml_tensor *k_cache_view =
            ggml_view_3d(gctx, k_cache, head_size, qlen, num_attention_heads, k_cache->nb[1], k_cache->nb[2],
                         n_past * head_size * ggml_element_size(k_cache)); // [kv_heads, qlen, head_size]
    ggml_build_forward_expand(&ctx->gf, ggml_cpy(gctx, key_layer, k_cache_view));
    ggml_tensor *v_cache_view =
            ggml_view_3d(gctx, v_cache, qlen, head_size, num_attention_heads, v_cache->nb[1], v_cache->nb[2],
                         n_past * ggml_element_size(v_cache)); // [kv_heads, qlen, head_size]
    ggml_build_forward_expand(&ctx->gf, ggml_cpy(gctx, value_layer, v_cache_view));

    // concat key & value with past kv
    key_layer =
            ggml_view_3d(gctx, k_cache, head_size, n_past + qlen, num_attention_heads,
                         k_cache->nb[1], k_cache->nb[2], 0); // [kv_heads, klen, head_size]
    value_layer =
            ggml_view_3d(gctx, v_cache, n_past + qlen, head_size, num_attention_heads,
                         v_cache->nb[1], v_cache->nb[2], 0); // [kv_heads, head_size, klen]


    // attention
    ggml_tensor *attn_scores = ggml_mul_mat(gctx, key_layer, query_layer);

    attn_scores = ggml_scale_inplace(gctx, attn_scores,  ggml_new_f32(gctx, 1.f / std::sqrt(head_size)));

    attn_scores = ggml_alibi(gctx, attn_scores, n_past, num_attention_heads, 8.0f);
    if (n_past == 0) {
        // build attention mask for context input
        attn_scores = ggml_diag_mask_inf_inplace(gctx, attn_scores, n_past);
    }
    ggml_tensor *attn_probs = ggml_soft_max_inplace(gctx, attn_scores); // [kv_heads, mqa_scale * qlen, klen]

    ggml_tensor *context_layer = ggml_mul_mat(gctx, value_layer, attn_probs); // [kv_heads, mqa_scale * qlen, head_size]

    context_layer = ggml_cont(gctx, ggml_permute(gctx, context_layer, 0, 2, 1, 3)); // [qlen, heads, head_size]
    context_layer = ggml_reshape_2d(gctx, context_layer, hidden_size, qlen); // [qlen, hidden]

    ggml_tensor *attn_output = c_proj.forward(ctx, context_layer);
    return attn_output;

}

auto ChatllmMLP::forward(ModelContext *ctx, ggml_tensor *hidden_states) const -> ggml_tensor * {
    ggml_context *gctx = ctx->ctx_b.get();

    ggml_tensor *a2 = gate_proj.forward(ctx, hidden_states);
    a2 = ggml_silu_inplace(gctx, a2);
    ggml_tensor *a1 = up_proj.forward(ctx, hidden_states);

    ggml_tensor *output = ggml_mul_inplace(gctx, a2, a1);
    output = down_proj.forward(ctx, output);
    return output;
}

auto ChatllmBlock::forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor * {
    ggml_context *gctx = ctx->ctx_b.get();

    ggml_tensor *residual = hidden_states;
    hidden_states = ln_1.forward(ctx, hidden_states, 1e-6f);
    hidden_states = attn.forward(ctx, hidden_states, KQ_pos, n_ctx);
    hidden_states = ggml_add_inplace(gctx, hidden_states, residual);

    residual = hidden_states;
    hidden_states = ln_2.forward(ctx, hidden_states, 1e-6f);
    hidden_states = mlp.forward(ctx, hidden_states);
    hidden_states = ggml_add_inplace(gctx, hidden_states, residual);

    return hidden_states;
}

ChatllmModel::ChatllmModel(ModelContext *ctx, const Config &config)
        : wte(ctx, config.vocab_size, config.hidden_size), ln_f(ctx, config.hidden_size) {
    layers.reserve(config.num_hidden_layers);
    for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
        layers.emplace_back(ctx, config.hidden_size, config.num_attention_head,
                            config.intermediate_size, config.model_max_length);
    }
}

auto ChatllmModel::forward(ModelContext *ctx, ggml_tensor *input_ids, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor * {
    ggml_context *gctx = ctx->ctx_b.get();

    ggml_tensor *hidden_states = wte.forward(ctx, input_ids);

    for (const auto &layer : layers) {
        ggml_set_scratch(gctx, ctx->scratch);
        hidden_states = layer.forward(ctx, hidden_states, KQ_pos, n_ctx);
    }
    ggml_scratch empty_scratch = {0, 0, nullptr};
    ggml_set_scratch(gctx, empty_scratch);
    hidden_states = ln_f.forward(ctx, hidden_states, 1e-6f);
    return hidden_states;
}

ChatllmForCausalLM::ChatllmForCausalLM(const Config &config)
        : config(config) {
    ctx_.compute_buffer.resize(MEM_SIZE);
    ctx_.scratch_buffer.resize(SCRATCH_SIZE);
    ctx_.scratch = {0, ctx_.scratch_buffer.size(), ctx_.scratch_buffer.data()};
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t ctx_w_size = (3 + config.num_hidden_layers * 7) * tensor_ovhd;
    const size_t ctx_kv_size = 2 * config.num_hidden_layers *
                               (config.model_max_length * config.hidden_size * ggml_type_size(GGML_TYPE_F16) + tensor_ovhd);
    ctx_.ctx_w =  chatllm::make_unique_ggml_context(ctx_w_size, nullptr, true);
    ctx_.ctx_kv = chatllm::make_unique_ggml_context(ctx_kv_size + 1 * 1024 * 1024, nullptr, false);

    transformer = ChatllmModel(&ctx_, config);
    lm_head = Linear(&ctx_, config.hidden_size, config.vocab_size);

    // build state_dict
    state_dict_.reserve(3 + config.num_hidden_layers * 7);
    state_dict_.emplace_back("model.embed_tokens.weight", transformer.wte.weight);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::string layer_prefix = "model.layers." + std::to_string(i) + '.';
        state_dict_.emplace_back(layer_prefix + "input_layernorm.weight", transformer.layers[i].ln_1.weight);
        state_dict_.emplace_back(layer_prefix + "self_attn.W_pack.weight",
                                 transformer.layers[i].attn.c_attn.weight);
        state_dict_.emplace_back(layer_prefix + "self_attn.o_proj.weight",
                                 transformer.layers[i].attn.c_proj.weight);
        state_dict_.emplace_back(layer_prefix + "post_attention_layernorm.weight",
                                 transformer.layers[i].ln_2.weight);
        state_dict_.emplace_back(layer_prefix + "mlp.gate_proj.weight",
                                 transformer.layers[i].mlp.gate_proj.weight);
        state_dict_.emplace_back(layer_prefix + "mlp.down_proj.weight",
                                 transformer.layers[i].mlp.down_proj.weight);
        state_dict_.emplace_back(layer_prefix + "mlp.up_proj.weight",
                                 transformer.layers[i].mlp.up_proj.weight);
    }
    state_dict_.emplace_back("model.norm.weight", transformer.ln_f.weight);
    state_dict_.emplace_back("lm_head.weight", lm_head.weight);

}

ChatllmForCausalLM::~ChatllmForCausalLM() {}

auto ChatllmForCausalLM::generate_next_token(
        const std::vector<int32_t> &input_ids,
        const Config &gen_config,
        int n_past,
        int n_ctx
) -> int32_t {
    ctx_.ctx_b = chatllm::make_unique_ggml_context(ctx_.compute_buffer.size(), ctx_.compute_buffer.data(), false);
    ctx_.gf = {};

    int n_threads = 0; // user defined
    if (n_threads <= 0) {
        unsigned int n_thread = std::thread::hardware_concurrency();
        n_threads = n_thread > 0 ? (n_thread <= 4 ? n_thread : n_thread / 2) : 4;
        n_threads = std::min(n_threads, 16); // default thread num
    }
    int curr_input_ids_size = input_ids.size() - n_past;
    if (curr_input_ids_size >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas()) {
        n_threads = 1; // use 1 thread if BLAS is enabled
    }

    ggml_tensor *curr_input_ids = ggml_new_tensor_1d(ctx_.ctx_b.get(), GGML_TYPE_I32, curr_input_ids_size);

    memcpy(curr_input_ids->data, input_ids.data() + n_past, ggml_nbytes(curr_input_ids));

    ggml_tensor *KQ_pos = ggml_new_tensor_1d(ctx_.ctx_b.get(), GGML_TYPE_I32, curr_input_ids_size);
    int * data = static_cast<int *>(KQ_pos->data);
    for (int i = 0; i < curr_input_ids_size; ++i) {
        data[i] = n_past + i;
    }

    ggml_tensor *lm_logits = forward(&ctx_, curr_input_ids, KQ_pos, n_ctx);
    lm_logits->backend = GGML_BACKEND_CPU;

    ggml_build_forward_expand(&ctx_.gf, lm_logits);

    ggml_graph_compute_helper(ctx_.work_buffer, &ctx_.gf, n_threads);


    int vocab_size = lm_logits->ne[0];

    float *next_token_logits = (float *)lm_logits->data;

    // logits pre-process
    if (gen_config.repetition_penalty != 1.f) {
        sampling_repetition_penalty(next_token_logits, next_token_logits + vocab_size, input_ids,
                                    gen_config.repetition_penalty);
    }

    int next_token_id;

    // temperature sampling
    if (gen_config.temperature > 0) {
        sampling_temperature(next_token_logits, next_token_logits + vocab_size, gen_config.temperature);
    }

    std::vector<TokenIdScore> token_scores(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        token_scores[i] = TokenIdScore(i, next_token_logits[i]);
    }

    // top_k sampling
    if (0 < gen_config.top_k && gen_config.top_k < (int)token_scores.size()) {
        sampling_top_k(token_scores.data(), token_scores.data() + gen_config.top_k,
                       token_scores.data() + token_scores.size());
        token_scores.resize(gen_config.top_k);
    }

    // top_p sampling
    if (0.f < gen_config.top_p && gen_config.top_p < 1.f) {
        auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), gen_config.top_p);
        token_scores.resize(pos - token_scores.data());
    }

    // sample next token
    sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
    for (size_t i = 0; i < token_scores.size(); i++) {
        next_token_logits[i] = token_scores[i].score;
    }

    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());

    std::discrete_distribution<> dist(next_token_logits, next_token_logits + token_scores.size());
    next_token_id = token_scores[dist(gen)].id;


    return next_token_id;
}


auto ChatllmForCausalLM::sampling_repetition_penalty(
        float *first, float *last, const std::vector<int> &input_ids, float penalty
) -> void {
    std::unordered_set<int> unique_input_ids(input_ids.begin(), input_ids.end());
    for (int id : unique_input_ids) {
        if (first[id] > 0) {
            first[id] /= penalty;
        } else {
            first[id] *= penalty;
        }
    }
}

auto ChatllmForCausalLM::sampling_temperature(
        float *first, float *last, float temp
) -> void {
    float inv_temp = 1.f / temp;
    for (float *it = first; it != last; it++) {
        *it *= inv_temp;
    }
}

auto ChatllmForCausalLM::sampling_top_k(
        TokenIdScore *first, TokenIdScore *kth, TokenIdScore *last
) -> void {
    std::nth_element(first, kth, last, std::greater<TokenIdScore>());
}

auto ChatllmForCausalLM::sampling_top_p(
        TokenIdScore *first, TokenIdScore *last, float top_p
) -> TokenIdScore * {
    // fast top_p in expected O(n) time complexity
    sampling_softmax_inplace(first, last);

    while (first + 1 < last) {
        float pivot_score = (last - 1)->score; // use mid score?
        TokenIdScore *mid =
                std::partition(first, last - 1, [pivot_score](const TokenIdScore &x) { return x.score > pivot_score; });
        std::swap(*mid, *(last - 1));

        float prefix_sum =
                std::accumulate(first, mid, 0.f, [](float sum, const TokenIdScore &x) { return sum + x.score; });
        if (prefix_sum >= top_p) {
            last = mid;
        } else if (prefix_sum + mid->score < top_p) {
            first = mid + 1;
            top_p -= prefix_sum + mid->score;
        } else {
            return mid + 1;
        }
    }
    return last;
}

auto ChatllmForCausalLM::sampling_softmax_inplace(
        TokenIdScore *first, TokenIdScore *last
) -> void {
    float max_score = std::max_element(first, last)->score;
    float sum = 0.f;
    for (TokenIdScore *p = first; p != last; p++) {
        float s = std::exp(p->score - max_score);
        p->score = s;
        sum += s;
    }
    float inv_sum = 1.f / sum;
    for (TokenIdScore *p = first; p != last; p++) {
        p->score *= inv_sum;
    }
}

auto ChatllmForCausalLM::generate(
        const std::vector<int> &input_ids,
        const Config &gen_config,
        BaseStreamer *streamer
) -> std::vector<int> {
    std::vector<int> output_ids;
    output_ids.reserve(gen_config.model_max_length);
    output_ids = input_ids;

    int n_past = 0;
    const int n_ctx = input_ids.size();

    while ((int)output_ids.size() < gen_config.model_max_length) {
        auto next_token_id = generate_next_token(output_ids, gen_config, n_past, n_ctx);
//        std::cout << next_token_id << " " << std::endl;
        n_past = output_ids.size();
        output_ids.emplace_back(next_token_id);

        if (streamer) {
            streamer->put({next_token_id});
        }

        if (next_token_id == config.eos_token_id || next_token_id == config.bos_token_id) {
            break;
        }
    }

    return output_ids;
}

auto ChatllmForCausalLM::load(ModelLoader &loader) -> void {
    for (auto &item : state_dict_) {
        const std::string &name = item.first;
        ggml_tensor *tensor = item.second;
        loader.read_tensor(name, tensor);
    }

    ctx_.weight_buffer = std::string_view(loader.data, loader.size);
}

auto ChatllmForCausalLM::forward(
        ModelContext *ctx,
        ggml_tensor *input_ids,
        ggml_tensor *KQ_pos,
        int n_ctx
) const -> ggml_tensor * {

    ggml_tensor *transformer_outputs = transformer.forward(ctx, input_ids, KQ_pos, n_ctx);

    // NOTE: only compute next_token_logits for the last token
    if (input_ids->ne[0] > 1) {
        transformer_outputs = ggml_view_1d(ctx->ctx_b.get(), transformer_outputs, config.hidden_size,
                             (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs));
    }
    ggml_tensor *lm_logits = lm_head.forward(ctx, transformer_outputs);
    return lm_logits;
}