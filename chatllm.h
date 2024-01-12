//
// Created by xiaoi on 12/29/23.
//
#ifndef CHATLLM_CPP_CHATLLM_H
#define CHATLLM_CPP_CHATLLM_H
#include <ggml.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include "sentencepiece_processor.h"
#include <thread>

namespace chatllm {
    struct Config {
        //User define
        std::string model_path;
        std::string tokenizer_path;
        std::string prompt;

        //model config
        int bos_token_id = 1;
        int eos_token_id = 2;
        std::string hidden_act = "silu";
        int hidden_size = 5120;
        float initializer_range = 0.02;
        int intermediate_size = 13696;
        int model_max_length = 4096;
        std::string model_type = "baichuan";
        int num_attention_head = 40;
        int num_hidden_layers = 40;
        int pad_token_id = 0;
        float rms_norm_eps = 1e-06;
        bool tie_word_embeddings = false;
        std::string data_type = "bfloat16";
        bool use_cache = true;
        int vocab_size = 125696;

        //generation config
        int user_token_id = 195;
        int assistant_token_id = 196;
        int max_new_tokens = 2048;
        float temperature = 0.3;
        int top_k = 5;
        float top_p = 0.85;
        float repetition_penalty = 1.05;
        bool do_sample = true;
    };


    struct TokenIdScore {
        int id;
        float score;

        TokenIdScore() = default;
        TokenIdScore(int id, float score) : id(id), score(score) {}

        auto operator<(const TokenIdScore &other) const -> bool { return score < other.score; }
        auto operator>(const TokenIdScore &other) const -> bool { return score > other.score; }

        friend auto operator<<(std::ostream &os, const TokenIdScore &self) -> std::ostream & {
            return os << "TokenIdScore(id=" << self.id << ", score=" << self.score << ")";
        }
    };

    struct ggml_context_deleter_t {
        auto operator()(ggml_context *ctx) const noexcept -> void { ggml_free(ctx); }
    };
    using unique_ggml_context_t = std::unique_ptr<ggml_context, ggml_context_deleter_t>;

    static inline auto make_unique_ggml_context(
            size_t mem_size, void *mem_buffer, bool no_alloc
    ) -> unique_ggml_context_t {
        return unique_ggml_context_t(ggml_init({mem_size, mem_buffer, no_alloc}));
    }

    struct uninitialized_char {
        char m;
        uninitialized_char() {}
    };
    auto ggml_graph_compute_helper(std::vector<uninitialized_char> &buf, ggml_cgraph *graph, int n_threads) -> void;

    struct ModelContext {
        ggml_type dtype;
        unique_ggml_context_t ctx_w;  // weight
        unique_ggml_context_t ctx_kv; // kv cache
        unique_ggml_context_t ctx_b;  // buffer
        ggml_cgraph gf;
        ggml_scratch scratch;
        std::vector<uninitialized_char> compute_buffer; // BLAS buffer
        std::vector<uninitialized_char> scratch_buffer; // intermediate tensor buffer
        std::string_view weight_buffer;                 // mapped weight
        std::vector<uninitialized_char> work_buffer;    // temporary buffer for graph computing
    };

    class Embedding {
    public:
        Embedding() : weight(nullptr) {}
        Embedding(ModelContext *ctx, int num_embeddings, int embedding_dim)
                : weight(ggml_new_tensor_2d(ctx->ctx_w.get(), ctx->dtype, embedding_dim, num_embeddings)) {}

        auto forward(ModelContext *ctx, ggml_tensor *input) const -> ggml_tensor *;

        ggml_tensor *weight;
    };

    class Linear {
    public:
        Linear() : weight(nullptr) {}
        Linear(ModelContext *ctx, int in_features, int out_features)
                : weight(ggml_new_tensor_2d(ctx->ctx_w.get(), ctx->dtype, in_features, out_features)){}

        auto in_features() const -> int { return weight->ne[0]; }
        auto out_features() const -> int { return weight->ne[1]; }

        auto forward(ModelContext *ctx, ggml_tensor *input) const -> ggml_tensor *;

        ggml_tensor *weight; // [out_features, in_features]
    };

    class RMSNorm {
    public:
        RMSNorm() : weight(nullptr), inplace(true) {}
        RMSNorm(ModelContext *ctx, int normalized_shape, bool inplace = true)
                : weight(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace) {}

        auto forward(ModelContext *ctx, ggml_tensor *input, float eps = 1e-6f) const -> ggml_tensor *;

        ggml_tensor *weight;
        bool inplace;
    };

    class SPTokenizer{
    public:
        sentencepiece::SentencePieceProcessor tokenizer;
    };

    class FileMmap{
    public:
        FileMmap(std::string& model_path);
        ~FileMmap();

        char* data;
        size_t size;
    };

    class ModelLoader {
    public:
        ModelLoader(std::string_view buffer) : data(buffer.data()), size(buffer.size()), ptr(buffer.data()) {}

        auto tell() const -> int64_t { return ptr - data; }

        auto seek(int64_t offset, int whence) -> void;

        template <typename T>
        auto read_basic() -> T {
            T obj = *(T *)ptr;
            ptr += sizeof(T);
            return obj;
        }

        auto read_string(size_t length) -> std::string;

        auto read_tensor(const std::string &name, ggml_tensor *tensor) -> void;

    public:
        const char *const data;
        size_t size;
        const char *ptr;
    };

    class ChatllmAttention {
    public:
        ChatllmAttention():num_attention_heads(0) {}
        ChatllmAttention(ModelContext *ctx, int hidden_size, int num_attention_heads, int max_length);
        auto forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor *;

        int num_attention_heads;
        Linear c_attn;
        Linear c_proj;
        ggml_tensor *k_cache; // [n_head, maxlen, head_size]
        ggml_tensor *v_cache; // [n_head, head_size, maxlen]
    };

    class ChatllmMLP {
    public:
        ChatllmMLP() = default;
        ChatllmMLP(ModelContext * ctx, int hidden_size, int intermediate_size)
        : gate_proj(ctx, hidden_size, intermediate_size),
          down_proj(ctx, intermediate_size, hidden_size),
          up_proj(ctx, hidden_size, intermediate_size) {}

        auto forward(ModelContext *ctx, ggml_tensor *hidden_states) const -> ggml_tensor *;

        Linear gate_proj;
        Linear down_proj;
        Linear up_proj ;
    };

    class ChatllmBlock {
    public:
        ChatllmBlock() = default;
        ChatllmBlock(ModelContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
                : ln_1(ctx, hidden_size, false),
                  attn(ctx, hidden_size, num_attention_heads, max_length),
                  ln_2(ctx, hidden_size, false),
                  mlp(ctx, hidden_size, intermediate_size) {}

        auto forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor *;

        RMSNorm ln_1;
        ChatllmAttention attn;
        RMSNorm ln_2;
        ChatllmMLP mlp;
    };

    class ChatllmModel {
    public:
        ChatllmModel() = default;
        ChatllmModel(ModelContext *ctx, const Config &config);

        auto forward(ModelContext *ctx, ggml_tensor *input_ids, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor *;

        Embedding wte;
        std::vector<ChatllmBlock> layers;
        RMSNorm ln_f;
    };

    class ChatllmForCausalLM {
    public:
        ChatllmForCausalLM(const Config &config);
        ~ChatllmForCausalLM();

        auto generate_next_token(
                const std::vector<int> &input_ids,
                const Config &gen_config,
                int n_past,
                int n_ctx
        ) -> int;

        auto generate(
                const std::vector<int> &input_ids,
                const Config &gen_config
        ) -> std::vector<int>;

        // logits processor
        static auto sampling_repetition_penalty(float *first, float *last, const std::vector<int32_t> &input_ids,
                                                float penalty) -> void;
        // logits warper
        static auto sampling_temperature(float *first, float *last, float temp) -> void;
        static auto sampling_top_k(TokenIdScore *first, TokenIdScore *kth, TokenIdScore *last) -> void;
        static auto sampling_top_p(TokenIdScore *first, TokenIdScore *last, float top_p) -> TokenIdScore *;

        static auto sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last) -> void;

        auto load(ModelLoader &loader) -> void;

        auto forward(ModelContext *ctx, ggml_tensor *input_ids, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor *;

        static constexpr size_t MEM_SIZE     = 512 * 1024 * 1024; // 2k context
        static constexpr size_t SCRATCH_SIZE = 1280 * 1024 * 1024; // 2k context

        Config config;
        ChatllmModel transformer;
        Linear lm_head;

    private:
        ModelContext ctx_;
        std::vector<std::pair<std::string, ggml_tensor *>> state_dict_;
    };


}


#endif //CHATLLM_CPP_CHATLLM_H
