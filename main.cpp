//
// Created by xiaoi on 12/29/23.
//
#include "chatllm.h"
#include <iomanip>
#include <iostream>

int main(){
    ggml_time_init();
    chatllm::Config config;

    config.model_path = "/opt/HuaZang/Baichuan2-13B-Chat/baichuan2-13b-chat-float.bin";
    std::unique_ptr<chatllm::FileMmap> file_mmap = std::make_unique<chatllm::FileMmap>(config.model_path);
    chatllm::ModelLoader loader(std::string_view((char *)file_mmap->data, file_mmap->size));
    std::string magic = loader.read_string(4);
    std::unique_ptr<chatllm::ChatllmForCausalLM> model = std::make_unique<chatllm::ChatllmForCausalLM>(config);
    model->load(loader);

    config.tokenizer_path = "/opt/HuaZang/Baichuan2-13B-Chat/tokenizer.model";
    chatllm::SPTokenizer sptokenizer;
    sptokenizer.tokenizer.Load(config.tokenizer_path);
    int64_t start_us_ = ggml_time_us();
    config.prompt = "噫吁嚱，危乎高哉，蜀道之难，难于上青天。蚕丛及鱼凫，开国何茫然。尔来四万八千岁，不与秦塞通人烟。西当太白有鸟道，可以横绝峨眉巅。地崩山摧壮士死，然后天梯石栈相钩连。上有六龙回日之高标，下有冲波逆折之回川。";
    std::vector<int> input_ids = sptokenizer.tokenizer.EncodeAsIds(config.prompt);
    input_ids.insert(input_ids.begin(), config.user_token_id);
    input_ids.insert(input_ids.end(), config.assistant_token_id);
    std::vector<int> output_ids = model->generate(input_ids, config);
    std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(), output_ids.end());
    std::string output = sptokenizer.tokenizer.DecodeIds(output_ids);
    int64_t end_us_ = ggml_time_us();
    std::cout << "The time of generating: " << output.c_str() << "is " << (end_us_ - start_us_) / 1000.f << "with time per token: " <<(end_us_ - start_us_) / 1000.f / output_ids.size()<< std::endl;

    return 0;
}