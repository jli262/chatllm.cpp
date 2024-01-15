//
// Created by xiaoi on 12/29/23.
//
#include "chatllm.h"
#include <iomanip>
#include <iostream>

static auto get_utf8_line(std::string &line) -> bool {
    return !!std::getline(std::cin, line);
}

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

    std::cout << R"(  ____ _           _   _     _     __  __                   )" << '\n'
              << R"( / ___| |__   __ _| |_| |   | |   |  \/  |  ___ _ __  _ __  )" << '\n'
              << R"(| |   | '_ \ / _` | __| |   | |   | |\/| | / __| '_ \| '_ \ )" << '\n'
              << R"(| |___| | | | (_| | |_| |___| |___| |  | || (__| |_) | |_) |)" << '\n'
              << R"( \____|_| |_|\__,_|\__|_____|_____|_|  |_(_)___| .__/| .__/ )" << '\n'
              << R"(                                               |_|   |_|    )" << '\n'
              << '\n';
    std::cout
            << "Welcome to ChatLLM.cpp! Ask whatever you want. Type 'clear' to clear context. Type 'stop' to exit.\n"
            << "\n";

    std::string model_name = "ChatLLM";
    auto text_streamer = std::make_shared<chatllm::TextStreamer>(std::cout, &sptokenizer);
    auto perf_streamer = std::make_shared<chatllm::PerfStreamer>();
    auto streamer = std::make_shared<chatllm::StreamerGroup>(
            std::vector<std::shared_ptr<chatllm::BaseStreamer>>{text_streamer, perf_streamer});

    std::vector<int> history;
    while (1) {
        std::cout << std::setw(model_name.size()) << std::left << "Prompt"
                  << " > " << std::flush;
        std::string prompt;
        if (!get_utf8_line(prompt) || prompt == "stop") {
            break;
        }
        if (prompt.empty()) {
            continue;
        }
        if (prompt == "clear") {
            history.clear();
            continue;
        }
        std::vector<int> input_ids = sptokenizer.tokenizer.EncodeAsIds(std::move(prompt));

        input_ids.insert(input_ids.begin(), config.user_token_id);
        input_ids.insert(input_ids.end(), config.assistant_token_id);
        history.insert(history.end(), input_ids.begin(), input_ids.end());
        if((int)history.size() > config.model_max_length){
            history.erase(history.begin(), history.end() - config.model_max_length);
        }
        std::cout << model_name << " > ";
        std::vector<int> output_ids = model->generate(history, config, streamer.get());
        std::vector<int> new_output_ids(output_ids.begin() + history.size(), output_ids.end());
        std::string output = sptokenizer.tokenizer.DecodeIds(output_ids);
        history.insert(history.end(), new_output_ids.begin(), new_output_ids.end());
        std::cout << "\n" << perf_streamer->to_string() << "\n\n";

        perf_streamer->reset();
    }
//    int64_t start_us_ = ggml_time_us();
//    config.prompt = "现代社会如何看待“苟利国家生死以”？";
//    std::vector<int> input_ids = sptokenizer.tokenizer.EncodeAsIds(config.prompt);
//    input_ids.insert(input_ids.begin(), config.user_token_id);
//    input_ids.insert(input_ids.end(), config.assistant_token_id);
//    std::vector<int> output_ids = model->generate(input_ids, config);
//    std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(), output_ids.end());
//    std::string output = sptokenizer.tokenizer.DecodeIds(output_ids);
//    int64_t end_us_ = ggml_time_us();
//    std::cout << "The time of generating: \"" << output.c_str() << "\" is " << (end_us_ - start_us_) / 1000.f << " with time per token: " <<(end_us_ - start_us_) / 1000.f / output_ids.size()<< std::endl;

    return 0;
}