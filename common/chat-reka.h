// ABOUTME: Declares the Reka Edge chat template handler.
// ABOUTME: Provides tool calling, thinking, and vision token support for Reka Edge.

#pragma once

#include "chat.h"

namespace autoparser {
struct generation_params;
}

common_chat_params common_chat_params_init_reka_edge(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs);
