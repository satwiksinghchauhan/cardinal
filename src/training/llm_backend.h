// =============================================================================
// PATCH for src/core/llm_backend.h — add these two virtual methods to the
// ILLMBackend class declaration (alongside generate_response etc.):
// =============================================================================
//
//     // LoRA adapter management (new in v1.4.0)
//     // LlamaCppBackend implements using llama_adapter_lora_init +
//     //   llama_set_adapters_lora (scale 1.0) / llama_set_adapters_lora (0 adapters).
//     // TensorRTBackend returns false / no-op.
//     virtual bool load_lora_adapter  (const std::string& gguf_path) { (void)gguf_path; return false; }
//     virtual void unload_lora_adapter()                              {}
//
// Both have default no-op implementations so TensorRTBackend and any mock
// backends need no changes.
//
// In src/core/backends/llama_cpp_backend.h add to the class:
//     bool load_lora_adapter  (const std::string& gguf_path) override;
//     void unload_lora_adapter()                              override;
//
// In src/core/backends/llama_cpp_backend.cpp add:
//
//     bool LlamaCppBackend::load_lora_adapter(const std::string& gguf_path) {
//         // llama_adapter_lora_init takes llama_model*, not llama_context*.
//         llama_adapter_lora* adapter =
//             llama_adapter_lora_init(model_, gguf_path.c_str());
//         if (!adapter) return false;
//
//         // llama_set_adapters_lora(ctx, adapters**, n_adapters, scales*)
//         float scale = 1.0f;
//         llama_set_adapters_lora(ctx_, &adapter, 1, &scale);
//
//         if (active_lora_adapter_) {
//             llama_adapter_lora_free(active_lora_adapter_);
//         }
//         active_lora_adapter_ = adapter;
//         return true;
//     }
//
//     void LlamaCppBackend::unload_lora_adapter() {
//         // Pass 0 adapters to clear all active adapters from the context.
//         llama_set_adapters_lora(ctx_, nullptr, 0, nullptr);
//         if (active_lora_adapter_) {
//             llama_adapter_lora_free(active_lora_adapter_);
//             active_lora_adapter_ = nullptr;
//         }
//     }
//
// Add to LlamaCppBackend private members:
//     llama_adapter_lora* active_lora_adapter_ = nullptr;
//
// FeelingContext::last_output (used by quick_infer_confidence):
// If FeelingContext does not have a last_output field, the fallback path
// in quick_infer_confidence already returns 0.5f safely — no change needed.
// =============================================================================

// This file is a patch guide only — it is not compiled.
// Apply the changes above manually to the respective files.
