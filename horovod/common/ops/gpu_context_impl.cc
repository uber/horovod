GPUContext::GPUContext() : pimpl{new impl} {}
GPUContext::~GPUContext() = default;

void GPUContext::Finalize() {
  finalizer_thread_pool.reset();
}

void GPUContext::ErrorCheck(std::string op_name, gpuError_t gpu_result) {
  pimpl->ErrorCheck(op_name, gpu_result);
}

void GPUContext::RecordEvent(std::queue<std::pair<std::string, Event>>& event_queue, std::string name, gpuStream_t& stream) {
  pimpl->RecordEvent(event_queue, name, stream);
}

Event GPUContext::RecordEvent(gpuStream_t& stream) {
  return pimpl->RecordEvent(stream);
}

void GPUContext::ReleaseEvent(Event event) {
  pimpl->ErrorCheck("ReleaseGpuEvent", pimpl->ReleaseGpuEvent(event));
}

void GPUContext::WaitForEvents(std::queue<std::pair<std::string, Event>>& event_queue, const std::vector<TensorTableEntry>& entries,
                               Timeline& timeline, const std::function<void()>& error_check_callback,
                               bool elastic) {
  pimpl->WaitForEvents(event_queue, entries, timeline, error_check_callback, elastic);
}

void GPUContext::ClearEvents(std::queue<std::pair<std::string, Event>>& event_queue, const std::vector<TensorTableEntry>& entries,
                             Timeline& timeline, const std::function<void()>& error_check_callback,
                             bool elastic) {
  pimpl->ClearEvents(event_queue, entries, timeline, error_check_callback, elastic);
}

void GPUContext::StreamCreate(gpuStream_t *stream) {
  pimpl->StreamCreate(stream);
}

void GPUContext::StreamSynchronize(gpuStream_t stream) {
  pimpl->StreamSynchronize(stream);
}

int GPUContext::GetDevice() {
  return pimpl->GetDevice();
}

void GPUContext::SetDevice(int device) {
  pimpl->SetDevice(device);
}

void GPUContext::MemcpyAsyncD2D(void* dst, const void* src, size_t count, gpuStream_t stream) {
  pimpl->MemcpyAsyncD2D(dst, src, count, stream);
}

void GPUContext::MemcpyAsyncH2D(void* dst, const void* src, size_t count, gpuStream_t stream) {
  pimpl->MemcpyAsyncH2D(dst, src, count, stream);
}

void GPUContext::MemcpyAsyncD2H(void* dst, const void* src, size_t count, gpuStream_t stream) {
  pimpl->MemcpyAsyncD2H(dst, src, count, stream);
}

void GPUContext::ScaleBufferImpl(const void* fused_input_data, void* buffer_data, int64_t num_elements,
                                 double scale_factor, DataType dtype, gpuStream_t stream) {
  pimpl->ScaleBufferImpl(fused_input_data, buffer_data, num_elements, scale_factor, dtype, stream);
}

