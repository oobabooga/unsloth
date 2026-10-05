/* Stands in for librocprofiler-sdk.so.1 under WSL, where rocprofiler-sdk 1.1 aborts: it enumerates
 * GPUs from the KFD sysfs topology, which the DXG bridge does not have. PyTorch links these entry
 * points for its profiler only; serving never calls them. rocprofiler-register finds a tool in torch
 * (Kineto's rocprofiler_configure) and hands the SDK the HIP API tables: accept and leave them. */
#define UNAVAILABLE 1
int rocprofiler_set_api_table(const char *name, unsigned long long version, unsigned long long instance, void **tables, unsigned long long count) { (void)name; (void)version; (void)instance; (void)tables; (void)count; return 0; }

int rocprofiler_is_initialized(int *status) { if (status) *status = 0; return 0; }
int rocprofiler_context_is_valid(unsigned long context, int *status) { (void)context; if (status) *status = 0; return 0; }
int rocprofiler_force_configure(void *configure) { (void)configure; return UNAVAILABLE; }
int rocprofiler_create_context(void *context) { (void)context; return UNAVAILABLE; }
int rocprofiler_start_context(unsigned long context) { (void)context; return UNAVAILABLE; }
int rocprofiler_stop_context(unsigned long context) { (void)context; return UNAVAILABLE; }
int rocprofiler_create_buffer() { return UNAVAILABLE; }
int rocprofiler_flush_buffer() { return UNAVAILABLE; }
int rocprofiler_configure_buffer_tracing_service() { return UNAVAILABLE; }
int rocprofiler_configure_callback_tracing_service() { return UNAVAILABLE; }
int rocprofiler_iterate_buffer_tracing_kind_operations() { return UNAVAILABLE; }
int rocprofiler_iterate_buffer_tracing_kinds() { return UNAVAILABLE; }
int rocprofiler_iterate_callback_tracing_kind_operation_args() { return UNAVAILABLE; }
int rocprofiler_iterate_callback_tracing_kind_operations() { return UNAVAILABLE; }
int rocprofiler_iterate_callback_tracing_kinds() { return UNAVAILABLE; }
int rocprofiler_query_available_agents() { return UNAVAILABLE; }
int rocprofiler_query_buffer_tracing_kind_name() { return UNAVAILABLE; }
int rocprofiler_query_buffer_tracing_kind_operation_name() { return UNAVAILABLE; }
int rocprofiler_query_callback_tracing_kind_name() { return UNAVAILABLE; }
int rocprofiler_query_callback_tracing_kind_operation_name() { return UNAVAILABLE; }
