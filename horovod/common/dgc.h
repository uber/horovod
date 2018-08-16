// DGC definations
// by Yuechao Pan
// for NVIDIA

#pragma once

#include <map>
#include <unordered_map>
#include <list>
#include <math.h>
#include <mpi.h>
#include <nccl.h>
#include <curand_kernel.h>
#include "common.h"
#include "types.h"

namespace horovod {
namespace dgc {

// Configuration for DGC
struct DgcConfig {
  // The number of warmup epoches for DGC.
  // DGC communication will use a gradient sparsity, which starts from
  // init_sparsity in the first epoch, and exponentially increases to
  // final_sparsity after warmup epoches.
  double warmup_epochs = 4.0;

  // Each epoch has (num_examples_per_epoch / (global_num_gpus * batch_size_per_gpu)
  // steps
  int num_examples_per_epoch = 1000000;
  int batch_size_per_gpu = 32;

  // Initial gradient sparsity for DGC.
  double init_sparsity = 0.75;

  // Final gradient sparsity for DGC, after the warmup epoches.
  double final_sparsity = 0.999;

  // Sampling rate for top-k selection in DGC
  double sampling_rate = 0.01;

  // dgc rand seed
  unsigned int rand_seed = 2800;

  // dgc grid and block sizes
  int grid_size = 32;
  int block_size = 512;

  // stream DGC works on
  cudaStream_t stream = 0;
  cudaStream_t stream2 = 0;
  cudaStream_t stream3 = 0;
  cudaStream_t stream4 = 0;

  // number of GPUs in all nodes
  int global_num_gpus = 1;

  // global GPU rank
  int global_gpu_rank = 0;

  // number of nodes
  int global_num_nodes = 1;

  // node rank
  int global_node_rank = 0;

  // number of GPUs in local node
  int local_num_gpus = 1;

  // local GPU rank
  int local_gpu_rank = 0;

  // NCCL communication handle
  ncclComm_t nccl_comm;
  MPI_Comm mpi_comm;
  MPI_Comm cross_comm;
  MPI_Comm local_comm;

  // whether DgcConfig has been configured
  bool configured = false;

  // the minimum number of elements to trigger sampling
  uint64_t min_sampling_num = 4000;

  // the minimum number of selected elements per layer
  uint64_t min_gradients_comm_per_layer = 10;

  // the minimum number of selected samples per layer
  uint64_t min_selected_samples_per_layer = 5;

  // Momentum
  float momentum = 0.9;

  // Whether to use local gradient clipping
  bool local_gradient_clipping = true;

  // Gradient clipping threshold
  float clipping_threshold = 6.0;

  // Whether to use allReduce instead of allGather for gradient communication
  bool use_allReduce = true;

  // Whether to use hierarchical_allreduce
  bool use_hierarchical_allreduce = true;

  // NCCL communicator for cross node communication, only GPU0
  ncclComm_t nccl_cross_comm;
  ncclComm_t nccl_local_comm;
  bool cross_comm_inited = false;

  // Learning rate decay parameters
  double learning_rate_decay_factor = 0.1;
  double num_epochs_per_decay = 30;
  double min_learning_rate_factor = 1e-6;

  // Number of steps per whole model gradient flush, 0 to disable
  uint64_t flush_steps = 0;

  // Whether to overlap MPI AllReduce on masks
  bool overlap_mask_allreduce = false;

  // Steps to skip overlap
  int overlap_skip_steps = 20;

  // Whether to use momentum correction; if not, directly accumulate gradients
  bool use_momentum_correction = true;

  // Whether to accumulate gradients, if momentum correction is not in use
  bool use_gradient_accumulation = true;

  // Whether to smooth out sparsity changes
  bool smooth_sparsity = false;

  // Horovod operation context
  std::shared_ptr<horovod::common::OpContext> context;

  // GPU index, or CPU_DEVICE_ID in case of CPU
  int device = CPU_DEVICE_ID;

  // function to set indivual configuration
  void Set(std::string key, std::string value);

  // Get configuration from environmental variables
  void ReadFromENV();
};

// Token for GradientAllReduce call, mainly for host arrays,
// to avoid CPU-side data being overwrote before moving to GPU,
// if another call happens before GPU operations of pervious calls
// are executed.
struct DgcToken {
  // Gradient layer and sample layer starts
  uint32_t* h_layer_starts = NULL;
  uint32_t  h_layer_starts_allocated = 0;
  uint32_t* h_samp_starts = NULL;
  uint32_t  h_samp_starts_allocated = 0;

  cudaEvent_t dgc_finish, stream2_begin, stream2_finish, stream3_begin;

  bool dgc_finished = false;
  cudaError_t Init();
  cudaError_t isFinished(bool &finished, int check = 0);
};

typedef uint32_t MaskT;
const int MASK_BITS = 32;
const int LOG_MASK_BITS = 5;
const MaskT MASK_BITS_MASK = (1 << LOG_MASK_BITS) - 1;

// Token for overlapping MPI Reduce of mask and computation
struct MaskToken {
  MaskT*   h_send_masks = NULL;
  MaskT*   h_recv_masks = NULL;
  uint64_t    num_masks    = 0;
  uint32_t    num_layers   = 0;
  uint32_t    num_layers_produced = 0;
  uint32_t    num_layers_comsumed = 0;
  uint64_t    mask_allocated = 0;
  cudaEvent_t d2h_finish;
  bool        d2h_finished = false;
  cudaEvent_t h2d_finish;
  bool        h2d_finished = false;

  MPI_Request mpi_request;
  bool        mpi_started  = false;
  bool        mpi_finished = false;

  cudaError_t Init();
  cudaError_t isFinished(bool &finished, int check = 0);
};

// Per layer information for overlapping mask communication
struct LayerRecord {
  MaskToken *token = NULL;
  uint32_t   layer_start = 0;
};

// Running state, including memory allocation of DGC
struct DgcState {

  // States for curand, one for each GPU thread
  // curandState *rand_states     = NULL;

  // Accumulated gradients
  //char     *accumulated_gradients = NULL;
  //uint64_t  accumulated_gradients_allocated = 0;

  //char     *pervious_accumulated_gradients = NULL;
  //uint64_t  pervious_accumulated_gradients_allocated = 0;

  // Verlocity
  //char     *verlocity          = NULL;
  //uint64_t  verlocity_allocated = 0;

  // Past verlocity
  //char     *pervious_verlocity = NULL;
  //uint64_t  pervious_verlocity_allocated = 0;

  // Accumulated verlociy
  //char     *accumulated_verlocity = NULL;
  //uint64_t  accumulated_verlocity_allocated = 0;

  //char     *pervious_accumulated_verlocity = NULL;
  //uint64_t  pervious_accumulated_verlocity_allocated = 0;

  // Sample counter
  //uint64_t *samp_counter       = NULL;

  // Sample data, raw data in chars; need to convert type before using
  //char     *samp_data          = NULL;
  //uint64_t  samp_allocated     = 0;

  // Gradient selection threshold
  //float    *thresholds = NULL;
  //uint64_t  thresholds_allocated = 0;

  // Counter for gradient selection
  //uint64_t *send_counter       = NULL;

  // Number of allocated elements for selected data
  //uint64_t  send_allocated     = 0;

  // Memory for selected gradient and indices
  //char     *send_data          = NULL;
  //uint32_t *send_indices       = NULL;

  // Number of allocated elements for recv data
  //uint64_t  recv_allocated     = 0;

  // Memory for recved gradient and indices
  //char     *recv_data          = NULL;
  //uint32_t *recv_indices       = NULL;

  // Number of allocated elements for global gradients
  //uint64_t  global_allocated   = 0;

  // Global gradients
  //char     *global_gradients   = NULL;

  // layer offset address book
  std::map<std::string, size_t  > layer_offset_bytes;

  // Per-tensor step counter
  std::map<std::string, uint64_t> step_counters;

  // Step number
  uint64_t step = 0;

  // Epoch number
  double epoch = 0;

  // Counter for adding new tensor to the end of memory space
  size_t offset_byte_counter = 0;

  // Temp storage
  //char* temp_storage = NULL;
  //size_t temp_storage_bytes = 0;
  //char* temp_storage2 = NULL;
  //size_t temp_storage2_bytes = 0;
  //char* temp_storage3 = NULL;
  //size_t temp_storage3_bytes = 0;

  // Maximum gradient
  //float* max_gradient = NULL;

  // Gradient and sample starts for each layer
  //uint32_t* layer_starts = NULL;
  //uint32_t  layer_starts_allocated = 0;
  //uint32_t* samp_starts = NULL;
  //uint32_t  samp_starts_allocated = 0;

  // Gradient selection mask for allReduce communication
  //MaskT* send_masks = NULL;
  //MaskT* recv_masks = NULL;
  MaskT* h_send_masks = NULL;
  MaskT* h_recv_masks = NULL;
  uint64_t  mask_allocated = 0;

  //uint32_t* mask_counters = NULL;
  //uint64_t  mask_counters_allocated = 0;
  //uint32_t* mask_offsets  = NULL;
  //uint64_t  mask_offsets_allocated = 0;

  uint32_t* h_num_gradients_to_communicate = NULL;

  // Tokens
  std::list<DgcToken*> free_tokens;
  std::list<DgcToken*> busy_tokens;

  std::list<MaskToken*> free_mask_tokens;
  std::list<MaskToken*> d2h_mask_queue;
  std::list<MaskToken*> mpi_mask_queue;
  std::list<MaskToken*> h2d_mask_queue;

  // Layer records
  std::map<std::string, LayerRecord> layer_records[2];

  // Memory buffer for DGC usage, keyed by "device ID::framework::name"
  std::unordered_map<std::string, //std::tuple<int, common::Framework, std::string>,
    std::pair<std::shared_ptr<common::PersistentBuffer>, size_t> >
    //std::shared_ptr<common::PersistentBuffer> >
    memory_table;

  //std::unordered_map<std::string, uint64_t> size_table;
};

// Entry warper function
cudaError_t GradientAllReduce(
  ncclDataType_t  gradient_type, // type of gradient
  void           *input_gradients, // GPU pointer to the input graients
  void           *output_gradients,// GPU pointer to the output gradients
  std::vector<std::pair<std::string, uint64_t> > &layers,
                                // <name, #elements> of layers
  DgcConfig      &config,       // DGC configuration
  DgcState       &state);       // DGC running states

} // end of namespace dgc
} // end of namespace horovod
