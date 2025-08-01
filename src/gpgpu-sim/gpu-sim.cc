// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpu-sim.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"

extern unsigned stream_num;

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

// #define MAX(a, b) (((a) > (b)) ? (a) : (b)) //redefined

bool g_interactive_debugger_enabled = false;

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

std::map<unsigned, unsigned> core_stream_mapping;
std::map<std::string, PolicyStats> policy_performance_stats;
std::map<std::string, std::vector<PolicyStats>> policy_performance_stats_vector;
// the global_stream_core_ranges_set could be merged with global_dynamic_core_ranges
std::map<unsigned long long, std::set<unsigned>> global_dynamic_core_ranges;
std::map<unsigned long long, std::vector<unsigned>> global_dynamic_core_ranges_vector;
std::map<unsigned long long, std::set<unsigned>> global_stream_core_ranges_set;
std::map<unsigned long long, std::vector<unsigned>> global_stream_core_ranges_vector;
std::map<unsigned long long, std::pair<unsigned, unsigned>> global_stream_core_ranges;
std::set<unsigned long long> global_unique_streams;
std::map<unsigned,long long > stream_inst_count;

bool is_policy_change = false;
bool is_policy_change_done = false;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");

  option_parser_register(opp, "-hw_perf_file_name", OPT_CSTR,
                         &g_hw_perf_file_name,
                         "Hardware Performance Statistics file", "hw_perf.csv");

  option_parser_register(
      opp, "-hw_perf_bench_name", OPT_CSTR, &g_hw_perf_bench_name,
      "Kernel Name in Hardware Performance Statistics file", "");

  option_parser_register(opp, "-power_simulation_mode", OPT_INT32,
                         &g_power_simulation_mode,
                         "Switch performance counter input for power "
                         "simulation (0=Sim, 1=HW, 2=HW-Sim Hybrid)",
                         "0");

  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL, &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
  option_parser_register(opp, "-aggregate_power_stats", OPT_BOOL,
                         &g_aggregate_power_stats,
                         "Accumulate power across all kernels", "0");

  // Accelwattch Hyrbid Configuration

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RH],
      "Get L1 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RM],
      "Get L1 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WH],
      "Get L1 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WM],
      "Get L1 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RH],
      "Get L2 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RM],
      "Get L2 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WH],
      "Get L2 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WM],
      "Get L2 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CC_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CC_ACC],
      "Get Constant Cache Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_SHARED_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_SHRD_ACC],
      "Get Shared Memory Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_RD", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_RD],
                         "Get DRAM Reads for Accelwattch-Hybrid from Accel-Sim",
                         "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_DRAM_WR", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_DRAM_WR],
      "Get DRAM Writes for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NOC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NOC],
      "Get Interconnect Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_PIPE_DUTY", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_PIPE_DUTY],
      "Get Pipeline Duty Cycle Acesses for Accelwattch-Hybrid from Accel-Sim",
      "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NUM_SM_IDLE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NUM_SM_IDLE],
      "Get Number of Idle SMs for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CYCLES", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CYCLES],
      "Get Executed Cycles for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_VOLTAGE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_VOLTAGE],
      "Get Chip Voltage for Accelwattch-Hybrid from Accel-Sim", "0");

  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");

  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(
      opp, "-gpgpu_cache:dl2", OPT_CSTR, &m_L2_config.m_config_string,
      "unified banked L2 data cache config "
      "{<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>:<set_"
      "index_fn>,<mshr>:<N>:<merge>,<mq>:<fifo_entry>,<data_port_width>",
      "S:32:128:24,L:B:m:L:P,A:192:4,32:0,32");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(opp, "-gpgpu_fill_entire_line", OPT_BOOL,
                         &m_fill_entire_line, 
                         "Fill entire cache line instead of just sectors in L1/L2 cache (default = no)",
                         "0");
  option_parser_register(opp, "-gpgpu_fill_entire_line_on_clean_l1", OPT_BOOL,
                         &m_fill_entire_line_on_cleanl1, 
                         "Fill entire cache line on clean instead of just sectors in L1 cache (default = no)",
                         "0");                       
  option_parser_register(opp, "-gpgpu_fill_entire_line_on_clean_l2", OPT_BOOL,
                         &m_fill_entire_line_on_cleanl2, 
                         "Fill entire cache line on clean instead of just sectors in L1 cache (default = no)",
                         "0");
  option_parser_register(opp, "-gpgpu_dynamic_fetch_mem_cache", OPT_BOOL,
                         &m_dynamic_fetch_mem,
                         "Enable dynamic fetch mode (default = no)",
                         "0");
  option_parser_register(opp, "-gpgpu_dynamic_fetch_size_cache", OPT_UINT32,
                         &m_dynamic_fetch_size,
                         "Dynamic fetch size (default = 32)",
                         "32");
  option_parser_register(opp, "-collect_sector_stats", OPT_BOOL,
                         &collect_sector_stats, 
                         "collect sector cache statistics (default = no)",
                         "0");                         
  // Cache partitioning options
  option_parser_register(
      opp, "-gpgpu_cache_stream_partitioning", OPT_BOOL, &gpgpu_cache_stream_partitioning,
      "Enable stream-based cache partitioning for L1 and L2 caches (default = disabled)", "0");
  option_parser_register(
      opp, "-gpgpu_cache_stream0_percentage", OPT_FLOAT, &gpgpu_cache_stream0_percentage,
      "Percentage of cache allocated to stream 0 (0.0-1.0, default = 0.4)", "0.4");
  option_parser_register(
      opp, "-gpgpu_cache_stream1_percentage", OPT_FLOAT, &gpgpu_cache_stream1_percentage,
      "Percentage of cache allocated to stream 1 (0.0-1.0, default = 0.6)", "0.6");

  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  // SST mode activate
  option_parser_register(opp, "-SST_mode", OPT_BOOL, &SST_mode, "SST mode",
                         "0");
  m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(
      opp, "-gpgpu_cache:il1", OPT_CSTR, &m_L1I_config.m_config_string,
      "shader L1 instruction cache config "
      " {<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
      "alloc>:<set_index_fn>,<mshr>:<N>:<merge>,<mq>} ",
      "N:64:128:16,L:R:f:N:L,S:2:48,4");
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<"
                         "alloc>:<wr_alloc>:<set_index_fn>,<mshr>:<N>:<merge>,<"
                         "mq>:<fifo_entry>,<data_port_width> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_cache_write_ratio", OPT_UINT32,
                         &m_L1D_config.m_wr_percent, "L1D write ratio", "0");
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<"
                         "alloc>:<wr_alloc>:<set_index_fn>,<mshr>:<N>:<merge>,<"
                         "mq>:<fifo_entry>,<data_port_width> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<"
                         "alloc>:<wr_alloc>:<set_index_fn>,<mshr>:<N>:<merge>,<"
                         "mq>:<fifo_entry>,<data_port_width> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D_stream0", OPT_BOOL, &gmem_skip_L1D_stream0,
                         "global memory access skip L1D cache for stream 0 (default = no skip)",
                         "0");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D_stream1", OPT_BOOL, &gmem_skip_L1D_stream1,
                         "global memory access skip L1D cache for stream 1 (default = no skip)",
                         "0");
  option_parser_register(opp, "-gpgpu_share_L1D", OPT_BOOL, &m_L1D_config.m_shareL1,
                         "share L1D cache between shader cores (default = off)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_sm_shareL1", OPT_UINT32, &m_L1D_config.m_num_sm_shareL1,
                         "number of shader cores sharing L1D cache (default = 4)",
                         "4");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(opp, "-gpgpu_profile_ipc_instnum_threshold", OPT_UINT32,
                         &profile_ipc_instnum_threshold,
                         "threshold of instructions to profile ipc (default = 10000)",
                         "10000");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 32)", "32");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_option", OPT_CSTR,
                         &gpgpu_shmem_option,
                         "Option list of shared memory sizes", "0");
  option_parser_register(
      opp, "-gpgpu_unified_l1d_size", OPT_UINT32,
      &m_L1D_config.m_unified_cache_size,
      "Size of unified data cache(L1D + shared memory) in KB", "0");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_BOOL,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");

  option_parser_register(
      opp, "-gpgpu_dynamic_fetch_mem", OPT_BOOL,
      &gpgpu_dynamic_fetch_mem,
      "dynamically change size of request memory fetches (default = off)",
      "0"
  );
  option_parser_register(opp, "-gpgpu_dynamic_fetch_size", OPT_INT32,
       &gpgpu_dynamic_fetch_size,
       "dynamic memory fetch size", "128");

  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_UINT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_UINT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_UINT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_UINT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_UINT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_UINT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_UINT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");

  
  // stream-based partitioning is enabled by default
  option_parser_register(
      opp, "-gpgpu_stream_based_allocation", OPT_BOOL, &gpgpu_stream_partitioning,
      "Enable stream-based core allocation with spatial partitioning (default = enabled)", "1");

  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_executed_kernel", OPT_INT32,
                         &gpgpu_max_executed_kernel_num,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-gpgpu_stream_intlv_core", OPT_BOOL, &gpgpu_stream_intlv_core,
  "interleave the cores of the two streams (default = disabled)", "0");
    option_parser_register(opp, "-gpgpu_dynamic_core_scheduling", OPT_BOOL, &gpgpu_dynamic_core_scheduling,
      "dynamic core scheduling (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_dyno_core_scheduling", OPT_BOOL, &gpgpu_dyno_core_scheduling,
      "dyno core scheduling (default = disabled)", "0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU, set this value "
      "according to max resident grids for your compute capability",
      "32");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");

  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

void gpgpu_sim::launch(kernel_info_t *kinfo) {
  unsigned kernelID = kinfo->get_uid();
  unsigned long long streamID = kinfo->get_streamID();

  kernel_time_t kernel_time = {gpu_tot_sim_cycle + gpu_sim_cycle, 0};
  if (gpu_kernel_time.find(streamID) == gpu_kernel_time.end()) {
    std::map<unsigned, kernel_time_t> new_val;
    new_val.insert(std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    gpu_kernel_time.insert(
        std::pair<unsigned long long, std::map<unsigned, kernel_time_t>>(
            streamID, new_val));
  } else {
    gpu_kernel_time.at(streamID).insert(
        std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    ////////// assume same kernel ID do not appear more than once
  }

  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      if (kinfo->get_streamID() == 0) {
        m_running_kernels_stream1.push_back(kinfo);
      } else {
        m_running_kernels_stream2.push_back(kinfo);
      }
      printf("kernel init %i %s streamID %llu kernelID %u, core range %s\n",
             n, kinfo->name().c_str(), kinfo->get_streamID(),
             kinfo->get_uid(), kinfo->print_core_range().c_str());
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

void gpgpu_sim::restart_stream(unsigned long long stream_id) {
  std::cout << "Restarting stream " << stream_id << std::endl;
  //if (stream_id == 0) {
  //  m_running_kernels_stream1.clear();
  //} else {
  //  m_running_kernels_stream2.clear();
  //}
}

bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

kernel_info_t *gpgpu_sim::select_kernel() {
  if (m_running_kernels[m_last_issued_kernel] &&
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      m_running_kernels[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      assert(std::find(m_executed_kernel_uids.begin(),
                       m_executed_kernel_uids.end(),
                       launch_uid) == m_executed_kernel_uids.end());
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(m_running_kernels[idx]->name());

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

kernel_info_t *gpgpu_sim::select_kernel(unsigned core_id) {
  kernel_info_t *kernel = NULL;
  unsigned total_cores = m_config.num_shader();
  // choose which stream current core belongs to
  
  bool isStreamOne = false;
  
  if (m_running_kernels_stream2.empty()) {
    isStreamOne = m_running_kernels_stream2.empty();
  } else if (m_config.get_stream_intlv_core()) {
    if (core_id % 2 == 0)
      isStreamOne = true;
  } else if (m_config.get_dynamic_core_scheduling() ||
             m_config.get_dyno_core_scheduling()) {
    isStreamOne = core_stream_mapping[core_id] == 0;
  } else {
    isStreamOne = core_id < total_cores/2;
  }

  if (isStreamOne ) {
    kernel = select_kernel(core_id, m_running_kernels_stream1, m_last_issued_kernel_stream1);
  } else {
    kernel = select_kernel(core_id, m_running_kernels_stream2, m_last_issued_kernel_stream2);
  }

  // in dynamic core mode, check the core id with kernel stream ID
  if (m_config.get_dynamic_core_scheduling()||
      m_config.get_dyno_core_scheduling()) {
    if (kernel) {
      if (kernel->get_streamID() != core_stream_mapping[core_id]) {
        //printf("kernel stream ID %u, core stream ID %u, kernel name %s\n",
        //kernel->get_streamID(), core_stream_mapping[core_id], kernel->name().c_str());
        assert (false);
      }
    }
  }

  return kernel;
}

kernel_info_t *gpgpu_sim::select_kernel(unsigned core_id,
                  std::vector<kernel_info_t *> &m_running_kernels_stream,
                  unsigned &m_last_issued_kernel_stream) {
  if (m_running_kernels_stream[m_last_issued_kernel_stream] && 
      !m_running_kernels_stream[m_last_issued_kernel_stream]->no_more_ctas_to_run() &&
      !m_running_kernels_stream[m_last_issued_kernel_stream]->m_kernel_TB_latency &&
      m_running_kernels_stream[m_last_issued_kernel_stream]->is_in_core_range(core_id)) {
    
    // Record statistics for first time execution
    unsigned launch_uid = m_running_kernels_stream[m_last_issued_kernel_stream]->get_uid();

    bool findKernel = std::find(m_executed_kernel_uids.begin(), 
                        m_executed_kernel_uids.end(), launch_uid) != m_executed_kernel_uids.end();
    if (findKernel) {
      m_running_kernels_stream[m_last_issued_kernel_stream]->start_cycle = 
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels_stream[m_last_issued_kernel_stream]->name());

     printf("select_kernel core_id %u, kernelId %u kernel %s\n",
     core_id, m_last_issued_kernel_stream, m_running_kernels_stream[m_last_issued_kernel_stream]->name().c_str());
    }

    return m_running_kernels_stream[m_last_issued_kernel_stream];
  }

  for (unsigned n = 0; n < m_running_kernels_stream.size(); n++) {

    unsigned idx = (n + m_last_issued_kernel_stream + 1) % m_running_kernels_stream.size();

    if (kernel_more_cta_left(m_running_kernels_stream[idx]) &&
        !m_running_kernels_stream[idx]->m_kernel_TB_latency) {
      if (!m_running_kernels_stream[idx]->is_in_core_range(core_id)) {
        // If the kernel is not in the core range, skip it
        printf("select_kernel skip core_id %u, stream %u kernel %s core range %s\n",
               core_id, idx, m_running_kernels_stream[idx]->name().c_str(),
               m_running_kernels_stream[idx]->print_core_range().c_str());
        continue;
      }

      m_last_issued_kernel_stream = idx;

      m_running_kernels_stream[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;

      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels_stream[idx]->get_uid();
      assert(std::find(m_executed_kernel_uids.begin(),
                       m_executed_kernel_uids.end(),
                       launch_uid) == m_executed_kernel_uids.end());
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(m_running_kernels_stream[idx]->name());

      //printf("select_kernel switch core_id %u, kernelId %u kernel %s\n",
      //core_id, m_last_issued_kernel_stream, m_running_kernels_stream[idx]->name().c_str());

      return m_running_kernels_stream[idx];
    }
  }

  return NULL;
}


unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty()) {
    last_streamID = -1;
    return 0;
  }
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  unsigned uid = kernel->get_uid();
  last_uid = uid;
  unsigned long long streamID = kernel->get_streamID();
  last_streamID = streamID;
  gpu_kernel_time.at(streamID).at(uid).end_cycle =
      gpu_tot_sim_cycle + gpu_sim_cycle;
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;

      if (streamID == 0) {
        // Remove from stream1
        auto it = std::find(m_running_kernels_stream1.begin(),
                            m_running_kernels_stream1.end(), kernel);
        if (it != m_running_kernels_stream1.end()) {
          *it = NULL;        
        }
      } else {
        // Remove from stream2
        auto it = std::find(m_running_kernels_stream2.begin(),
                            m_running_kernels_stream2.end(), kernel);
        if (it != m_running_kernels_stream2.end()) {
          *it = NULL;        
        }
      }
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

void exec_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats);
}

// SST get its own simt_cluster
void sst_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new sst_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                  m_shader_stats, m_memory_stats);
  SST_gpgpu_reply_buffer.resize(m_shader_config->n_simt_clusters);
}

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
  m_gpgpusim_wrapper = new gpgpu_sim_wrapper(
      config.g_power_simulation_enabled, config.g_power_config_name,
      config.g_power_simulation_mode, config.g_dvfs_enabled);
#endif

  m_shader_stats = new shader_core_stats(m_shader_config);
  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);

  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_tot_issued_cta = 0;
  gpu_completed_cta = 0;
  m_total_cta_launched = 0;
  gpu_deadlock = false;

  gpu_stall_dramfull = 0;
  gpu_stall_icnt2sh = 0;
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;
  last_streamID = -1;

  gpu_kernel_time.clear();

  collect_info_core_bitset.reset();

  // TODO: somehow move this logic to the sst_gpgpu_sim constructor?
  if (!m_config.is_SST_mode()) {
    // Init memory if not in SST mode
    m_memory_partition_unit =
        new memory_partition_unit *[m_memory_config->m_n_mem];
    m_memory_sub_partition =
        new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      m_memory_partition_unit[i] =
          new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
      for (unsigned p = 0;
           p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
        unsigned submpid =
            i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
        m_memory_sub_partition[submpid] =
            m_memory_partition_unit[i]->get_sub_partition(p);
      }
    }

    icnt_wrapper_init();
    icnt_create(m_shader_config->n_simt_clusters,
                m_memory_config->m_n_mem_sub_partition);
  }
  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0

  m_last_issued_kernel_stream1 = 0;
  m_last_issued_kernel_stream2 = 0;
  
  // Initialize stream-based allocation variables

  if (m_config.get_dynamic_core_scheduling() ||
      m_config.get_dyno_core_scheduling()) {
    m_last_cluster_issue_stream1 = m_shader_config->n_simt_clusters;
    m_last_cluster_issue_stream2 = m_shader_config->n_simt_clusters;
  } else if (m_config.get_stream_intlv_core()){
    m_last_cluster_issue_stream1 = m_shader_config->n_simt_clusters-1;
    m_last_cluster_issue_stream2 = m_shader_config->n_simt_clusters;
  } else {
    m_last_cluster_issue_stream1 = m_shader_config->n_simt_clusters/2;
    m_last_cluster_issue_stream2 = m_shader_config->n_simt_clusters;
  }
  
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;
}

unsigned gpgpu_sim::get_next_core_id_by_core_range(unsigned stream_id) {

  auto& core_range = (m_config.get_dynamic_core_scheduling() || m_config.get_dyno_core_scheduling()) ?
                    global_dynamic_core_ranges_vector[stream_id] :
                    global_stream_core_ranges_vector[stream_id];

  unsigned last_stream_id = stream_id == 0 ? m_last_cluster_issue_stream1 : m_last_cluster_issue_stream2;

  if (last_stream_id >= core_range.back()) {
    return core_range.front();
  }

  for (auto core_id : core_range) {
    if (core_id > last_stream_id) {
      return core_id;
    }
  }

  return core_range.front();
}


void sst_gpgpu_sim::SST_receive_mem_reply(unsigned core_id, void *mem_req) {
  assert(core_id < m_shader_config->n_simt_clusters);
  mem_fetch *mf = (mem_fetch *)mem_req;

  (SST_gpgpu_reply_buffer[core_id]).push_back(mf);
}

mem_fetch *sst_gpgpu_sim::SST_pop_mem_reply(unsigned core_id) {
  if (SST_gpgpu_reply_buffer[core_id].size() > 0) {
    mem_fetch *temp = SST_gpgpu_reply_buffer[core_id].front();
    SST_gpgpu_reply_buffer[core_id].pop_front();
    return temp;
  } else
    return NULL;
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

shader_core_ctx *gpgpu_sim::get_core_by_sid(unsigned sid) const {
  unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
  unsigned core_id = m_shader_config->sid_to_cid(sid);
  return m_cluster[cluster_id]->get_core(core_id);
}

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

void gpgpu_sim::release_core_range_limit() {
  // This function is used to release the core range limit for all running kernels.
  // It is used when one stream is done and the next stream could run with all cores.
  for (unsigned i = 0; i < m_running_kernels.size(); i++) {
    if (m_running_kernels[i] != NULL)
      m_running_kernels[i]->set_core_range(0, get_config().num_shader()-1);
  }
}

void gpgpu_sim::info_transition_done() {
  // set all the cores with transition done
  for (unsigned core_id = 0; core_id < m_config.num_shader(); core_id++) {
    shader_core_ctx *core = get_core_by_sid(core_id);
    if (core) {
      core->transition_done = true;

      //printf("GPGPU-Sim uArch: Shader %d transition done, core %d\n",
      //core_id, core->get_sid());
      // invalidate all l1 cache
      //core->cache_flush();
      //core->cache_invalidate();
      //printf("GPGPU-Sim uArch: Shader %d transition done, cache flushed\n", core_id);
    }
  }

  /*
  // flush all the l2 cache
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {

    for (unsigned p = 0;
      p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
      unsigned submpid =
          i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
      unsigned flushed_data = m_memory_sub_partition[submpid]->flushL2();
      printf("GPGPU-Sim uArch: L2 cache flushed, flushed_data: %d\n", flushed_data);
      unsigned invalidated_data = m_memory_sub_partition[submpid]->invalidateL2();
      printf("GPGPU-Sim uArch: L2 cache invalidated, invalidated_data: %d\n", invalidated_data);
    }
  }
  */
}

void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
}

bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  if (icnt_busy()) return true;
  if (get_more_cta_left()) return true;
  return false;
}

bool sst_gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  if (get_more_cta_left()) return true;
  return false;
}

void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

// McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
               gpu_tot_sim_insn, gpu_sim_insn);
  }
#endif

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode) icnt_init();
}

void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

PowerscalingCoefficients *gpgpu_sim::get_scaling_coeffs() {
  return m_gpgpusim_wrapper->get_scaling_coeffs();
}

void gpgpu_sim::print_stats(unsigned long long streamID) {
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat(streamID);

  if (g_network_mode) {
    printf(
        "----------------------------Interconnect-DETAILS----------------------"
        "----------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf(
        "----------------------------END-of-Interconnect-DETAILS---------------"
        "----------\n");
  }
}

void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    fflush(stdout);
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
        gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
        (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
        (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
    printf(
          "\n\nGPGPU-Sim uArch: sim_cycle: %lld, tot_sim_cycle: %lld\n",
          gpu_sim_cycle, gpu_tot_sim_cycle);
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
      printf("\n");
      m_cluster[i]->print_not_completed_detail(stdout);
      printf("\n");
      // print out l1d cache
      for (unsigned i = 0; i < m_config.num_cluster(); i++) {
        m_cluster[i]->print_core_detail(stdout);
      }
      printf("\n");
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    if (icnt_busy()) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}

std::string gpgpu_sim::executed_kernel_name() {
  std::stringstream statout;
  if (m_executed_kernel_names.size() == 1)
    statout << m_executed_kernel_names[0];
  else {
    for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " ";
    }
  }
  return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}

//fprintf(stdout, "core %d ipc = %.4lf, kernel %p name %s, cta %u, cycle %llu, tot_cycle %llu\n",
//m_sid, m_stats->get_ipc(m_sid), kernel, kernel->name().c_str(), cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle);

void gpgpu_sim::profile_kernel_stats(unsigned m_sid, double ipc, kernel_info_t *kernel){
    if (!m_config.get_dynamic_core_scheduling() && !m_config.get_dyno_core_scheduling())
        return;

    unsigned stream_id = core_stream_mapping[m_sid];

    // if dynamic scheduling is configured, we don't need to profile the kernel stats
    //if (dynamic_scheduling_configured) {
    //  return;
    //}
    if (!dynamic_scheduling_configured) {
      // if the core id is not in the map, initialize it to 0
      if (collect_info_core_inst_count.find(m_sid%12) == collect_info_core_inst_count.end()) {
        collect_info_core_inst_count[m_sid%12] = 0;
      } else {
        collect_info_core_inst_count[m_sid%12]++;
      }
    } else {

      if (collect_info_stream_inst_count.find(stream_id) == collect_info_stream_inst_count.end()) {
        collect_info_stream_inst_count[stream_id] = 0;
      } else {
        collect_info_stream_inst_count[stream_id]++;
      }
    }


    // 500 threshold for core profiling
    if (!dynamic_scheduling_configured && collect_info_core_inst_count[m_sid%12] > 200) {
      collect_info_core_bitset.set(m_sid%12);
    } else if (dynamic_scheduling_configured && collect_info_stream_inst_count[stream_id] > 20000) {
      // 500000 threshold for stream running
      collect_stream_bitset.set(stream_id);
    }
  
    if (collect_info_core_bitset.all()) {
      printf("All cores have collected info\n");
    } else if (collect_stream_bitset.all()) {
      printf("All streams have collected info\n");
    } else {
      // print out the core id and inst count and also collect_info_core_bitset
      //for (unsigned core_id = 0; core_id < 12; core_id++) {
      //  printf("Core %u has collected %u insts\n",
      //  core_id, collect_info_core_inst_count[core_id]);
      //}
      //printf("Collect info core bitset: %s\n", collect_info_core_bitset.to_string().c_str());
      return;
    }


    if (dynamic_scheduling_configured) {
      sampling_dynamic_core_scheduling();
      dynamic_scheduling_configured = false;
    } else {
      dynamic_core_scheduling();
    }

    collect_info_core_inst_count.clear();
    collect_info_core_bitset.reset();
    collect_stream_bitset.reset();
}

void gpgpu_sim::sampling_dyno_core_scheduling() {
  unsigned total_cores = m_config.num_shader();
  unsigned num_streams = global_unique_streams.size();
  // consider SM pairs:
  // SM0/SM1 will be shared by two streams, sharing L1 cache
  // SM2/SM3 will be shared by two streams, SM2 bypasses L1 cache
  // SM4/SM5 will be shared by two streams, SM5 bypasses L1 cache
  // SM6/SM7 will be shared by two streams, both bypasses L1 cache
  // SM8/SM9 will be used by stream 0 exclusively, sharing L1 cache
  // SM10/SM11 will be used by stream 1 exclusively, sharing L1 cache
  // stream 0 will be using: 0,2,4,6, 8, 9, 12,14,16,18,20,21
  // stream 1 will be using: 1,3,5,7,10,11, 13,15,17,19,22,23

  std::vector<unsigned long long> stream_list(global_unique_streams.begin(), global_unique_streams.end());
  std::sort(stream_list.begin(), stream_list.end());

  for (unsigned i = 0; i < num_streams; i++) {
    unsigned long long stream_id = stream_list[i];
    unsigned start_core = stream_id == 0 ? 0 : 1;
    unsigned end_core = total_cores - 1;

    std::set<unsigned> core_range;
    std::vector<unsigned> core_range_vector;

    // during the sampling phase, we only need to consider the first 12 cores
    unsigned core = start_core;
    core_range.insert(core);
    if (stream_id == 0) {
      if (core + 2 <= end_core)
        core_range.insert(core + 2);
      if (core + 4 <= end_core)
          core_range.insert(core + 4);
      if (core + 6 <= end_core)
        core_range.insert(core + 6);
      if (core + 8 <= end_core)
        core_range.insert(core + 8);
      if (core + 9 <= end_core)
        core_range.insert(core + 9);
    } else {
      if (core + 2 <= end_core)
        core_range.insert(core + 2);
      if (core + 4 <= end_core)
        core_range.insert(core + 4);
      if (core + 6 <= end_core)
        core_range.insert(core + 6);
      if (core + 9 <= end_core)
        core_range.insert(core + 9);
      if (core + 10 <= end_core)
        core_range.insert(core + 10);
    }      
  
    for (auto core : core_range) {
        core_stream_mapping[core] = stream_id;
    }

    global_dynamic_core_ranges[stream_id] = core_range;

    core_range_vector = std::vector<unsigned>(core_range.begin(), core_range.end());
    // sort the core_range_vector in ascending order
    std::sort(core_range_vector.begin(), core_range_vector.end());
    global_dynamic_core_ranges_vector[stream_id] = core_range_vector;
    std::cout << "GLOBAL: STREAM " << stream_id << " -> cores [" << start_core << "-" << end_core << "]" << std::endl;
    // print out the core_range_vector
    std::cout << "GLOBAL: STREAM VECTOR " << stream_id << " -> cores: ";
    for (auto core : core_range_vector) {
      std::cout << core << " ";
    }
    std::cout << std::endl;
  }

  unsigned core_id = 0;
  unsigned sm2_id = core_id + 2;
  unsigned sm5_id = core_id + 5;
  unsigned sm6_id = core_id + 6;
  unsigned sm7_id = core_id + 7;
  shader_core_ctx *core = get_core_by_sid(sm2_id);
  core->set_stream_bypassL1D(0,true);
  core = get_core_by_sid(sm5_id);
  core->set_stream_bypassL1D(1,true);
  core = get_core_by_sid(sm6_id);
  core->set_stream_bypassL1D(0,true);
  core = get_core_by_sid(sm7_id);
  core->set_stream_bypassL1D(1,true); 
}

void gpgpu_sim::sampling_dynamic_core_scheduling() {
  printf("Sampling dynamic core scheduling cycle %lld\n", gpu_sim_cycle+gpu_tot_sim_cycle);
  global_dynamic_core_ranges.clear();
  global_dynamic_core_ranges_vector.clear();
  core_stream_mapping.clear();
  unsigned total_cores = m_config.num_shader();
  unsigned num_streams = global_unique_streams.size();

  // create a map to store the core range for each stream
  std::map<unsigned long long, std::set<unsigned>> core_range_map;
  std::map<unsigned long long, std::vector<unsigned>> core_range_vector_map;

  for (unsigned core_id = 0; core_id < total_cores; core_id++) {
    shader_core_ctx *core = get_core_by_sid(core_id);
    core->copy_stream_bypassL1D_new_to_old();
  }

  // create a map to store the core stream mapping
  
  // consider SM pairs:
  // SM0/SM1 will be shared by two streams, sharing L1 cache
  // SM2/SM3 will be shared by two streams, SM2 bypasses L1 cache
  // SM4/SM5 will be shared by two streams, SM5 bypasses L1 cache
  // SM6/SM7 will be shared by two streams, both bypasses L1 cache
  // SM8/SM9 will be used by stream 0 exclusively, sharing L1 cache
  // SM10/SM11 will be used by stream 1 exclusively, sharing L1 cache
  // stream 0 will be using: 0,2,4,6, 8, 9, 12,14,16,18,20,21
  // stream 1 will be using: 1,3,5,7,10,11, 13,15,17,19,22,23

  std::vector<unsigned long long> stream_list(global_unique_streams.begin(), global_unique_streams.end());
  std::sort(stream_list.begin(), stream_list.end());

  for (unsigned i = 0; i < num_streams; i++) {
    unsigned long long stream_id = stream_list[i];
    unsigned start_core = stream_id == 0 ? 0 : 1;
    unsigned end_core = total_cores - 1;

    std::set<unsigned> core_range;
    std::vector<unsigned> core_range_vector;
    for (unsigned core = start_core; core <= end_core; core+=12) {
      core_range.insert(core);
      if (stream_id == 0) {
          if (core + 2 <= end_core)
            core_range.insert(core + 2);
          if (core + 4 <= end_core)
            core_range.insert(core + 4);
          if (core + 6 <= end_core)
            core_range.insert(core + 6);
          if (core + 8 <= end_core)
            core_range.insert(core + 8);
          if (core + 9 <= end_core)
            core_range.insert(core + 9);
      } else {
        if (core + 2 <= end_core)
          core_range.insert(core + 2);
        if (core + 4 <= end_core)
          core_range.insert(core + 4);
        if (core + 6 <= end_core)
          core_range.insert(core + 6);
        if (core + 9 <= end_core)
          core_range.insert(core + 9);
        if (core + 10 <= end_core)
          core_range.insert(core + 10);
      }
    }

    for (auto core : core_range) {
      core_stream_mapping[core] = stream_id;
    }

    global_dynamic_core_ranges[stream_id] = core_range;

    // update core range for m_running_kernels_stream
    if (stream_id == 0) {
      for (auto& kernel : m_running_kernels_stream1) {
        if (kernel!=nullptr) {
          kernel->set_core_range(core_range);
        }
      }
    } else {
      for (auto& kernel : m_running_kernels_stream2) {
        if (kernel!=nullptr) {
          kernel->set_core_range(core_range);
        }
      }
    }

    // also update the m_running_kernels

    for (auto& kernel : m_running_kernels) {
      if (kernel!=nullptr && kernel->get_streamID() == stream_id) {
        kernel->set_core_range(core_range);
      }
    }

    core_range_vector = std::vector<unsigned>(core_range.begin(), core_range.end());
    // sort the core_range_vector in ascending order
    std::sort(core_range_vector.begin(), core_range_vector.end());
    global_dynamic_core_ranges_vector[stream_id] = core_range_vector;
    std::cout << "GLOBAL: STREAM " << stream_id << " -> cores [" << start_core << "-" << end_core << "]" << std::endl;
    // print out the core_range_vector
    std::cout << "GLOBAL: STREAM VECTOR " << stream_id << " -> cores: ";
    for (auto core : core_range_vector) {
      std::cout << core << " ";
    }
    std::cout << std::endl;
  }
  for (unsigned core_id = 0; core_id < total_cores; core_id+=12) {
    unsigned sm2_id = core_id + 2;
    unsigned sm5_id = core_id + 5;
    unsigned sm6_id = core_id + 6;
    unsigned sm7_id = core_id + 7;
    shader_core_ctx *core = get_core_by_sid(sm2_id);
    core->set_stream_bypassL1D_new(0,true);
    core = get_core_by_sid(sm5_id);
    core->set_stream_bypassL1D_new(1,true);
    core = get_core_by_sid(sm6_id);
    core->set_stream_bypassL1D_new(0,true);
    core = get_core_by_sid(sm7_id);
    core->set_stream_bypassL1D_new(1,true);
  }

  for (unsigned core_id = 0; core_id < total_cores; core_id++) {
    shader_core_ctx *core = get_core_by_sid(core_id);
    core->start_transition = true;
    core->transition_count = 0;
    core->get_stats()->reset_m_num_sim_insn_profile(core_id, gpu_sim_cycle+gpu_tot_sim_cycle);
  }

  is_policy_change = true;
  is_policy_change_done = false;
}

void gpgpu_sim::dynamic_core_scheduling() {
  printf("Dynamic core scheduling cycle %lld\n", gpu_sim_cycle+gpu_tot_sim_cycle);
  dynamic_scheduling_configured = true;

  // get core and print out the ipc from stats and inst count
  for (unsigned core_id = 0; core_id < m_config.num_shader(); core_id++) {
    shader_core_ctx *core = get_core_by_sid(core_id);
    if (core) {
      core->print_core_stats();
    }
  }

  for (unsigned core_id = 0; core_id < m_config.num_shader(); core_id++) {
    shader_core_ctx *core = get_core_by_sid(core_id);
    core->copy_stream_bypassL1D_new_to_old();
  }

  unsigned end_core_id = m_config.get_dyno_core_scheduling() ? 12 : m_config.num_shader();

  for (unsigned SM_pair_id = 0; SM_pair_id < end_core_id/2; SM_pair_id++) {
    unsigned even_core_id = SM_pair_id*2;
    unsigned odd_core_id = SM_pair_id*2 + 1;
    shader_core_ctx *even_core = get_core_by_sid(even_core_id);
    shader_core_ctx *odd_core = get_core_by_sid(odd_core_id);

    //double even_ipc = even_core->get_stats()->get_ipc(even_core_id);  
    //double odd_ipc = odd_core->get_stats()->get_ipc(odd_core_id);
    double even_ipc = even_core->get_stats()->get_ipc_profile(even_core_id);  
    double odd_ipc = odd_core->get_stats()->get_ipc_profile(odd_core_id);
    long long even_inst_count = even_core->get_stats()->m_num_sim_insn[even_core_id];
    long long odd_inst_count = odd_core->get_stats()->m_num_sim_insn[odd_core_id];

    if (even_ipc > 0 && odd_ipc > 0) {
      printf("SM pair %u even core ipc: %.4f, odd core ipc: %.4f, even inst count: %lld, odd inst count: %lld\n",
        SM_pair_id, even_ipc, odd_ipc, even_inst_count, odd_inst_count);
    } else {
      continue;
    }

    std::string policy_name = determine_policy_for_core(even_core_id);

    PolicyStats core_stats;
    core_stats.even_ipc = even_ipc;
    core_stats.odd_ipc = odd_ipc;
    core_stats.even_inst_count = even_inst_count;
    core_stats.odd_inst_count = odd_inst_count;

    assert(policy_name != "");

    policy_performance_stats_vector[policy_name].push_back(core_stats);
  }

  // calculate the average of each element in the vector and create policy stats for each policy
  for (auto& pair : policy_performance_stats_vector) {
    std::string policy_name = pair.first;
    std::vector<PolicyStats>& stats_vector = pair.second;
    PolicyStats avg_stats;
    avg_stats.policy_name = policy_name;
    for (auto& stats : stats_vector) {
      avg_stats.even_ipc += stats.even_ipc;
      avg_stats.odd_ipc += stats.odd_ipc;
      avg_stats.even_inst_count += stats.even_inst_count;
      avg_stats.odd_inst_count += stats.odd_inst_count;
    }
    avg_stats.even_ipc /= stats_vector.size();
    avg_stats.odd_ipc /= stats_vector.size();
    avg_stats.even_inst_count /= stats_vector.size();
    avg_stats.odd_inst_count /= stats_vector.size();
    policy_performance_stats[policy_name] = avg_stats;
  }

  // merge exclusive_stream0 and exclusive_stream1 into exclusive
  PolicyStats exclusive_stats;
  exclusive_stats.policy_name = "exclusive";
  exclusive_stats.even_ipc = (policy_performance_stats["exclusive_stream0"].even_ipc + policy_performance_stats["exclusive_stream0"].odd_ipc)/2;
  exclusive_stats.even_cta_count = (policy_performance_stats["exclusive_stream0"].even_cta_count + policy_performance_stats["exclusive_stream0"].odd_cta_count)/2;
  exclusive_stats.even_inst_count = (policy_performance_stats["exclusive_stream0"].even_inst_count + policy_performance_stats["exclusive_stream0"].odd_inst_count)/2;

  exclusive_stats.odd_ipc = (policy_performance_stats["exclusive_stream1"].odd_ipc + policy_performance_stats["exclusive_stream1"].even_ipc)/2;
  exclusive_stats.odd_cta_count = (policy_performance_stats["exclusive_stream1"].odd_cta_count + policy_performance_stats["exclusive_stream1"].even_cta_count)/2;
  exclusive_stats.odd_inst_count = (policy_performance_stats["exclusive_stream1"].odd_inst_count + policy_performance_stats["exclusive_stream1"].even_inst_count)/2;
  policy_performance_stats["exclusive"] = exclusive_stats;

  print_policy_comparison();

  // choose the best policy based on performance metrics
  std::string best_policy = "";
  double best_score = 0.0;

  // Calculate baseline IPC values for fairness comparison
  double exclusive_stream0_ipc = (policy_performance_stats["exclusive_stream0"].even_ipc + policy_performance_stats["exclusive_stream0"].odd_ipc) / 2.0;
  double exclusive_stream1_ipc = (policy_performance_stats["exclusive_stream1"].even_ipc + policy_performance_stats["exclusive_stream1"].odd_ipc) / 2.0;
  
  // remove exclusive_stream0 and exclusive_stream1
  policy_performance_stats.erase("exclusive_stream0");
  policy_performance_stats.erase("exclusive_stream1");

  for (auto& pair : policy_performance_stats) {
    const PolicyStats& stats = pair.second;
    double avg_ipc = stats.get_average_ipc();
    
    // Calculate IPC improvement compared to baseline exclusive policy
    double ipc_improvement = 0.0;
    if (stats.policy_name != "exclusive") {
      // Calculate baseline IPC (average of exclusive_stream0 and exclusive_stream1)
      double baseline_ipc = (exclusive_stream0_ipc + exclusive_stream1_ipc) / 2.0;
      ipc_improvement = (avg_ipc - baseline_ipc) / baseline_ipc; // Percentage improvement
    }
    
    double fairness_score = stats.even_inst_count+stats.odd_inst_count;

    if (fairness_score > best_score) {
      best_score = fairness_score;
      best_policy = stats.policy_name;
    }

    printf("Policy %s fairness score: %.4f, even ipc/inst_count: %.4f/%lld, odd ipc/inst_count: %.4f/%lld\n", 
           stats.policy_name.c_str(), fairness_score, stats.even_ipc, stats.even_inst_count, stats.odd_ipc, stats.odd_inst_count);
  }
  
  // Apply the best policy
  if (!best_policy.empty()) {
    printf("DYNAMIC SCHEDULING: Selected policy '%s' with score %.4f cycle %lld total cycle %lld\n", 
           best_policy.c_str(), best_score, gpu_core_abs_cycle, gpu_tot_sim_cycle+gpu_sim_cycle);
    
    // Update core allocations based on the best policy
    update_core_allocation_for_policy(best_policy);

    m_last_cluster_issue_stream1 = m_shader_config->n_simt_clusters;
    m_last_cluster_issue_stream2 = m_shader_config->n_simt_clusters;
  }
}

void gpgpu_sim::update_policy_stats(const std::string& policy, PolicyStats& core_stats) {
  // add stats to policy_performance_stats
  auto& stats = policy_performance_stats[policy];
  stats.even_ipc = (stats.even_ipc * stats.even_cta_count + core_stats.even_ipc*core_stats.even_cta_count)/
                   (stats.even_cta_count + core_stats.even_cta_count);
  stats.even_cta_count+=core_stats.even_cta_count;
  stats.even_inst_count+=core_stats.even_inst_count;
  stats.odd_ipc = (stats.odd_ipc * stats.odd_cta_count + core_stats.odd_ipc*core_stats.odd_cta_count)/
                (stats.odd_cta_count + core_stats.odd_cta_count);
  stats.odd_cta_count+=core_stats.odd_cta_count;
  stats.odd_inst_count+=core_stats.odd_inst_count;
  stats.total_ipc += core_stats.total_ipc*core_stats.cta_count;
  stats.cta_count+=core_stats.cta_count;
  stats.policy_count++;
}


void gpgpu_sim::print_policy_comparison() {
  // Calculate baseline IPC values for fairness comparison
  double exclusive_stream0_ipc = (policy_performance_stats["exclusive_stream0"].even_ipc + policy_performance_stats["exclusive_stream0"].odd_ipc) / 2.0;
  double exclusive_stream1_ipc = (policy_performance_stats["exclusive_stream1"].even_ipc + policy_performance_stats["exclusive_stream1"].odd_ipc) / 2.0;

  long long exclusive_stream0_inst_count = (policy_performance_stats["exclusive_stream0"].even_inst_count + policy_performance_stats["exclusive_stream0"].odd_inst_count) / 2;
  long long exclusive_stream1_inst_count = (policy_performance_stats["exclusive_stream1"].even_inst_count + policy_performance_stats["exclusive_stream1"].odd_inst_count) / 2;
  
  // print out odd, even ipc and total ipc
  printf("\n=== IPC ===\n");
  printf("%-20s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n", "Policy", "Even IPC", "Even Inst", "Odd IPC", "Odd Inst", "Avg IPC", "IPC Improvement", "Fairness Score", "Weighted Score");
  printf("--------------------------------------------------------------------------------\n");
  for (const auto& pair : policy_performance_stats) {
      const PolicyStats& stats = pair.second;
      double avg_ipc = stats.get_average_ipc();
      
      // Calculate IPC improvement compared to baseline exclusive policy
      double ipc_improvement = 0.0;
      if (stats.policy_name != "exclusive") {
          ipc_improvement = ((stats.even_ipc/exclusive_stream0_ipc)*exclusive_stream0_inst_count
                          + (stats.odd_ipc/exclusive_stream1_ipc)*exclusive_stream1_inst_count
                        - 2.0 * (exclusive_stream0_inst_count + exclusive_stream1_inst_count)) /
                        (2.0 * (exclusive_stream0_inst_count + exclusive_stream1_inst_count)) * 100.0; // Percentage improvement
      }
      
      // Create fairness-aware metric
      double fairness_factor = 2;
      //double fairness_score = avg_ipc + fairness_factor * ipc_improvement * avg_ipc / 100.0;
      double fairness_score = stats.even_inst_count + stats.odd_inst_count;
      
      double weighted_score = fairness_score;
      if (stats.policy_name == "exclusive") {
          weighted_score = fairness_score * 1;
      } else if (stats.policy_name == "bypass_bypass") {
          weighted_score = fairness_score * 1;
      } else if (stats.policy_name == "bypass_share" || stats.policy_name == "share_bypass") {
          weighted_score = fairness_score * 1;
      }
      printf("%-20s %-12.4f %-12d %-12.4f %-12d %-12.4f %-12.2f %-12.4f %-12.4f\n",
             stats.policy_name.c_str(),
             stats.even_ipc,
             stats.even_inst_count,
             stats.odd_ipc,
             stats.odd_inst_count,
             avg_ipc,
             ipc_improvement,
             fairness_score,
             weighted_score);
  }

  // print out the best policy
  printf("\n=== BEST POLICY ===\n");
  std::string best_policy = "";
  double best_weighted_score = 0.0;
  
  for (const auto& pair : policy_performance_stats) {
      const PolicyStats& stats = pair.second;
      double avg_ipc = stats.get_average_ipc();
      
      // Calculate IPC improvement compared to baseline exclusive policy
      double ipc_improvement = 0.0;
      if (stats.policy_name != "exclusive") {
        ipc_improvement = ((stats.even_ipc/exclusive_stream0_ipc)*exclusive_stream0_inst_count
                        + (stats.odd_ipc/exclusive_stream1_ipc)*exclusive_stream1_inst_count
                      - 2.0 * (exclusive_stream0_inst_count + exclusive_stream1_inst_count)) /
                      (2.0 * (exclusive_stream0_inst_count + exclusive_stream1_inst_count)) * 100.0; // Percentage improvement
      }
      
      if (stats.policy_name == "exclusive_stream0" || stats.policy_name == "exclusive_stream1") {
        continue;
      }

      // Create fairness-aware metric: IPC + fairness_factor * IPC_improvement
      double fairness_factor = 2; // Weight for improvement consideration
      //double fairness_score = avg_ipc + fairness_factor * ipc_improvement * avg_ipc;
      double fairness_score = stats.even_inst_count + stats.odd_inst_count;
      
      // Apply policy-specific weights to the fairness score
      double weighted_score = fairness_score;
      if (stats.policy_name == "exclusive") {
          weighted_score = fairness_score * 1;  // 20% boost for exclusive policy
      } else if (stats.policy_name == "bypass_bypass") {
          weighted_score = fairness_score * 1;  // 10% boost for bypass_bypass policy
      } else if (stats.policy_name == "bypass_share" || stats.policy_name == "share_bypass") {
          weighted_score = fairness_score * 1; // 5% boost for bypass_share and share_bypass policies
      }
      
      if (weighted_score > best_weighted_score) {
          best_weighted_score = weighted_score;
          best_policy = stats.policy_name;
      }
  }
  
  printf("\n=== WEIGHTED FAIRNESS POLICY RECOMMENDATION ===\n");
  printf("Best weighted policy: %s (Weighted Fairness Score: %.4f)\n", 
         best_policy.c_str(), best_weighted_score);
  
  // Provide recommendations based on results
  printf("\n=== POLICY RECOMMENDATIONS ===\n");
  if (best_policy == "shared_l1_cache") {
      printf("Recommendation: Use shared L1 cache policy for better performance\n");
  } else if (best_policy == "share_bypass" || best_policy == "bypass_share") {
      printf("Recommendation: Use shared with bypass policy for better performance\n");
  } else if (best_policy == "bypass_bypass") {
      printf("Recommendation: Use bypass bypass policy for better performance\n");
  } else if (best_policy == "exclusive") {
      printf("Recommendation: Use exclusive stream allocation for better performance\n");
  } else {
      printf("Recommendation: Default allocation performs best\n");
  }
}

void gpgpu_sim::update_core_allocation_for_policy(const std::string& policy,
  bool bypass_bypass_stream0_better,
  bool bypass_bypass_stream1_better) {


  // Update core allocations based on the selected policy
  if (policy == "shared_l1_cache") {
    printf("Applying shared L1 cache policy\n");
    // Core allocation remains as is - both cores share L1 cache
    dynamic_scheduling_set_shared_cores();
  } else if (policy == "share_bypass") {
    // Shared with bypass policy: One core bypasses L1, other uses full cache
    printf("Applying shared with bypass policy\n");
    dynamic_scheduling_set_shared_cores(false, true);
  } else if (policy == "bypass_share") {
    printf("Applying bypass share policy\n");
    dynamic_scheduling_set_shared_cores(true, false);
  } else if (policy == "bypass_bypass") {
    printf("Applying bypass bypass policy\n");
    dynamic_scheduling_set_shared_cores(true, true);
  } else if (policy == "exclusive") {
    // Exclusive stream policy: Dedicated cores for each stream
    printf("Applying exclusive stream policy: %s\n", policy.c_str());
    dynamic_scheduling_exclusive(bypass_bypass_stream0_better, bypass_bypass_stream1_better);
  } else {
    assert (false);
    // Default policy: No special allocation
    printf("Applying default allocation policy\n");
  }

  // first reset all core to not bypass L1
    for (unsigned core_id = 0; core_id < m_config.num_shader(); core_id++) {
      shader_core_ctx *core = get_core_by_sid(core_id);
      if (core) {
        //core->set_bypassL1D(false);
        //core->set_stream_bypassL1D(0,false);
        //core->set_stream_bypassL1D(1,false);
        core->start_transition = true;
        core->transition_count = 0;
        // SILLY DEBUG
        //core->transition_done = true;
      }
    }

    is_policy_change = true;
    is_policy_change_done = false;
}

std::string gpgpu_sim::determine_policy_for_core(unsigned core_id) {
    // Determine policy based on core ID and current allocation
    // This maps to the allocation strategy you described
    
    // Check if this core is in the dynamic allocation range
    if (core_stream_mapping.find(core_id) != core_stream_mapping.end()) {        
        // Determine policy based on core position
        if (core_id % 12 == 0 || core_id % 12 == 1) {
            return "shared_l1_cache";  // SM0/SM1 - shared L1 cache
        } else if (core_id % 12 == 2 || core_id % 12 == 3) {
            return "bypass_share";  // SM2/SM3 - bypass with share
        } else if (core_id % 12 == 4 || core_id % 12 == 5) {
            return "share_bypass";  // SM4/SM5 - share with bypass
        } else if (core_id % 12 == 6 || core_id % 12 == 7) {
          return "bypass_bypass";   // SM6/SM7 - bypass bypass
        } else if (core_id % 12 == 8 || core_id % 12 == 9) {
            return "exclusive_stream0";
        } else if (core_id % 12 == 10 || core_id % 12 == 11) {
            return "exclusive_stream1";
        }
    }
    
    // Default policy for cores not in dynamic allocation
    return "default_allocation";
}


void gpgpu_sim::dynamic_scheduling_set_shared_cores(
  bool is_stream0_bypass,  bool is_stream1_bypass) {
  // clear the global_stream_core_ranges_set
  global_dynamic_core_ranges.clear();
  global_dynamic_core_ranges_vector.clear();
  // Create sorted stream list for consistent core assignment
  std::vector<unsigned long long> stream_list(global_unique_streams.begin(), global_unique_streams.end());
  std::sort(stream_list.begin(), stream_list.end());

  unsigned total_cores = get_config().num_shader();

  core_stream_mapping.clear();

  // in the dyno core scheduling, the first 12 core are always in sampling phase
  // so we need to start from the 13th core
  unsigned start_core_id = m_config.get_dyno_core_scheduling() ? 12 : 0;

  for (unsigned i = 0; i < stream_list.size(); i++) {
    std::set<unsigned> core_range;
    std::vector<unsigned> core_range_vector;
    unsigned long long stream_id = stream_list[i];
    unsigned start_core = stream_id == 0 ? start_core_id :start_core_id+1;
    unsigned end_core = stream_id == 0 ? total_cores -2 : total_cores -1;
    for (unsigned core = start_core; core <= end_core; core+=2) {
      core_range.insert(core);
      core_stream_mapping[core] = stream_id;
      core_range_vector.push_back(core);
    }
    global_dynamic_core_ranges[stream_id] = core_range;
    global_dynamic_core_ranges_vector[stream_id] = core_range_vector;
    std::cout << "GLOBAL: STREAM " << stream_id << " -> cores [";
    for (auto core : core_range_vector) {
      std::cout << core << " ";
    }
    std::cout << std::endl;

    // update core range for m_running_kernels_stream
    if (stream_id == 0) {
      for (auto& kernel : m_running_kernels_stream1) {
        if (kernel!=nullptr) {
          kernel->set_core_range(core_range);
        }
      }
    } else {
      for (auto& kernel : m_running_kernels_stream2) {
        if (kernel!=nullptr) {
          kernel->set_core_range(core_range);
        }
      }
    }

    // also update the m_running_kernels

    for (auto& kernel : m_running_kernels) {
      if (kernel!=nullptr && kernel->get_streamID() == stream_id) {
        kernel->set_core_range(core_range);
      }
    }


    // print out the core range for this stream
    std::cout << "GLOBAL: STREAM " << stream_id << " -> cores ["; 
    for (auto core_id : core_range) {
      std::cout << core_id << " ";
    }
    std::cout << "]" << std::endl;
  }


  for (unsigned core_id = start_core_id; core_id < total_cores; core_id+=1) {
    shader_core_ctx *core = get_core_by_sid(core_id);
    if (is_stream0_bypass && core_stream_mapping[core_id] == 0) {
      core->set_stream_bypassL1D_new(0,true);
    }
    if (is_stream1_bypass && core_stream_mapping[core_id] == 1) {
      core->set_stream_bypassL1D_new(1,true);
    }
  }

  // print out bypass L1D information
  // for (unsigned core_id = 0; core_id < total_cores; core_id+=1) {
  //   shader_core_ctx *core = get_core_by_sid(core_id);
  //   printf("CORE %u: bypassL1D = %d, stream0_bypassL1D = %d, stream1_bypassL1D = %d\n",
  //         core_id, core->get_bypassL1D(), core->get_stream_bypassL1D(0), core->get_stream_bypassL1D(1));
  // }
}

void gpgpu_sim::dynamic_scheduling_exclusive(
  bool is_stream0_bypass, bool is_stream1_bypass) {
  // clear the global_stream_core_ranges_set
  global_dynamic_core_ranges.clear();
  global_dynamic_core_ranges_vector.clear();

  // Create sorted stream list for consistent core assignment
  std::vector<unsigned long long> stream_list(global_unique_streams.begin(), global_unique_streams.end());
  std::sort(stream_list.begin(), stream_list.end());

  unsigned total_cores = get_config().num_shader();

    // in the dyno core scheduling, the first 12 core are always in sampling phase
  // so we need to start from the 13th core
  unsigned start_core_id = m_config.get_dyno_core_scheduling() ? 12 : 0;
  unsigned end_core_id_stream0 = m_config.get_dyno_core_scheduling() ?
                                (total_cores-12)/2-1+12:total_cores/2 -1;
  core_stream_mapping.clear();

  // stream 0 use half of the cores
  // stream 1 use the other half
  for (unsigned i = 0; i < stream_list.size(); i++) {
    std::set<unsigned> core_range;
    std::vector<unsigned> core_range_vector;
    unsigned long long stream_id = stream_list[i];
    unsigned start_core = stream_id == 0 ? start_core_id : end_core_id_stream0+1;
    unsigned end_core = stream_id == 0 ? end_core_id_stream0 : total_cores -1;

    for (unsigned core = start_core; core <= end_core; core++) {
      core_range.insert(core);
      core_stream_mapping[core] = stream_id;
      core_range_vector.push_back(core);
    }
    global_dynamic_core_ranges[stream_id] = core_range;
    global_dynamic_core_ranges_vector[stream_id] = core_range_vector;
    // print out the core range for this stream
    std::cout << "GLOBAL: STREAM " << stream_id << " -> cores ["; 
    for (auto core_id : core_range_vector) {
      std::cout << core_id << " ";
    }
    std::cout << "]" << std::endl;

    // update core range for m_running_kernels_stream
    if (stream_id == 0) {
      for (auto& kernel : m_running_kernels_stream1) {
        if (kernel!=nullptr) {
          kernel->set_core_range(core_range);
        }
      }
    } else {
      for (auto& kernel : m_running_kernels_stream2) {
        if (kernel!=nullptr) {
          kernel->set_core_range(core_range);
        }
      }
    }

    // also update the m_running_kernels

    for (auto& kernel : m_running_kernels) {
      if (kernel!=nullptr && kernel->get_streamID() == stream_id) {
        kernel->set_core_range(core_range);
      }
    }
  }

  // set bypass L1D for each core
  for (unsigned i = 0; i < stream_list.size(); i++) {
    unsigned long long stream_id = stream_list[i];
    unsigned start_core = stream_id == 0 ? 0 : total_cores/2;
    unsigned end_core = stream_id == 0 ? total_cores/2 -1 : total_cores -1;
    for (unsigned coreId = start_core; coreId <= end_core; coreId++) {
      shader_core_ctx *core = get_core_by_sid(coreId);
      if (is_stream0_bypass && stream_id == 0) {
        core->set_stream_bypassL1D_new(0,true);
      }
      if (is_stream1_bypass && stream_id == 1) {
        core->set_stream_bypassL1D_new(1,true);
      }
    }
  }

  // print out bypass L1D information
  // for (unsigned core_id = 0; core_id < total_cores; core_id+=1) {
  //   shader_core_ctx *core = get_core_by_sid(core_id);
  //   printf("CORE %u: bypassL1D = %d, stream0_bypassL1D = %d, stream1_bypassL1D = %d\n",
  //         core_id, core->get_bypassL1D(), core->get_stream_bypassL1D(0), core->get_stream_bypassL1D(1));
  // }
}


void gpgpu_sim::gpu_print_stat(unsigned long long streamID) {
  FILE *statfout = stdout;

  std::string kernel_info_str = executed_kernel_info_string();
  fprintf(statfout, "%s", kernel_info_str.c_str());

  printf("kernel_stream_id = %llu\n", streamID);

  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  printf("gpu_core_abs_cycle = %lld\n",gpu_core_abs_cycle);
  printf("gpu_sim_insn = %lld\n", gpu_sim_insn);

  // print out gpu kernel time
  for (auto iter = gpu_kernel_time.begin(); iter !=gpu_kernel_time.end(); iter++) {
    for (auto kernel_iter = iter->second.begin(); kernel_iter != iter->second.end(); kernel_iter++) {
      fprintf(statfout, "gpu_kernel_time[%u][%u] = %lld\n",
              iter->first, kernel_iter->first,
              kernel_iter->second.end_cycle -
              kernel_iter->second.start_cycle);
    }
  }

  m_shader_stats->print_ipc(stdout);
  
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);

  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
  printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

  // printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
  // printf("partiton_reqs_in_parallel_total    = %lld\n",
  // partiton_reqs_in_parallel_total );
  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  printf("partiton_level_parallism_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
             (gpu_tot_sim_cycle + gpu_sim_cycle));
  // printf("partiton_reqs_in_parallel_util = %lld\n",
  // partiton_reqs_in_parallel_util);
  // printf("partiton_reqs_in_parallel_util_total    = %lld\n",
  // partiton_reqs_in_parallel_util_total ); printf("gpu_sim_cycle_parition_util
  // = %lld\n", gpu_sim_cycle_parition_util);
  // printf("gpu_tot_sim_cycle_parition_util    = %lld\n",
  // gpu_tot_sim_cycle_parition_util );
  printf("partiton_level_parallism_util = %12.4f\n",
         (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel_util +
                 partiton_reqs_in_parallel_util_total) /
             (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
  // printf("partiton_replys_in_parallel = %lld\n",
  // partiton_replys_in_parallel); printf("partiton_replys_in_parallel_total =
  // %lld\n", partiton_replys_in_parallel_total );
  printf("L2_BW  = %12.4f GB/Sec\n",
         ((float)(partiton_replys_in_parallel * 32) /
          (gpu_sim_cycle * m_config.core_period)) /
             1000000000);
  printf("L2_BW_total  = %12.4f GB/Sec\n",
         ((float)((partiton_replys_in_parallel +
                   partiton_replys_in_parallel_total) *
                  32) /
          ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.core_period)) /
             1000000000);

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

if (!m_config.get_stream_partitioning()) {
  // shader_print_l1_miss_stat( stdout );
  shader_print_cache_stats(stdout);
}

  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }

if (!m_config.get_stream_partitioning()) {
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, streamID,
                               "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout, streamID,
                                    "Total_core_cache_fail_stats_breakdown");
  shader_print_scheduler_stat(stdout, false);
}

  // in stream partitioning mode, we don't print out the following stats
  if (!m_config.get_stream_partitioning()) {
    m_shader_stats->print(stdout);
  }
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    if (m_config.g_power_simulation_mode > 0) {
      // if(!m_config.g_aggregate_power_stats)
      mcpat_reset_perf_count(m_gpgpusim_wrapper);
      calculate_hw_mcpat(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                         m_power_stats, m_config.gpu_stat_sample_freq,
                         gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                         gpu_sim_insn, m_config.g_power_simulation_mode,
                         m_config.g_dvfs_enabled, m_config.g_hw_perf_file_name,
                         m_config.g_hw_perf_bench_name, executed_kernel_name(),
                         m_config.accelwattch_hybrid_configuration,
                         m_config.g_aggregate_power_stats);
    }
    m_gpgpusim_wrapper->print_power_kernel_stats(
        gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
        kernel_info_str, true);
    // if(!m_config.g_aggregate_power_stats)
    mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif
if (!m_config.get_stream_partitioning()) {
  // performance counter that are not local to one shader
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    if (stream_num == 1) {
      m_memory_partition_unit[i]->print(stdout);
    }
  }
}


if (!m_config.get_stream_partitioning()) {
  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, streamID, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, streamID,
                                "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }
}
  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->detect_print_steady_state(
        1, gpu_tot_sim_insn + gpu_sim_insn);
  }
#endif

  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  time_vector_print();
  fflush(stdout);

  clear_executed_kernel_info();

  executed_kernel_num++;
  if (m_config.gpgpu_max_executed_kernel_num != 0 &&
      executed_kernel_num >= m_config.gpgpu_max_executed_kernel_num) {
    printf(
        "Terminating GPU simulation early due to max executed kernel limit.\n");
    exit(0);
  }
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case sstarr_space:
      m_stats->gpgpu_n_sstarr_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      abort();
  }
}
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas, on shader %d\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas, m_sid);
  }

  return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  if (!m_config->gpgpu_concurrent_kernel_sm)
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  /*
  std::cout << "issue_block2core: " << kernel.get_name() <<
  " core_id: " << m_sid << " stream_id: " << kernel.get_streamID() <<
  " kernel.print_core_range(): " << kernel.print_core_range() <<
  " at cycle: " << m_gpu->gpu_sim_cycle << std::endl;
  */

  kernel.inc_running();

  if (m_stats->shader_start_cycle[m_sid] == 0) {
    m_stats->shader_start_cycle[m_sid] = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
    m_stats->shader_current_cycle[m_sid] = m_stats->shader_start_cycle[m_sid];
    m_stats->shader_profile_cycle[m_sid] = m_stats->shader_start_cycle[m_sid];
    m_stats->m_num_sim_curr_kernel_insn[m_sid] = 0;
  }

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(),
                 kernel.get_name().c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    printf("Queue Length DRAM[%d] ", id);
    StatDisp(mrqq_Dist);
  }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  double smallest = min3(core_time, icnt_time, dram_time);
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  return mask;
}

void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      m_total_cta_launched += num;
    }
  }
}

void gpgpu_sim::issue_block2core_by_core_range() {
  if (m_config.get_dynamic_core_scheduling() ||
      m_config.get_dyno_core_scheduling()) {
    for (auto& pair : global_dynamic_core_ranges_vector) {
      unsigned streamID = pair.first;
      std::vector<unsigned>& core_range = pair.second;
      for (unsigned i = 0; i < core_range.size(); i++) {
        unsigned core_id = get_next_core_id_by_core_range(streamID);
        unsigned num = m_cluster[core_id]->issue_block2core();
        if (num) {
          if (streamID == 0) {
            m_last_cluster_issue_stream1 = core_id;
          } else if (streamID == 1) {
            m_last_cluster_issue_stream2 = core_id;
          } else {
            assert(false && "Invalid stream id");
          }
          m_total_cta_launched += num;
        }
      }
    }
  } else {
    for (auto& pair : global_stream_core_ranges_vector) {
      unsigned streamID = pair.first;
      std::vector<unsigned>& core_range = pair.second;
      for (unsigned i = 0; i < core_range.size(); i++) {
        unsigned core_id = get_next_core_id_by_core_range(streamID);
        unsigned num = m_cluster[core_id]->issue_block2core();
        if (num) {
          if (streamID == 0) {
            m_last_cluster_issue_stream1 = core_id;
          } else if (streamID == 1) {
            m_last_cluster_issue_stream2 = core_id;
          } else {
            assert(false && "Invalid stream id");
          }
          m_total_cta_launched += num;
        }
      }
    }
 }
}


/*

void gpgpu_sim::issue_block2core_by_core_range() {
  std::set<unsigned> stream0_core_ids;
  std::set<unsigned> stream1_core_ids;

  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    unsigned stream_id = core_stream_mapping[i];
    unsigned core_id = get_next_core_id_by_core_range(stream_id);
    if (stream_id == 0) {
      if (stream0_core_ids.find(core_id) != stream0_core_ids.end()) {
        continue;
      }
    } else if (stream_id == 1) {
      if (stream1_core_ids.find(core_id) != stream1_core_ids.end()) {
        continue;
      }
    } else {
      assert(false && "Invalid stream id");
    }

    unsigned num = m_cluster[core_id]->issue_block2core();
    if (stream_id == 0 && num > 0) {
      stream0_core_ids.insert(core_id);
      //std::cout << "STREAM 0: last issued " << m_last_cluster_issue_stream1 << " issue " << num << core_id << std::endl;
      m_last_cluster_issue_stream1 = core_id;
    } else if (stream_id == 1 && num > 0) {
      stream1_core_ids.insert(core_id);
      //std::cout << "STREAM 1: last issued " << m_last_cluster_issue_stream2 << " issue " << num << core_id << std::endl;
      m_last_cluster_issue_stream2 = core_id;
    }
    if (num) {
      m_total_cta_launched += num;
    }
  }
}

*/

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline

void gpgpu_sim::cycle() {
  int clock_mask = next_clock_domain();

  if (clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      m_cluster[i]->icnt_cycle();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  if (clock_mask & ICNT) {
    // pop from memory controller to interconnect
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          // if (!mf->get_is_write())
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size);
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;
        }
      } else {
        m_memory_sub_partition[i]->pop();
      }
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (clock_mask & DRAM) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m_memory_config->simple_dram_model)
        m_memory_partition_unit[i]->simple_dram_model_cycle();
      else
        m_memory_partition_unit[i]
            ->dram_cycle();  // Issue the dram command (scheduler + delay model)
      // Update performance counters for DRAM
      if (m_config.g_power_simulation_enabled) {
        m_memory_partition_unit[i]->set_dram_power_stats(
            m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
      }
    }
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;
  if (clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;
      } else {
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        if (mf) {
          // temp debug
          //printf("Push mf to m_memory_sub_partition[%d]: %p, data_size: %d, addr: %lx, dynamic_fetch_mode=%d\n",
          //i, mf, mf->get_data_size(), mf->get_addr(), mf->get_dynamic_fetch_mode());
          //mf->print(stdout, false);
          partiton_reqs_in_parallel_per_cycle++;
        }
      }
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      if (m_config.g_power_simulation_enabled) {
        m_memory_sub_partition[i]->accumulate_L2cache_stats(
            m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
      }
    }
  }
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (clock_mask & ICNT) {
    icnt_transfer();
  }

  if (clock_mask & CORE) {
    // L1 cache + shader core pipeline stages
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        m_cluster[i]->core_cycle();
        *active_sms += m_cluster[i]->get_n_active_sms();
      }
      // Update core icnt/cache stats for AccelWattch
      if (m_config.g_power_simulation_enabled) {
        m_cluster[i]->get_icnt_stats(
            m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
        m_cluster[i]->get_cache_stats(
            m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
      }
      m_cluster[i]->get_current_occupancy(
          gpu_occupancy.aggregate_warp_slot_filled,
          gpu_occupancy.aggregate_theoretical_warp_slots);
    }
    float temp = 0;
    for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
      temp += m_shader_stats->m_pipeline_duty_cycle[i];
    }
    temp = temp / m_shader_config->num_shader();
    *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
    // cout<<"Average pipeline duty cycle:
    // "<<*average_pipeline_duty_cycle<<endl;

    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      raise(SIGTRAP);  // Debug breakpoint
    }
    gpu_sim_cycle++;
    gpu_core_abs_cycle++;

    if (g_interactive_debugger_enabled) gpgpu_debug();

      // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
      if (m_config.g_power_simulation_mode == 0) {
        mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                    m_power_stats, m_config.gpu_stat_sample_freq,
                    gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                    gpu_sim_insn, m_config.g_dvfs_enabled);
      }
    }
#endif

  if (m_config.get_dynamic_core_scheduling() ||
      m_config.get_dyno_core_scheduling()) {
    issue_block2core_by_core_range();
  } else if (m_config.get_stream_partitioning()) {
    //issue_block2core_stream_partitioning();
    issue_block2core_by_core_range();
  } else {
    issue_block2core();
  }
    decrement_kernel_latency();

    // Depending on configuration, invalidate the caches once all of threads are
    // completed.
    int all_threads_complete = 1;
    if (m_config.gpgpu_flush_l1_cache) {
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (m_cluster[i]->get_not_completed() == 0)
          m_cluster[i]->cache_invalidate();
        else
          all_threads_complete = 0;
      }
    }

    if (m_config.gpgpu_flush_l2_cache) {
      if (!m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() != 0) {
            all_threads_complete = 0;
            break;
          }
        }
      }

      if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
        printf("Flushed L2 caches...\n");
        if (m_memory_config->m_L2_config.get_num_lines()) {
          int dlc = 0;
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            dlc = m_memory_sub_partition[i]->flushL2();
            assert(dlc == 0);  // TODO: need to model actual writes to DRAM here
            printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
          }
        }
      }
    }

    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
      time_t days, hrs, minutes, sec;
      time_t curr_time;
      time(&curr_time);
      unsigned long long elapsed_time =
          MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
      if ((elapsed_time - last_liveness_message_time) >=
              m_config.liveness_message_freq &&
          DTRACE(LIVENESS)) {
        days = elapsed_time / (3600 * 24);
        hrs = elapsed_time / 3600 - 24 * days;
        minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
        sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

        unsigned long long active = 0, total = 0;
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          m_cluster[i]->get_current_occupancy(active, total);
        }
        DPRINTFG(LIVENESS,
                 "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f%% [%llu / %llu]) "
                 "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                 gpu_tot_sim_insn + gpu_sim_insn,
                 (double)gpu_sim_insn / (double)gpu_sim_cycle,
                 float(active) / float(total) * 100, active, total,
                 (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                 (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                 (unsigned)sec, ctime(&curr_time));
        fflush(stdout);
        last_liveness_message_time = elapsed_time;
      }
      visualizer_printstat();
      m_memory_stats->memlatstat_lat_pw();
      if (m_config.gpgpu_runtime_stat &&
          (m_config.gpu_runtime_stat_flag != 0)) {
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
            m_memory_partition_unit[i]->print_stat(stdout);
          printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
          printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
        }
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
          shader_print_runtime_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
          shader_print_l1_miss_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
          shader_print_scheduler_stat(stdout, false);
      }
    }

    // DEADLOCK DETECTION
    if (!(gpu_sim_cycle % 100000)) {
      // deadlock detection
      if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
        gpu_deadlock = true;
      } else {
        last_gpu_sim_insn = gpu_sim_insn;
      }
    }
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
    // launch device kernel
    gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
  }
}

void sst_gpgpu_sim::cycle() {
  SST_cycle();
  return;
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w]->print(fout);
}

void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    for (unsigned counter = 0; counter < count; counter += SECTOR_SIZE) {
      const unsigned wr_addr = dst_start_addr + counter;
      addrdec_t raw_addr;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / SECTOR_SIZE);
      m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
      const unsigned partition_id =
          raw_addr.sub_partition /
          m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, raw_addr.sub_partition, mask);
    }
  }
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }

simt_core_cluster *gpgpu_sim::getSIMTCluster() { return *m_cluster; }

// Stream-based core partitioning function implementation
void gpgpu_sim::set_kernel_core_range(kernel_info_t *kernel, unsigned start_core, unsigned end_core) {
  // Store the core range information in the kernel for use during shader core allocation
  // This enables stream-based resource partitioning
  std::cout << "Setting core range for kernel " << kernel->get_name() 
            << " (uid:" << kernel->get_uid() << ", stream:" << kernel->get_streamID() 
            << ") to cores [" << start_core << "-" << end_core << "]" << std::endl;
          
  // Store core range in kernel metadata
  kernel->set_core_range(start_core, end_core);
  
  std::cout << "CONFIRMED: Kernel " << kernel->get_name() 
            << " core range set - has_range:" << kernel->has_core_range()
            << ", core_range:" << kernel->print_core_range() << std::endl;
}

void gpgpu_sim::set_kernel_core_range(kernel_info_t *kernel, std::set<unsigned> core_range) {
  // Store core range in kernel metadata
  kernel->set_core_range(core_range);
  std::cout << "CONFIRMED: Kernel " << kernel->get_name() 
            << " core range set - has_range:" << kernel->has_core_range()
            << ", core_range:" << kernel->print_core_range() << std::endl;
}

void sst_gpgpu_sim::SST_gpgpusim_numcores_equal_check(unsigned sst_numcores) {
  if (m_shader_config->n_simt_clusters != sst_numcores) {
    assert(
        "\nSST core is not equal the GPGPU-sim cores. Open gpgpu-sim.config "
        "file and ensure n_simt_clusters"
        "is the same as SST gpu cores.\n" &&
        0);
  } else {
    printf("\nSST GPU core is equal the GPGPU-sim cores = %d\n", sst_numcores);
  }
}

void sst_gpgpu_sim::SST_cycle() {
  // shader core loading (pop from ICNT into core) follows CORE clock
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    static_cast<sst_simt_core_cluster *>(m_cluster[i])->icnt_cycle_SST();

  // L1 cache + shader core pipeline stages
  m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
      m_cluster[i]->core_cycle();
      *active_sms += m_cluster[i]->get_n_active_sms();
    }
    // Update core icnt/cache stats for GPUWattch
    m_cluster[i]->get_icnt_stats(
        m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
        m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
    m_cluster[i]->get_cache_stats(
        m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
  }
  float temp = 0;
  for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
    temp += m_shader_stats->m_pipeline_duty_cycle[i];
  }
  temp = temp / m_shader_config->num_shader();
  *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
  // cout<<"Average pipeline duty cycle: "<<*average_pipeline_duty_cycle<<endl;

  if (g_single_step && ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
    asm("int $03");
  }
  gpu_sim_cycle++;
  if (g_interactive_debugger_enabled) gpgpu_debug();

    // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                m_power_stats, m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle,
                gpu_sim_cycle, gpu_tot_sim_insn, gpu_sim_insn,
                m_config.g_dvfs_enabled);
  }
#endif

  issue_block2core();

  if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
    time_t days, hrs, minutes, sec;
    time_t curr_time;
    time(&curr_time);
    unsigned long long elapsed_time =
        MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
    if ((elapsed_time - last_liveness_message_time) >=
        m_config.liveness_message_freq) {
      days = elapsed_time / (3600 * 24);
      hrs = elapsed_time / 3600 - 24 * days;
      minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
      sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

      last_liveness_message_time = elapsed_time;
    }
    visualizer_printstat();
    m_memory_stats->memlatstat_lat_pw();
    if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0)) {
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
          m_memory_partition_unit[i]->print_stat(stdout);
        printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
        printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
      }
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
        shader_print_runtime_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
        shader_print_l1_miss_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
        shader_print_scheduler_stat(stdout, false);
    }
  }

  if (!(gpu_sim_cycle % 1000)) {
    // deadlock detection
    if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
      gpu_deadlock = true;
    } else {
      last_gpu_sim_insn = gpu_sim_insn;
    }
  }
  try_snap_shot(gpu_sim_cycle);
  spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
  // launch device kernel
  gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
}

unsigned long long gpgpu_sim::get_first_stream_id() const {
  // Helper function to get the first stream ID for stream partitioning
  // This assumes we track the first stream we see
  for (unsigned i = 0; i < m_running_kernels.size(); i++) {
    if (m_running_kernels[i] != NULL) {
      return m_running_kernels[i]->get_streamID();
    }
  }
  return 0;  // Default stream ID
}

ldst_unit *gpgpu_sim::get_ldst_unit(unsigned ldst_unit_id) {
  // Map ldst_unit_id to shader core ID
  // Each shader core has one ldst_unit, so ldst_unit_id corresponds to shader core ID
  if (ldst_unit_id >= m_shader_config->num_shader()) {
    // Invalid ldst_unit_id
    return NULL;
  }
  
  // Get the cluster and core within the cluster
  unsigned cluster_id = m_shader_config->sid_to_cluster(ldst_unit_id);
  unsigned core_id = m_shader_config->sid_to_cid(ldst_unit_id);
  
  // Access the ldst_unit from the shader core using the public method
  shader_core_ctx *core = m_cluster[cluster_id]->get_core(core_id);
  if (core == NULL) {
    return NULL;
  }
  return core->m_ldst_unit;
}

void gpgpu_sim::reset_simulation_state() {
  printf("=== RESETTING SIMULATION EXECUTION STATE ===\n");
  
  // Reset global cycle counters
  gpu_sim_cycle = 0;
  gpu_core_abs_cycle = 0;
  gpu_tot_sim_cycle = 0;
  
  // Clear all shader cores execution state
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    for (unsigned j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
      unsigned sid = m_shader_config->cid_to_sid(j, i);
      shader_core_ctx *core = get_core_by_sid(sid);
      if (core) {
        core->clear_extra_stats();
      }
    }
  }
  
  // Invalidate all caches by calling on each core
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    for (unsigned j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
      unsigned sid = m_shader_config->cid_to_sid(j, i);
      shader_core_ctx *core = get_core_by_sid(sid);
      if (core) {
        core->cache_flush();
        core->cache_invalidate();
      }
    }
  }
  
  // Clear kernel tracking
  m_finished_kernel.clear();
  
  printf("=== SIMULATION STATE RESET COMPLETE ===\n");
}
