// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington,
// Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan,
// Timothy G. Rogers
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

#include "gpu-cache.h"
#include <assert.h>
#include "gpu-sim.h"
#include "hashing.h"
#include "stat-tool.h"

// used to allocate memory that is large enough to adapt the changes in cache
// size across kernels

const char *cache_request_status_str(enum cache_request_status status) {
  static const char *static_cache_request_status_str[] = {
      "HIT",         "HIT_RESERVED", "MISS", "RESERVATION_FAIL",
      "SECTOR_MISS", "MSHR_HIT" ,"EVICTION",};

  assert(sizeof(static_cache_request_status_str) / sizeof(const char *) ==
         NUM_CACHE_REQUEST_STATUS);
  assert(status < NUM_CACHE_REQUEST_STATUS);

  return static_cache_request_status_str[status];
}

const char *cache_fail_status_str(enum cache_reservation_fail_reason status) {
  static const char *static_cache_reservation_fail_reason_str[] = {
      "LINE_ALLOC_FAIL", "MISS_QUEUE_FULL", "MSHR_ENRTY_FAIL",
      "MSHR_MERGE_ENRTY_FAIL", "MSHR_RW_PENDING"};

  assert(sizeof(static_cache_reservation_fail_reason_str) /
             sizeof(const char *) ==
         NUM_CACHE_RESERVATION_FAIL_STATUS);
  assert(status < NUM_CACHE_RESERVATION_FAIL_STATUS);

  return static_cache_reservation_fail_reason_str[status];
}

unsigned l1d_cache_config::set_bank(new_addr_type addr) const {
  // For sector cache, we select one sector per bank (sector interleaving)
  // This is what was found in Volta (one sector per bank, sector interleaving)
  // otherwise, line interleaving
  return cache_config::hash_function(addr, l1_banks,
                                     l1_banks_byte_interleaving_log2,
                                     l1_banks_log2, l1_banks_hashing_function);
}

unsigned cache_config::set_index(new_addr_type addr) const {
  return cache_config::hash_function(addr, m_nset, m_line_sz_log2, m_nset_log2,
                                     m_set_index_function);
}

unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                     unsigned m_line_sz_log2,
                                     unsigned m_nset_log2,
                                     unsigned m_index_function) const {
  unsigned set_index = 0;

  switch (m_index_function) {
    case FERMI_HASH_SET_FUNCTION: {
      /*
       * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
       * Distance Theory" Cedric Nugteren et al. HPCA 2014
       */
      unsigned lower_xor = 0;
      unsigned upper_xor = 0;

      if (m_nset == 32 || m_nset == 64) {
        // Lower xor value is bits 7-11
        lower_xor = (addr >> m_line_sz_log2) & 0x1F;

        // Upper xor value is bits 13, 14, 15, 17, and 19
        upper_xor = (addr & 0xE000) >> 13;    // Bits 13, 14, 15
        upper_xor |= (addr & 0x20000) >> 14;  // Bit 17
        upper_xor |= (addr & 0x80000) >> 15;  // Bit 19

        set_index = (lower_xor ^ upper_xor);

        // 48KB cache prepends the set_index with bit 12
        if (m_nset == 64) set_index |= (addr & 0x1000) >> 7;

      } else { /* Else incorrect number of sets for the hashing function */
        assert(
            "\nGPGPU-Sim cache configuration error: The number of sets should "
            "be "
            "32 or 64 for the hashing set index function.\n" &&
            0);
      }
      break;
    }

    case BITWISE_XORING_FUNCTION: {
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = bitwise_hash_function(higher_bits, index, m_nset);
      break;
    }
    case HASH_IPOLY_FUNCTION: {
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = ipoly_hash_function(higher_bits, index, m_nset);
      break;
    }
    case CUSTOM_SET_FUNCTION: {
      /* No custom set function implemented */
      break;
    }

    case LINEAR_SET_FUNCTION: {
      set_index = (addr >> m_line_sz_log2) & (m_nset - 1);
      break;
    }

    default: {
      assert("\nUndefined set index function.\n" && 0);
      break;
    }
  }

  // Linear function selected or custom set index function not implemented
  assert((set_index < m_nset) &&
         "\nError: Set index out of bounds. This is caused by "
         "an incorrect or unimplemented custom set index function.\n");

  return set_index;
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping) {
  cache_config::init(m_config_string, FuncCachePreferNone);
  m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const {
  new_addr_type part_addr = addr;

  if (m_address_mapping) {
    // Calculate set index without memory partition bits to reduce set camping
    part_addr = m_address_mapping->partition_address(addr);
  }

  return cache_config::set_index(part_addr);
}

tag_array::~tag_array() {
  unsigned cache_lines_num = m_config.get_max_num_lines();
  for (unsigned i = 0; i < cache_lines_num; ++i) delete m_lines[i];
  delete[] m_lines;
}

tag_array::tag_array(const char *name, cache_config &config, int core_id, int type_id,
                     cache_block_t **new_lines)
    : m_name(name),m_config(config), m_lines(new_lines) {
  init(core_id, type_id);
}

void tag_array::update_cache_parameters(cache_config &config) {
  m_config = config;
}

tag_array::tag_array(const char *name, cache_config &config, int core_id, int type_id)
    : m_name(name), m_config(config) {
  // assert( m_config.m_write_policy == READ_ONLY ); Old assert
  unsigned cache_lines_num = config.get_max_num_lines();
  m_lines = new cache_block_t *[cache_lines_num];
  if (config.m_cache_type == NORMAL) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new line_cache_block();
  } else if (config.m_cache_type == SECTOR) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new sector_cache_block();
  } else
    assert(0);

  init(core_id, type_id);

  // Stream partitioning configuration
  if (config.is_stream_partitioning_enabled()) {
    m_stream_partitioning_enabled = true;
    m_stream_allocations = config.m_stream_allocations;
    for (auto &entry : m_stream_allocations) {
      unsigned long long streamID = entry.first;
      float percentage = entry.second;
      set_stream_allocation(streamID, percentage);
    }

    printf("stream 0/1 way number: %llu/%llu\n",
           stream0_wayNum, stream1_wayNum);
  } else {
    m_stream_partitioning_enabled = false;
  }
}

void tag_array::init(int core_id, int type_id) {
  m_access = 0;
  m_miss = 0;
  m_pending_hit = 0;
  m_res_fail = 0;
  m_sector_miss = 0;
  // initialize snapshot counters for visualizer
  m_prev_snapshot_access = 0;
  m_prev_snapshot_miss = 0;
  m_prev_snapshot_pending_hit = 0;
  m_core_id = core_id;
  m_type_id = type_id;
  is_used = false;
  m_dirty = 0;

  // Snapshot counters
  m_prev_snapshot_evictions = 0;
  m_prev_snapshot_evictions_no_reuse = 0;
}

void tag_array::add_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i == pending_lines.end()) {
    pending_lines[addr] = mf->get_inst().get_uid();
  }
}

void tag_array::remove_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i != pending_lines.end()) {
    pending_lines.erase(addr);
  }
}

enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_fetch *mf, bool is_write,
                                           bool probe_mode){
  mem_access_sector_mask_t mask = mf->get_access_sector_mask();
  return probe(addr, idx, mask, is_write, mf->get_sid(), probe_mode, mf);
}

bool tag_array::stream_reserved_exceeds_allocation(new_addr_type addr, unsigned long long streamID) const {
  unsigned max_ways = streamID == 0 ? stream0_wayNum : stream1_wayNum;
  unsigned set_index = m_config.set_index(addr);
  unsigned reserved_count = 0;
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];
    if (line->m_stream_id == streamID && line->is_reserved_line()) {
      reserved_count++;
      if (reserved_count >= max_ways) {
        return true;
      }
    }
  }
  return false;
}

void tag_array::update_sector_mask_stats(mem_access_sector_mask_t sector_mask, uint64_t addr,
                                         warp_inst_t inst){
  auto it = addr_to_sector_mask.find(addr);
  uint64_t num_sector_mask = get_sector_mask_key(sector_mask);
  if (it != addr_to_sector_mask.end()) {
    if (it->second == num_sector_mask) {
      return;
    } else {
      if (sector_mask_stats[num_sector_mask] > 0) {
        sector_mask_stats[num_sector_mask]--;
      } else {
        sector_mask_stats.erase(num_sector_mask);
      }
    }
  }
  addr_to_sector_mask[addr] = num_sector_mask;
  sector_mask_stats[num_sector_mask]++;
  addr_to_pc_set[addr].insert(inst.get_pc());
  inst_to_mask_set[inst.get_pc()].insert(num_sector_mask);
  pc_to_inst_set[inst.get_pc()] = inst;
}

void tag_array::print_sector_mask_stats(FILE *stream){
  if (addr_to_sector_mask.size() == 0)
    return;
  for (auto it = sector_mask_stats.begin(); it != sector_mask_stats.end(); it++) {
    std::bitset<4> sector_mask(it->first);
    fprintf(stream, "%s sector_mask: %s, count: %d\n",
    __func__, sector_mask.to_string().c_str(), it->second);
  }

  for (auto it = addr_to_pc_set.begin(); it != addr_to_pc_set.end(); it++) {
    std::bitset<4> sector_mask(addr_to_sector_mask[it->first]);
    fprintf(stream, "%s addr: %0#llx sector_mask: %s ",
            __func__, it->first,sector_mask.to_string().c_str());

    for (auto pc_it = it->second.begin(); pc_it != it->second.end(); pc_it++) {
      auto& inst = pc_to_inst_set[*pc_it];
      inst.print_pc(stream);
    }
    fprintf(stream, "\n");
  }

  for (auto it = inst_to_mask_set.begin(); it != inst_to_mask_set.end(); it++) {
    fprintf(stream, "%s inst: %0#llx, mask: %d ", __func__, it->first, it->second.size());
    for (auto mask_it = it->second.begin(); mask_it != it->second.end(); mask_it++) {
      std::bitset<4> sector_mask(*mask_it);
      fprintf(stream, "  mask: %s", sector_mask.to_string().c_str());
      // print out the inst
      warp_inst_t& inst = pc_to_inst_set[it->first];
      inst.print_pc(stream);
      fprintf(stream," ");
    }
    fprintf(stream, "\n");
  }
  fprintf(stream, "%s total sector mask count: %d\n", __func__, sector_mask_stats.size());
  fprintf(stream, "%s total addr count: %d\n", __func__, addr_to_sector_mask.size());

  sector_mask_stats.clear();
  addr_to_sector_mask.clear();
  addr_to_pc_set.clear();
  inst_to_mask_set.clear();
  pc_to_inst_set.clear();
}

void tag_array::update_sid_to_addr_set(unsigned sid, new_addr_type addr) {
  // align to 128 bytes
  addr = addr & (~(128-1));
  auto it = m_sid_to_addr_set.find(sid);
  if (it != m_sid_to_addr_set.end() && it->second.find(addr) != it->second.end()) {
    it->second.insert(addr);
  } else {
    m_sid_to_addr_set[sid] = std::set<new_addr_type>{addr};
  }

  if (sid == 0) {
    // check hit in sid 1
    auto it = m_sid_to_addr_set.find(1);
    if (it != m_sid_to_addr_set.end() && it->second.find(addr) != it->second.end()) {
      m_extra_stats.m_hit_other_sid++;
      //printf("hit in sid 1\n");
    }
  } else if (sid == 1) {
    // check hit in sid 0
    auto it = m_sid_to_addr_set.find(0);
    if (it != m_sid_to_addr_set.end() && it->second.find(addr) != it->second.end()) {
      m_extra_stats.m_hit_other_sid++;
      //printf("hit in sid 0\n");
    }
  }
}

enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_access_sector_mask_t mask,
                                           bool is_write,
                                           unsigned sid,
                                           bool probe_mode,
                                           mem_fetch *mf){
  //update_sid_to_addr_set(sid, addr);
  // assert( m_config.m_write_policy == READ_ONLY );
  unsigned set_index = m_config.set_index(addr);
  new_addr_type tag = m_config.tag(addr);

  unsigned invalid_line = (unsigned)-1;
  unsigned valid_line = (unsigned)-1;
  unsigned long long valid_timestamp = (unsigned)-1;
  unsigned long long streamID = mf ? mf->get_streamID() : 0;

  // count total probe count
  m_extra_stats.m_total_probe_count++;
  
  bool all_reserved = true;
  // check for hit or pending hit
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];
    if (line->m_tag == tag && line->m_stream_id == streamID) {
      if (line->get_sid() != sid) {
        m_extra_stats.m_hit_other_sid++;
      }
      m_extra_stats.m_total_hit_count++;
      
      if (line->get_status(mask) == RESERVED) {
        idx = index;
        line->increment_access_count(mask);
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID) {
        idx = index;
        line->increment_access_count(mask);
        return HIT;
      } else if (line->get_status(mask) == MODIFIED) {
        if ((!is_write && line->is_readable(mask)) || is_write) {
          idx = index;
          line->increment_access_count(mask);
          return HIT;
        } else {
          idx = index;
          if (m_config.is_fill_entire_line()) {
            line->set_m_readable(true, mask);
            line->set_status(VALID, mask);
            line->increment_access_count(mask);
            return HIT;
          } else {
            return SECTOR_MISS;
          }
        }

      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        idx = index;
        if (m_config.is_fill_entire_line() ||
            m_config.is_fill_entire_line_on_clean()) {
          line->set_m_readable(true, mask);
          line->set_status(VALID, mask);
          line->increment_access_count(mask);
          update_sector_mask_stats(line->get_valid_sector_mask(), line->m_block_addr,
                                   mf->get_inst());
          return HIT;
        } else {
          return SECTOR_MISS;
        }
      } else {
        assert(line->get_status(mask) == INVALID);
      }
    }
    if (!line->is_reserved_line()) {
      // percentage of dirty lines in the cache
      // number of dirty lines / total lines in the cache
      float dirty_line_percentage =
          ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
      // If the cacheline is from a load op (not modified),
      // or the total dirty cacheline is above a specific value,
      // Then this cacheline is eligible to be considered for replacement
      // candidate i.e. Only evict clean cachelines until total dirty cachelines
      // reach the limit.
      if (!line->is_modified_line() ||
          dirty_line_percentage >= m_config.m_wr_percent) {
        all_reserved = false;
        if (line->is_invalid_line()) {
          invalid_line = index;
        } else {
          // valid line : keep track of most appropriate replacement candidate
          if (m_config.m_replacement_policy == LRU) {
            if (line->get_last_access_time() < valid_timestamp) {
              valid_timestamp = line->get_last_access_time();
              valid_line = index;
            }
          } else if (m_config.m_replacement_policy == FIFO) {
            if (line->get_alloc_time() < valid_timestamp) {
              valid_timestamp = line->get_alloc_time();
              valid_line = index;
            }
          }
        }
      }
    }
  }
  if (all_reserved) {
    assert(m_config.m_alloc_policy == ON_MISS);
    // print out memory request address and failed reasion
    //printf("Cache %s addr %0#llx, all lines reserved, "
    //    "not enough space to allocate on miss\n",
    //    get_name().c_str(), addr);
    return RESERVATION_FAIL;  // miss and not enough space in cache to allocate
                              // on miss
  }

  if (m_config.is_stream_partitioning_enabled()) {
    if (stream_exceeds_allocation(set_index, streamID)) {
        // Find LRU/FIFO victim within the same stream
        unsigned stream_victim = (unsigned)-1;
        unsigned long long stream_victim_time = (unsigned)-1;
        
        for (unsigned way = 0; way < m_config.m_assoc; way++) {
          unsigned index = set_index * m_config.m_assoc + way;
          cache_block_t *line = m_lines[index];
          
          if (!line->is_reserved_line() && line->m_stream_id == streamID) {
            if (m_config.m_replacement_policy == LRU) {
              if (line->get_last_access_time() < stream_victim_time) {
                stream_victim_time = line->get_last_access_time();
                stream_victim = index;
              }
            } else if (m_config.m_replacement_policy == FIFO) {
              if (line->get_alloc_time() < stream_victim_time) {
                stream_victim_time = line->get_alloc_time();
                stream_victim = index;
              } 
            }
          }

          //printf("#### streamID: %llu, index %d line->m_stream_id: %llu, "
          //      "line reserved/modified/valid: %d/%d/%d, last_access: %llu\n",
          //      streamID, index, line->m_stream_id, line->is_reserved_line(),
          //      line->is_modified_line(), line->is_valid_line(),
          //      line->get_last_access_time());
        }

        idx = stream_victim;
        
        if (stream_victim == (unsigned)-1) {
          for (unsigned way = 0; way < m_config.m_assoc; way++) {
            unsigned index = set_index * m_config.m_assoc + way;
            cache_block_t *line = m_lines[index];
            /*
            printf("## cache %s addr %0#llx streamID: %llu, index %llu line->m_stream_id: %llu, "
                  "reserved: %d, last_access: %llu\n",
                  get_name().c_str(), streamID, index, line->m_stream_id,
                  line->is_reserved_line(), line->get_last_access_time());
            */
          }
          return RESERVATION_FAIL;  // no enough ways reserved for the stream
        }
    } else if (invalid_line != (unsigned)-1){
      idx = invalid_line;
      
      for (unsigned way = 0; way < m_config.m_assoc; way++) {
          if (set_index != 7)
            continue;
          unsigned index = set_index * m_config.m_assoc + way;
          cache_block_t *line = m_lines[index];
          /*
          printf("#### cache %s addr %0#llx streamID: %llu, invalid %d line->m_stream_id: %llu, index %llu "
                "line reserved/modified/valid: %d/%d/%d, blk addr %0#llx last_access: %llu stream way %d cycle %llu\n",
                get_name().c_str(), addr, streamID, invalid_line, line->m_stream_id, index,
                line->is_reserved_line(), line->is_modified_line(), line->is_valid_line(),
                line->m_block_addr,
                line->get_last_access_time(), get_stream_wayNum(streamID), 
                gpu->gpu_tot_sim_cycle+gpu->gpu_sim_cycle);
          */
        
      }
            
    } else {
        
        for (unsigned way = 0; way < m_config.m_assoc; way++) {
          unsigned index = set_index * m_config.m_assoc + way;
          cache_block_t *line = m_lines[index];
          
          printf("#### cache %s addr %0#x streamID: %llu, line->m_stream_id: %llu, index %llu "
                "line reserved/modified/valid: %d/%d/%d, blk addr %0#llx last_access: %llu\n",
                get_name().c_str(),addr, streamID, line->m_stream_id, index, line->is_reserved_line(),
                line->is_modified_line(), line->is_valid_line(),
                line->m_block_addr,
                line->get_last_access_time());
        }
        
        
        // just hack
        abort();  // if an unreserved block exists, it is either invalid or
                  // replaceable
        //idx = valid_line;
    }

  } else {
      if (invalid_line != (unsigned)-1) {
        idx = invalid_line;
      } else if (valid_line != (unsigned)-1) {
        idx = valid_line;
        // this line will be evicted, so reset the sid
        m_lines[idx]->set_sid((unsigned) -1);
        m_extra_stats.m_all_evictions++;
      } else
        abort();  // if an unreserved block exists, it is either invalid or
                  // replaceable
  }

  return MISS;
}

enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, mem_fetch *mf) {
  bool wb = false;
  evicted_block_info evicted;
  enum cache_request_status result = access(addr, time, idx, wb, evicted, mf);
  assert(!wb);
  return result;
}

enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, bool &wb,
                                            evicted_block_info &evicted,
                                            mem_fetch *mf) {
  m_access++;
  is_used = true;
  shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to cache
  enum cache_request_status status = probe(addr, idx, mf, mf->is_write(), mf->get_sid());
  switch (status) {
    case HIT_RESERVED:
      m_pending_hit++;
    case HIT:
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
      break;
    case MISS:
      m_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      if (m_config.m_alloc_policy == ON_MISS) {
        if (m_lines[idx]->is_modified_line()) {
          wb = true;
          // m_lines[idx]->set_byte_mask(mf);
          evicted.set_info(m_lines[idx]->m_block_addr,
                           m_lines[idx]->get_modified_size(),
                           m_lines[idx]->get_dirty_byte_mask(),
                           m_lines[idx]->get_dirty_sector_mask());
          evicted.set_stream_id(m_lines[idx]->m_stream_id);
          m_dirty--;
        }

        if (m_config.is_fill_entire_line() && false) {
          m_lines[idx]->allocate_all_sectors(m_config.tag(addr), m_config.block_addr(addr),
                               time, mf->get_access_sector_mask());
        } else {
          m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               time, mf->get_access_sector_mask());
        }


        if (m_lines[idx]->is_valid_line() || m_lines[idx]->is_modified_line()) {
          m_evictions++;
          // If the line was never used after allocation (no hits), mark as no-reuse
          if (m_lines[idx]->get_last_access_time() == m_lines[idx]->get_alloc_time()) {
            m_evictions_no_reuse++;
          }
        }
        
        // Set stream ID for the allocated line
        if (mf) {
          //m_lines[idx]->m_stream_id = mf->get_streamID();
          m_lines[idx]->set_stream_id(mf->get_streamID(),time);
          m_lines[idx]->set_sid(mf,time);
        }
      }
      break;
    case SECTOR_MISS:
      assert(m_config.m_cache_type == SECTOR);
      m_sector_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      if (m_config.m_alloc_policy == ON_MISS) {
        bool before = m_lines[idx]->is_modified_line();
        ((sector_cache_block *)m_lines[idx])
            ->allocate_sector(time, mf->get_access_sector_mask());

        // Set stream ID for the allocated line
        if (mf) {
          //m_lines[idx]->m_stream_id = mf->get_streamID();
          //m_lines[idx]->set_stream_id(mf->get_streamID(),time);
          m_lines[idx]->set_stream_id(mf,time);
          m_lines[idx]->set_sid(mf,time);
        }
        if (before && !m_lines[idx]->is_modified_line()) {
          m_dirty--;
        }
      }
      break;
    case RESERVATION_FAIL:
      m_res_fail++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      break;
    default:
      fprintf(stderr,
              "tag_array::access - Error: Unknown"
              "cache_request_status %d\n",
              status);
      abort();
  }
  return status;
}

void tag_array::fill(new_addr_type addr, unsigned time, mem_fetch *mf,
                     bool is_write) {
  fill(addr, time, mf->get_access_sector_mask(), mf->get_access_byte_mask(),
       is_write, mf->get_streamID(), mf->get_sid());
}

void tag_array::fill(new_addr_type addr, unsigned time,
                     mem_access_sector_mask_t mask,
                     mem_access_byte_mask_t byte_mask, bool is_write,
                    unsigned long long m_stream_id, unsigned m_sid) {
  // assert( m_config.m_alloc_policy == ON_FILL );
  unsigned idx;
  enum cache_request_status status = probe(addr, idx, mask, is_write, m_sid);

  if (status == RESERVATION_FAIL) {
    return;
  }

  bool before = m_lines[idx]->is_modified_line();
  // assert(status==MISS||status==SECTOR_MISS); // MSHR should have prevented
  // redundant memory request
  if (status == MISS) {
      if (m_config.is_fill_entire_line() && false) {
        m_lines[idx]->allocate_all_sectors(m_config.tag(addr), m_config.block_addr(addr),
                               time, mask);
      } else {
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr), time,
                             mask);
    }
  } else if (status == SECTOR_MISS) {
    assert(m_config.m_cache_type == SECTOR);
    ((sector_cache_block *)m_lines[idx])->allocate_sector(time, mask);
  }
  // Set stream ID for the allocated line
  if (before && !m_lines[idx]->is_modified_line()) {
    m_dirty--;
  }
  before = m_lines[idx]->is_modified_line();

  if (m_config.is_fill_entire_line() && false) {
    m_lines[idx]->fill_all_sectors(time, mask, byte_mask);
  } else {
    m_lines[idx]->fill(time, mask, byte_mask);
  }

  if (m_lines[idx]->is_modified_line() && !before) {
    m_dirty++;
  }

  //m_lines[idx]->m_stream_id = m_stream_id;
  m_lines[idx]->set_stream_id(m_stream_id,time);
}

void tag_array::fill(unsigned index, unsigned time, mem_fetch *mf) {
  assert(m_config.m_alloc_policy == ON_MISS);
  bool before = m_lines[index]->is_modified_line();
  if (m_config.is_fill_entire_line() && false) {
    m_lines[index]->fill_all_sectors(time, mf->get_access_sector_mask(),
                       mf->get_access_byte_mask());
  } else {
    m_lines[index]->fill(time, mf->get_access_sector_mask(),
                       mf->get_access_byte_mask());
  }

  if (m_lines[index]->is_modified_line() && !before) {
    m_dirty++;
  }
  //m_lines[index]->m_stream_id = mf->get_streamID();
  m_lines[index]->set_stream_id(mf->get_streamID(),time);
  m_lines[index]->set_sid(mf,time);
}

// TODO: we need write back the flushed data to the upper level
void tag_array::flush() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    if (m_lines[i]->is_modified_line()) {
      for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++) {
        m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));
      }
    }

  m_dirty = 0;
  is_used = false;
}

void tag_array::invalidate() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++)
      m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));

  m_dirty = 0;
  is_used = false;
}

float tag_array::windowed_miss_rate() const {
  unsigned n_access = m_access - m_prev_snapshot_access;
  unsigned n_miss = (m_miss + m_sector_miss) - m_prev_snapshot_miss;
  // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

  float missrate = 0.0f;
  if (n_access != 0) missrate = (float)(n_miss + m_sector_miss) / n_access;
  return missrate;
}

void tag_array::new_window() {
  m_prev_snapshot_access = m_access;
  m_prev_snapshot_miss = m_miss;
  m_prev_snapshot_miss = m_miss + m_sector_miss;
  m_prev_snapshot_pending_hit = m_pending_hit;

  // Take eviction snapshots
  m_prev_snapshot_evictions = m_evictions;
  m_prev_snapshot_evictions_no_reuse = m_evictions_no_reuse;
}

void tag_array::print(FILE *stream, unsigned &total_access,
                      unsigned &total_misses) const {
  m_config.print(stream);
  fprintf(stream,
          "\t\tAccess = %d, Miss = %d, Sector_Miss = %d, Total_Miss = %d "
          "(%.3g), PendingHit = %d (%.3g), Evictions = %d (%.3g), Evictions_No_Reuse = %d (%.3g)\n",
          m_access, m_miss, m_sector_miss, (m_miss + m_sector_miss),
          (float)(m_miss + m_sector_miss) / m_access, m_pending_hit,
          (float)m_pending_hit / m_access, m_evictions,
          (float)m_evictions / m_access, m_evictions_no_reuse,
          (float)m_evictions_no_reuse / m_access);
  total_misses += (m_miss + m_sector_miss);
  total_access += m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses,
                          unsigned &total_hit_res,
                          unsigned &total_res_fail) const {
  // Update statistics from the tag array
  total_access = m_access;
  total_misses = (m_miss + m_sector_miss);
  total_hit_res = m_pending_hit;
  total_res_fail = m_res_fail;
}



void tag_array::enable_stream_partitioning() {
  m_stream_partitioning_enabled = true;
}

void tag_array::enable_fill_all_sectors() {
  m_fill_entire_line = true;
}
  
void tag_array::set_stream_allocation(unsigned long long streamID, float percentage) {
  if (percentage >= 0.0f && percentage <= 1.0f) {
    m_stream_allocations[streamID] = percentage;
    if (streamID == 0) {
      stream0_wayNum = (unsigned)(m_config.m_assoc * percentage);
      stream1_wayNum = m_config.m_assoc - stream0_wayNum;
    } else if (streamID == 1) {
      stream1_wayNum = (unsigned)(m_config.m_assoc * percentage);
      stream0_wayNum = m_config.m_assoc - stream1_wayNum;
    }
  }
}
  
float tag_array::get_stream_allocation(unsigned long long streamID) const {
  auto it = m_stream_allocations.find(streamID);
  return (it != m_stream_allocations.end()) ? it->second : 0.0f;
}

bool tag_array::is_stream_partitioning_enabled() const {
  return m_stream_partitioning_enabled;
}


unsigned tag_array::count_stream_ways_in_set(unsigned set_index, unsigned long long streamID) const {
  unsigned count = 0;
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    if (m_lines[index]->is_valid_line() && m_lines[index]->m_stream_id == streamID) {
      count++;
    }
  }
  return count;
}

bool tag_array::stream_exceeds_allocation(unsigned set_index, unsigned long long streamID) const {
  if (!m_stream_partitioning_enabled) {
    return false;
  }
  unsigned max_ways = streamID == 0 ? stream0_wayNum : stream1_wayNum;
  unsigned current_ways = 0;
  
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    if ((m_lines[index]->is_valid_line() || m_lines[index]->is_reserved_line())
      && m_lines[index]->m_stream_id == streamID) {
      current_ways++;
      if (current_ways >= max_ways) {
        // If we already exceed the maximum ways, no need to continue checking
        return true;
      }
    }
  }
  
  return false;
}

bool was_write_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_REQUEST_SENT) return true;
  }
  return false;
}

bool was_writeback_sent(const std::list<cache_event> &events,
                        cache_event &wb_event) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_BACK_REQUEST_SENT) {
      wb_event = *e;
      return true;
    }
  }
  return false;
}

bool was_read_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == READ_REQUEST_SENT) return true;
  }
  return false;
}

bool was_writeallocate_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_ALLOCATE_SENT) return true;
  }
  return false;
}
/****************************************************************** MSHR
 * ******************************************************************/

/// Checks if there is a pending request to the lower memory level already
bool mshr_table::probe(new_addr_type block_addr) const {
  table::const_iterator a = m_data.find(block_addr);
  return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
bool mshr_table::full(new_addr_type block_addr) const {
  table::const_iterator i = m_data.find(block_addr);
  if (i != m_data.end())
    return i->second.m_list.size() >= m_max_merged;
  else
    return m_data.size() >= m_num_entries;
}

/// Add or merge this access
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf) {
  m_data[block_addr].m_list.push_back(mf);
  assert(m_data.size() <= m_num_entries);
  assert(m_data[block_addr].m_list.size() <= m_max_merged);
  // indicate that this MSHR entry contains an atomic operation
  if (mf->isatomic()) {
    m_data[block_addr].m_has_atomic = true;
  }
}

/// check is_read_after_write_pending
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
  std::list<mem_fetch *> my_list = m_data[block_addr].m_list;
  bool write_found = false;
  for (std::list<mem_fetch *>::iterator it = my_list.begin();
       it != my_list.end(); ++it) {
    if ((*it)->is_write())  // Pending Write Request
      write_found = true;
    else if (write_found)  // Pending Read Request and we found previous Write
      return true;
  }

  return false;
}

/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
  assert(!busy());
  table::iterator a = m_data.find(block_addr);
  assert(a != m_data.end());
  m_current_response.push_back(block_addr);
  has_atomic = a->second.m_has_atomic;
  assert(m_current_response.size() <= m_data.size());
}

/// Returns next ready access
mem_fetch *mshr_table::next_access() {
  assert(access_ready());
  new_addr_type block_addr = m_current_response.front();
  assert(!m_data[block_addr].m_list.empty());
  mem_fetch *result = m_data[block_addr].m_list.front();
  m_data[block_addr].m_list.pop_front();
  if (m_data[block_addr].m_list.empty()) {
    // release entry
    m_data.erase(block_addr);
    m_current_response.pop_front();
  }
  return result;
}

/// Returns next ready access
mem_fetch *mshr_table::peek_next_access() {
  assert(access_ready());
  new_addr_type block_addr = m_current_response.front();
  assert(!m_data[block_addr].m_list.empty());
  mem_fetch *result = m_data[block_addr].m_list.front();
  return result;
}

void mshr_table::display(FILE *fp) const {
  fprintf(fp, "MSHR contents\n");
  for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e) {
    unsigned block_addr = e->first;
    fprintf(fp, "MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr,
            e->second.m_has_atomic, e->second.m_list.size());
    if (!e->second.m_list.empty()) {
      mem_fetch *mf = e->second.m_list.front();
      fprintf(fp, "%p :", mf);
      mf->print(fp);
    } else {
      fprintf(fp, " no memory requests???\n");
    }
  }
}
/***************************************************************** Caches
 * *****************************************************************/
cache_stats::cache_stats() {
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear() {
  ///
  /// Zero out all current cache statistics
  ///
  m_stats.clear();
  m_stats_pw.clear();
  m_fail_stats.clear();

  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear_pw() {
  ///
  /// Zero out per-window cache statistics
  ///
  m_stats_pw.clear();
}

void cache_stats::inc_stats(int access_type, int access_outcome,
                            unsigned long long streamID) {
  ///
  /// Increment the stat corresponding to (access_type, access_outcome) by 1.
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  if (m_stats.find(streamID) == m_stats.end()) {
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_stats.insert(std::pair<unsigned long long,
                             std::vector<std::vector<unsigned long long>>>(
        streamID, new_val));
  }
  m_stats.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::inc_stats_pw(int access_type, int access_outcome,
                               unsigned long long streamID) {
  ///
  /// Increment the corresponding per-window cache stat
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_stats_pw.insert(std::pair<unsigned long long,
                                std::vector<std::vector<unsigned long long>>>(
        streamID, new_val));
  }
  m_stats_pw.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::inc_fail_stats(int access_type, int fail_outcome,
                                 unsigned long long streamID) {
  if (!check_fail_valid(access_type, fail_outcome))
    assert(0 && "Unknown cache access type or access fail");

  if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_RESERVATION_FAIL_STATUS, 0);
    }
    m_fail_stats.insert(std::pair<unsigned long long,
                                  std::vector<std::vector<unsigned long long>>>(
        streamID, new_val));
  }
  m_fail_stats.at(streamID)[access_type][fail_outcome]++;
}

void cache_stats::inc_eviction_stats(int access_type, unsigned long long streamID) {
  if (m_stats.find(streamID) == m_stats.end()) {
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_stats.insert(std::pair<unsigned long long,
                             std::vector<std::vector<unsigned long long>>>(
        streamID, new_val));
  }
  m_stats.at(streamID)[access_type][EVICTION]++;
}

enum cache_request_status cache_stats::select_stats_status(
    enum cache_request_status probe, enum cache_request_status access) const {
  ///
  /// This function selects how the cache access outcome should be counted.
  /// HIT_RESERVED is considered as a MISS in the cores, however, it should be
  /// counted as a HIT_RESERVED in the caches.
  ///
  if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
    return probe;
  else if (probe == SECTOR_MISS && access == MISS)
    return probe;
  else
    return access;
}

unsigned long long &cache_stats::operator()(int access_type, int access_outcome,
                                            bool fail_outcome,
                                            unsigned long long streamID) {
  ///
  /// Simple method to read/modify the stat corresponding to (access_type,
  /// access_outcome) Used overloaded () to avoid the need for separate
  /// read/write member functions
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats.at(streamID)[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats.at(streamID)[access_type][access_outcome];
  }
}

unsigned long long cache_stats::operator()(int access_type, int access_outcome,
                                           bool fail_outcome,
                                           unsigned long long streamID) const {
  ///
  /// Const accessor into m_stats.
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats.at(streamID)[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats.at(streamID)[access_type][access_outcome];
  }
}

cache_stats cache_stats::operator+(const cache_stats &cs) {
  ///
  /// Overloaded + operator to allow for simple stat accumulation
  ///
  cache_stats ret;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_stats.insert(std::pair<unsigned long long,
                                 std::vector<std::vector<unsigned long long>>>(
        streamID, m_stats.at(streamID)));
  }
  for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_stats_pw.insert(
        std::pair<unsigned long long,
                  std::vector<std::vector<unsigned long long>>>(
            streamID, m_stats_pw.at(streamID)));
  }
  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_fail_stats.insert(
        std::pair<unsigned long long,
                  std::vector<std::vector<unsigned long long>>>(
            streamID, m_fail_stats.at(streamID)));
  }
  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_stats.find(streamID) == ret.m_stats.end()) {
      ret.m_stats.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_stats_pw.find(streamID) == ret.m_stats_pw.end()) {
      ret.m_stats_pw.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_stats_pw.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats_pw.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end();
       ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_fail_stats.find(streamID) == ret.m_fail_stats.end()) {
      ret.m_fail_stats.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_fail_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
             ++status) {
          ret.m_fail_stats.at(streamID)[type][status] +=
              cs(type, status, true, streamID);
        }
      }
    }
  }
  ret.m_cache_port_available_cycles =
      m_cache_port_available_cycles + cs.m_cache_port_available_cycles;
  ret.m_cache_data_port_busy_cycles =
      m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles;
  ret.m_cache_fill_port_busy_cycles =
      m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles;
  return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs) {
  ///
  /// Overloaded += operator to allow for simple stat accumulation
  ///
  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_stats.find(streamID) == m_stats.end()) {
      m_stats.insert(std::pair<unsigned long long,
                               std::vector<std::vector<unsigned long long>>>(
          streamID, cs.m_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          m_stats.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
      m_stats_pw.insert(std::pair<unsigned long long,
                                  std::vector<std::vector<unsigned long long>>>(
          streamID, cs.m_stats_pw.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          m_stats_pw.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end();
       ++iter) {
    unsigned long long streamID = iter->first;
    if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
      m_fail_stats.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_fail_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
             ++status) {
          m_fail_stats.at(streamID)[type][status] +=
              cs(type, status, true, streamID);
        }
      }
    }
  }
  m_cache_port_available_cycles += cs.m_cache_port_available_cycles;
  m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles;
  m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles;
  return *this;
}

void cache_stats::print_stats(FILE *fout, unsigned long long streamID,
                              const char *cache_name) const {
  ///
  /// For a given CUDA stream, print out each non-zero cache statistic for every
  /// memory access type and status "cache_name" defaults to "Cache_stats" when
  /// no argument is provided, otherwise the provided name is used. The printed
  /// format is
  /// "<cache_name>[<request_type>][<request_status>] = <stat_value>"
  /// Specify streamID to be -1 to print every stream.

  std::vector<unsigned> total_access;
  std::string m_cache_name = cache_name;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != -1) && (streamid != streamID)) continue;
    total_access.clear();
    total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
                mem_access_type_str((enum mem_access_type)type),
                cache_request_status_str((enum cache_request_status)status),
                m_stats.at(streamid)[type][status]);

        if (status != RESERVATION_FAIL && status != MSHR_HIT)
          // MSHR_HIT is a special type of SECTOR_MISS
          // so its already included in the SECTOR_MISS
          total_access[type] += m_stats.at(streamid)[type][status];
      }
    }
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      if (total_access[type] > 0)
        fprintf(fout, "\t%s[%s][%s] = %u\n", m_cache_name.c_str(),
                mem_access_type_str((enum mem_access_type)type), "TOTAL_ACCESS",
                total_access[type]);
    }
  }
}

void cache_stats::print_fail_stats(FILE *fout, unsigned long long streamID,
                                   const char *cache_name) const {
  std::string m_cache_name = cache_name;
  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    unsigned long long streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != -1) && (streamid != streamID)) continue;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS;
           ++fail) {
        if (m_fail_stats.at(streamid)[type][fail] > 0) {
          fprintf(
              fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
              mem_access_type_str((enum mem_access_type)type),
              cache_fail_status_str((enum cache_reservation_fail_reason)fail),
              m_fail_stats.at(streamid)[type][fail]);
        }
      }
    }
  }
}

void cache_sub_stats::print_port_stats(FILE *fout,
                                       const char *cache_name) const {
  float data_port_util = 0.0f;
  if (port_available_cycles > 0) {
    data_port_util = (float)data_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util);
  float fill_port_util = 0.0f;
  if (port_available_cycles > 0) {
    fill_port_util = (float)fill_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util);
}

unsigned long long cache_stats::get_stats(
    enum mem_access_type *access_type, unsigned num_access_type,
    enum cache_request_status *access_status,
    unsigned num_access_status) const {
  ///
  /// Returns a sum of the stats corresponding to each "access_type" and
  /// "access_status" pair. "access_type" is an array of "num_access_type"
  /// mem_access_types. "access_status" is an array of "num_access_status"
  /// cache_request_statuses.
  ///
  unsigned long long total = 0;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    for (unsigned type = 0; type < num_access_type; ++type) {
      for (unsigned status = 0; status < num_access_status; ++status) {
        if (!check_valid((int)access_type[type], (int)access_status[status]))
          assert(0 && "Unknown cache access type or access outcome");
        total += m_stats.at(streamID)[access_type[type]][access_status[status]];
      }
    }
  }
  return total;
}

void cache_stats::get_sub_stats(struct cache_sub_stats &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats t_css;
  t_css.clear();

  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        if (status == HIT || status == MISS || status == SECTOR_MISS ||
            status == HIT_RESERVED)
          t_css.accesses += m_stats.at(streamID)[type][status];

        if (status == MISS || status == SECTOR_MISS)
          t_css.misses += m_stats.at(streamID)[type][status];

        if (status == HIT_RESERVED)
          t_css.pending_hits += m_stats.at(streamID)[type][status];

        if (status == RESERVATION_FAIL)
          t_css.res_fails += m_stats.at(streamID)[type][status];

        if (status == EVICTION)
          t_css.evictions += m_stats.at(streamID)[type][status];
      }
    }
  }

  t_css.port_available_cycles = m_cache_port_available_cycles;
  t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles;
  t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles;

  css = t_css;
}

void cache_stats::get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats_pw t_css;
  t_css.clear();

  for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        if (status == HIT || status == MISS || status == SECTOR_MISS ||
            status == HIT_RESERVED)
          t_css.accesses += m_stats_pw.at(streamID)[type][status];

        if (status == HIT) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_hits += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_hits += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == MISS || status == SECTOR_MISS) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_misses += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_misses += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == HIT_RESERVED) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_pending_hits += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_pending_hits += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == EVICTION) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_evictions += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_evictions += m_stats_pw.at(streamID)[type][status];
          }
        }
        
        if (status == RESERVATION_FAIL) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_res_fails += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_res_fails += m_stats_pw.at(streamID)[type][status];
          }
        }
      }
    }
  }

  css = t_css;
}

bool cache_stats::check_valid(int type, int status) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) &&
      (status < NUM_CACHE_REQUEST_STATUS))
    return true;
  else
    return false;
}

bool cache_stats::check_fail_valid(int type, int fail) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (fail >= 0) &&
      (fail < NUM_CACHE_RESERVATION_FAIL_STATUS))
    return true;
  else
    return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy,
                                            bool fill_port_busy) {
  m_cache_port_available_cycles += 1;
  if (data_port_busy) {
    m_cache_data_port_busy_cycles += 1;
  }
  if (fill_port_busy) {
    m_cache_fill_port_busy_cycles += 1;
  }
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config)
    : m_config(config) {
  m_data_port_occupied_cycles = 0;
  m_fill_port_occupied_cycles = 0;
}

/// use the data port based on the outcome and events generated by the mem_fetch
/// request
void baseline_cache::bandwidth_management::use_data_port(
    mem_fetch *mf, enum cache_request_status outcome,
    const std::list<cache_event> &events) {
  unsigned data_size = mf->get_data_size();
  unsigned port_width = m_config.m_data_port_width;
  switch (outcome) {
    case HIT: {
      unsigned data_cycles =
          data_size / port_width + ((data_size % port_width > 0) ? 1 : 0);
      m_data_port_occupied_cycles += data_cycles;
    } break;
    case HIT_RESERVED:
    case MISS: {
      // the data array is accessed to read out the entire line for write-back
      // in case of sector cache we need to write bank only the modified sectors
      cache_event ev(WRITE_BACK_REQUEST_SENT);
      if (was_writeback_sent(events, ev)) {
        unsigned data_cycles = ev.m_evicted_block.m_modified_size / port_width;
        m_data_port_occupied_cycles += data_cycles;
      }
    } break;
    case SECTOR_MISS:
    case RESERVATION_FAIL:
      // Does not consume any port bandwidth
      break;
    default:
      assert(0);
      break;
  }
}

/// use the fill port
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf) {
  // assume filling the entire line with the returned request
  unsigned fill_cycles = m_config.get_atom_sz() / m_config.m_data_port_width;
  m_fill_port_occupied_cycles += fill_cycles;
}

/// called every cache cycle to free up the ports
void baseline_cache::bandwidth_management::replenish_port_bandwidth() {
  if (m_data_port_occupied_cycles > 0) {
    m_data_port_occupied_cycles -= 1;
  }
  assert(m_data_port_occupied_cycles >= 0);

  if (m_fill_port_occupied_cycles > 0) {
    m_fill_port_occupied_cycles -= 1;
  }
  assert(m_fill_port_occupied_cycles >= 0);
}

/// query for data port availability
bool baseline_cache::bandwidth_management::data_port_free() const {
  return (m_data_port_occupied_cycles == 0);
}

/// query for fill port availability
bool baseline_cache::bandwidth_management::fill_port_free() const {
  return (m_fill_port_occupied_cycles == 0);
}

/// Sends next request to lower level of memory
void baseline_cache::cycle() {
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();
    if (!m_memport->full(mf->size(), mf->get_is_write())) {
      m_miss_queue.pop_front();
      m_memport->push(mf);
      //printf("%s push mf: %p, data_size: %d, addr: %lx, dynamic_fetch_mode=%d\n",
      //   get_name().c_str(), mf, mf->get_data_size(), mf->get_addr(), mf->get_dynamic_fetch_mode());
      //mf->print(stdout, false);
    }
  }
  bool data_port_busy = !m_bandwidth_management.data_port_free();
  bool fill_port_busy = !m_bandwidth_management.fill_port_free();
  m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
  m_bandwidth_management.replenish_port_bandwidth();
}

/// Interface for response from lower memory level (model bandwidth restictions
/// in caller)
void baseline_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_ASSOC) {
    assert(mf->get_original_mf());
    extra_mf_fields_lookup::iterator e_org =
        m_extra_mf_fields.find(mf->get_original_mf());
      //print out mf detailed information
    /*
    printf("%s org mf: %p, data_size: %d, addr: %lx, dynamic_fetch_mode=%d pending_read=%d fill\n",
    get_name().c_str(), mf->get_original_mf(),
    mf->get_original_mf()->get_data_size(),
    mf->get_original_mf()->get_addr(),
    mf->get_original_mf()->get_dynamic_fetch_mode(),
    e_org->second.pending_read);
    mf->get_original_mf()->print(stdout, false);
    assert(e_org != m_extra_mf_fields.end());
    */
    e_org->second.pending_read--;

    if (e_org->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }

  //print out mf detailed information
  //printf("%s mf: %p, data_size: %d, addr: %lx, dynamic_fetch_mode=%d base cache fill\n",
  //       get_name().c_str(), mf, mf->get_data_size(), mf->get_addr(), mf->get_dynamic_fetch_mode());
  //if (mf->get_original_mf() != nullptr) {
  //  mf->get_original_mf()->print(stdout, false);
  //}
  //mf->print(stdout, false);

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  //if (m_config.m_dynamic_fetch_mem && mf->get_original_mf() != nullptr) {
  //  e = m_extra_mf_fields.find(mf->get_original_mf());
  //} else {
  //  e = m_extra_mf_fields.find(mf);
  //}
  if (e == m_extra_mf_fields.end() || !e->second.m_valid) {
    printf("error: e == m_extra_mf_fields.end() || !e->second.m_valid\n");
    assert(0);
  }
  mf->set_data_size(e->second.m_data_size);
  mf->set_addr(e->second.m_addr);
  //printf("mf: %p, data_size: %d, addr: %lx, dynamic_fetch_mode=%d\n",
  //       mf, mf->get_data_size(), mf->get_addr(), mf->get_dynamic_fetch_mode());
  if (m_config.m_alloc_policy == ON_MISS)
    m_tag_array->fill(e->second.m_cache_index, time, mf);
  else if (m_config.m_alloc_policy == ON_FILL) {
    m_tag_array->fill(e->second.m_block_addr, time, mf, mf->is_write());
  } else
    abort();
  bool has_atomic = false;
  m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
  if (has_atomic) {
    assert(m_config.m_alloc_policy == ON_MISS);
    cache_block_t *block = m_tag_array->get_block(e->second.m_cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as dirty for
                                                      // atomic operation
    block->set_byte_mask(mf);
  }
  m_extra_mf_fields.erase(mf);
  // print out mf detailed information
  //printf("%s mf: %p, data_size/original_data_size: %d/%d, addr/original_addr: %lx/%lx, dynamic_fetch_mode=%d erase from m_extra_mf_fields\n",
  //       get_name().c_str(), mf, mf->get_data_size(), mf->get_original_data_size(), mf->get_addr(), mf->get_original_addr(), mf->get_dynamic_fetch_mode());
  //mf->print(stdout, false);
  m_bandwidth_management.use_fill_port(mf);
}

/// Checks if mf is waiting to be filled by lower memory level
bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses,
                           unsigned &misses) const {
  fprintf(fp, "Cache %s:\t", m_name.c_str());
  m_tag_array->print(fp, accesses, misses);
}

void baseline_cache::display_state(FILE *fp) const {
  fprintf(fp, "Cache %s:\n", m_name.c_str());
  m_mshrs.display(fp);
  fprintf(fp, "\n");
}

void baseline_cache::inc_aggregated_stats(cache_request_status status,
                                          cache_request_status cache_status,
                                          mem_fetch *mf,
                                          enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    m_gpu->aggregated_l1_stats.inc_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    m_gpu->aggregated_l2_stats.inc_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

void baseline_cache::inc_aggregated_fail_stats(
    cache_request_status status, cache_request_status cache_status,
    mem_fetch *mf, enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    m_gpu->aggregated_l1_stats.inc_fail_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    m_gpu->aggregated_l2_stats.inc_fail_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

void baseline_cache::inc_aggregated_stats_pw(cache_request_status status,
                                             cache_request_status cache_status,
                                             mem_fetch *mf,
                                             enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    m_gpu->aggregated_l1_stats.inc_stats_pw(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    m_gpu->aggregated_l2_stats.inc_stats_pw(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

/// Read miss handler without writeback
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  bool wb = false;
  evicted_block_info e;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e,
                    events, read_only, wa);
}

/// Read miss handler. Check MSHR hit or MSHR available
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss, bool &wb,
                                       evicted_block_info &evicted,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (mshr_hit && mshr_avail) {
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    m_mshrs.add(mshr_addr, mf);
    //printf("mshr_hit && mshr_avail mf: %p, data_size: %d, addr: %lx, dynamic_fetch_mode=%d\n",
    //     mf, mf->get_data_size(), mf->get_addr(), mf->get_dynamic_fetch_mode());
    //mf->print(stdout, false);
    m_stats.inc_stats(mf->get_access_type(), MSHR_HIT, mf->get_streamID());
    do_miss = true;

  } else if (!mshr_hit && mshr_avail &&
             (m_miss_queue.size() < m_config.m_miss_queue_size)) {
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    m_mshrs.add(mshr_addr, mf);
    m_extra_mf_fields[mf] = extra_mf_fields(
        mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    /*
    mem_fetch *temp = nullptr;
    temp = mf->get_original_mf();
    if (temp!=nullptr) {
      m_extra_mf_fields[temp] = extra_mf_fields(
        mshr_addr, temp->get_addr(), cache_index, temp->get_data_size(), m_config);
      // print out mf detailed information
      printf("%s org mf: %p, data_size: %d, addr: %lx, dynamic_fetch_mode=%d, cache_index: %d, mshr_addr: %lx into m_extra_mf_fields\n",
        get_name().c_str(), temp, temp->get_data_size(), temp->get_addr(), temp->get_dynamic_fetch_mode(), cache_index, mshr_addr);
      temp->print(stdout, false);
    } else {
      m_extra_mf_fields[mf] = extra_mf_fields(
      mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
      // print out mf detailed information
      // use mf->print to print out the mf detailed information
      printf("%s mf: %p, data_size/original_data_size: %d/%d, addr/original_addr: %lx/%lx, dynamic_fetch_mode=%d, cache_index: %d, mshr_addr: %lx into m_extra_mf_fields\n",
      get_name().c_str(), mf, mf->get_data_size(), mf->get_original_data_size(), mf->get_addr(), mf->get_original_addr(), mf->get_dynamic_fetch_mode(), cache_index, mshr_addr);
      mf->print(stdout, false);
    }
    */

    mf->set_data_size(m_config.get_atom_sz());
    mf->set_addr(mshr_addr);
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status, time);
    if (!wa) events.push_back(cache_event(READ_REQUEST_SENT));

    do_miss = true;
  } else if (mshr_hit && !mshr_avail)
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                           mf->get_streamID());
  else if (!mshr_hit && !mshr_avail)
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                           mf->get_streamID());
  else
    assert(0);
}

/// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                    unsigned time,
                                    std::list<cache_event> &events) {
  events.push_back(request);
  m_miss_queue.push_back(mf);
  mf->set_status(m_miss_queue_status, time);
}

void data_cache::update_m_readable(mem_fetch *mf, unsigned cache_index) {
  cache_block_t *block = m_tag_array->get_block(cache_index);
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
    if (mf->get_access_sector_mask().test(i)) {
      bool all_set = true;
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        // If any bit in the byte mask (within the sector) is not set,
        // the sector is unreadble
        if (!block->get_dirty_byte_mask().test(k)) {
          all_set = false;
          break;
        }
      }
      if (all_set) block->set_m_readable(true, mf->get_access_sector_mask());
    }
  }
}

/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
cache_request_status data_cache::wr_hit_wb(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  return HIT;
}

/// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  // generate a write-through
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate
/// corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // generate a write-through/evict
  cache_block_t *block = m_tag_array->get_block(cache_index);
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  // Invalidate block
  block->set_status(INVALID, mf->get_access_sector_mask());

  return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  bool evict = (mf->get_access_type() ==
                GLOBAL_ACC_W);  // evict a line that hits on global memory write
  if (evict)
    return wr_hit_we(addr, cache_index, mf, time, events,
                     status);  // Write-evict
  else
    return wr_hit_wb(addr, cache_index, mf, time, events,
                     status);  // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
enum cache_request_status data_cache::wr_miss_wa_naive(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (miss_queue_full(2) ||
      (!(mshr_hit && mshr_avail) &&
       !(!mshr_hit && mshr_avail &&
         (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
    // check what is the exactly the failure reason
    if (miss_queue_full(2))
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID());
    else if (mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                             mf->get_streamID());
    else if (!mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                             mf->get_streamID());
    else
      assert(0);

    return RESERVATION_FAIL;
  }

  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  // Tries to send write allocate request, returns true on success and false on
  // failure
  // if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
  //    return RESERVATION_FAIL;

  const mem_access_t *ma =
      new mem_access_t(m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
                       false,  // Now performing a read
                       mf->get_access_warp_mask(), mf->get_access_byte_mask(),
                       mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

  mem_fetch *n_mf = new mem_fetch(
      *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
      mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;

  // Send read request resulting from write miss
  send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                    evicted, events, false, true);

  events.push_back(cache_event(WRITE_ALLOCATE_SENT));

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      assert(status ==
             MISS);  // SECTOR_MISS and HIT_RESERVED should not send write back
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, evicted.m_stream_id);

      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
      // update eviction stats
      m_stats.inc_eviction_stats(mf->get_access_type(), mf->get_streamID());
    }
    return MISS;
  }

  return RESERVATION_FAIL;
}

enum cache_request_status data_cache::wr_miss_wa_fetch_on_write(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // if the request writes to the whole cache line/sector, then, write and set
    // cache line Modified. and no need to send read request to memory or
    // reserve mshr

    if (miss_queue_full(0)) {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID());
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }

    bool wb = false;
    evicted_block_info evicted;

    cache_request_status status =
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
    assert(status != HIT);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
    if (status == HIT_RESERVED)
      block->set_ignore_on_fill(true, mf->get_access_sector_mask());

    if (status != RESERVATION_FAIL) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL, evicted.m_stream_id);
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
        // update eviction stats
        m_stats.inc_eviction_stats(mf->get_access_type(), mf->get_streamID());
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  } else {
    bool mshr_hit = m_mshrs.probe(mshr_addr);
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    if (miss_queue_full(1) ||
        (!(mshr_hit && mshr_avail) &&
         !(!mshr_hit && mshr_avail &&
           (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
      // check what is the exactly the failure reason
      if (miss_queue_full(1))
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                               mf->get_streamID());
      else if (mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                               mf->get_streamID());
      else if (!mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                               mf->get_streamID());
      else
        assert(0);

      return RESERVATION_FAIL;
    }

    // prevent Write - Read - Write in pending mshr
    // allowing another write will override the value of the first write, and
    // the pending read request will read incorrect result from the second write
    if (m_mshrs.probe(mshr_addr) &&
        m_mshrs.is_read_after_write_pending(mshr_addr) && mf->is_write()) {
      // assert(0);
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_RW_PENDING,
                             mf->get_streamID());
      return RESERVATION_FAIL;
    }

    const mem_access_t *ma = new mem_access_t(
        m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
        false,  // Now performing a read
        mf->get_access_warp_mask(), mf->get_access_byte_mask(),
        mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

    mem_fetch *n_mf = new mem_fetch(
        *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
        mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, NULL, mf);

    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    evicted_block_info evicted;
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true);

    cache_block_t *block = m_tag_array->get_block(cache_index);
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    if (do_miss) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL, evicted.m_stream_id);
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
        // update eviction stats
        m_stats.inc_eviction_stats(mf->get_access_type(), mf->get_streamID());
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  }
}

enum cache_request_status data_cache::wr_miss_wa_lazy_fetch_on_read(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);

  // if the request writes to the whole cache line/sector, then, write and set
  // cache line Modified. and no need to send read request to memory or reserve
  // mshr

  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  if (m_config.m_write_policy == WRITE_THROUGH) {
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  }

  bool wb = false;
  evicted_block_info evicted;

  cache_request_status m_status =
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
  assert(m_status != HIT);
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  if (m_status == HIT_RESERVED) {
    block->set_ignore_on_fill(true, mf->get_access_sector_mask());
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);
  }

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    block->set_m_readable(true, mf->get_access_sector_mask());
  } else {
    block->set_m_readable(false, mf->get_access_sector_mask());
    if (m_status == HIT_RESERVED)
      block->set_readable_on_fill(true, mf->get_access_sector_mask());
  }
  update_m_readable(mf, cache_index);

  if (m_status != RESERVATION_FAIL) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, evicted.m_stream_id);
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
      // update eviction stats
      m_stats.inc_eviction_stats(mf->get_access_type(), mf->get_streamID());
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // on miss, generate write through (no write buffering -- too many threads for
  // that)
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status data_cache::rd_hit_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);
  // Atomics treated as global read/write requests - Perform read, mark line as
  // MODIFIED
  if (mf->isatomic()) {
    assert(mf->get_access_type() == GLOBAL_ACC_R);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as
    block->set_byte_mask(mf);
  }
  return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  if (miss_queue_full(1)) {
    // cannot handle request this cycle
    // (might need to generate two requests)
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, false);

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, evicted.m_stream_id);
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
      // update eviction stats
      m_stats.inc_eviction_stats(mf->get_access_type(), mf->get_streamID());
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
enum cache_request_status read_only_cache::access(
    new_addr_type addr, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  //assert(mf->get_data_size() <= m_config.get_atom_sz());
  assert(m_config.m_write_policy == READ_ONLY);
  assert(!mf->get_is_write());
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), mf->get_sid());
  enum cache_request_status cache_status = RESERVATION_FAIL;

  if (status == HIT) {
    cache_status = m_tag_array->access(block_addr, time, cache_index,
                                       mf);  // update LRU state
  } else if (status != RESERVATION_FAIL) {
    if (!miss_queue_full(0)) {
      bool do_miss = false;
      send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                        events, true, false);
      if (do_miss)
        cache_status = MISS;
      else
        cache_status = RESERVATION_FAIL;
    } else {
      cache_status = RESERVATION_FAIL;
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID());
    }
  } else {
    m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                           mf->get_streamID());
  }

  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status),
                    mf->get_streamID());
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status),
                       mf->get_streamID());
  return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
enum cache_request_status data_cache::process_tag_probe(
    bool wr, enum cache_request_status probe_status, new_addr_type addr,
    unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
  // data_cache constructor to reflect the corresponding cache configuration
  // options. Function pointers were used to avoid many long conditional
  // branches resulting from many cache configuration options.
  cache_request_status access_status = probe_status;
  if (wr) {  // Write
    if (probe_status == HIT) {
      access_status =
          (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if ((probe_status != RESERVATION_FAIL) ||
               (probe_status == RESERVATION_FAIL &&
                m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {
      access_status =
          (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                             mf->get_streamID());
    }
  } else {  // Read
    if (probe_status == HIT) {
      access_status =
          (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if (probe_status != RESERVATION_FAIL) {
      access_status =
          (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                             mf->get_streamID());
    }
  }

  m_bandwidth_management.use_data_port(mf, access_status, events);
  return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events) {
  //assert(mf->get_data_size() <= m_config.get_atom_sz());
  bool wr = mf->get_is_write();
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);
  enum cache_request_status access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status),
                    mf->get_streamID());
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(probe_status, access_status),
                       mf->get_streamID());
  // print out address, cache_index, and access_status
  //printf("Cache %s address: %llx, cache_index: %d, probe_status: %d access_status: %d\n",
  //get_name().c_str(), addr, cache_index, probe_status, access_status);
  return access_status;
}

bool data_cache::stream_reserved_exceeds_allocation(new_addr_type addr, unsigned long long streamID) const {
  return m_tag_array->stream_reserved_exceeds_allocation(addr, streamID);
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
enum cache_request_status l1_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status l2_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access(new_addr_type addr, mem_fetch *mf,
                                            unsigned time,
                                            std::list<cache_event> &events) {
  if (m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full())
    return RESERVATION_FAIL;
  assert(mf->get_data_size() <= m_config.get_line_sz());

  // at this point, we will accept the request : access tags and immediately
  // allocate line
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status =
      m_tags.access(block_addr, time, cache_index, mf);
  enum cache_request_status cache_status = RESERVATION_FAIL;
  assert(status != RESERVATION_FAIL);
  assert(status != HIT_RESERVED);  // as far as tags are concerned: HIT or MISS
  m_fragment_fifo.push(
      fragment_entry(mf, cache_index, status == MISS, mf->get_data_size()));
  if (status == MISS) {
    // we need to send a memory request...
    unsigned rob_index = m_rob.push(rob_entry(cache_index, mf, block_addr));
    m_extra_mf_fields[mf] = extra_mf_fields(rob_index, m_config);
    // print out mf detailed information
    //printf("tex cache mf: %p, data_size: %d, addr: %lx, dynamic_fetch_mode=%d, cache_index: %d, mshr_addr: %lx into m_extra_mf_fields\n",
    //     mf, mf->get_data_size(), mf->get_addr(), mf->get_dynamic_fetch_mode(), cache_index, block_addr);
    mf->set_data_size(m_config.get_line_sz());
    m_tags.fill(cache_index, time, mf);  // mark block as valid
    m_request_fifo.push(mf);
    mf->set_status(m_request_queue_status, time);
    events.push_back(cache_event(READ_REQUEST_SENT));
    cache_status = MISS;
  } else {
    // the value *will* *be* in the cache already
    cache_status = HIT_RESERVED;
  }
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status),
                    mf->get_streamID());
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status),
                       mf->get_streamID());
  return cache_status;
}

void tex_cache::cycle() {
  // send next request to lower level of memory
  // TODO: Use different full() for sst_mem_interface?
  if (!m_request_fifo.empty()) {
    mem_fetch *mf = m_request_fifo.peek();
    if (!m_memport->full(mf->get_ctrl_size(), false)) {
      m_request_fifo.pop();
      m_memport->push(mf);
    }
  }
  // read ready lines from cache
  if (!m_fragment_fifo.empty() && !m_result_fifo.full()) {
    const fragment_entry &e = m_fragment_fifo.peek();
    if (e.m_miss) {
      // check head of reorder buffer to see if data is back from memory
      unsigned rob_index = m_rob.next_pop_index();
      const rob_entry &r = m_rob.peek(rob_index);
      assert(r.m_request == e.m_request);
      // assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr())
      // );
      if (r.m_ready) {
        assert(r.m_index == e.m_cache_index);
        m_cache[r.m_index].m_valid = true;
        m_cache[r.m_index].m_block_addr = r.m_block_addr;
        m_result_fifo.push(e.m_request);
        m_rob.pop();
        m_fragment_fifo.pop();
      }
    } else {
      // hit:
      assert(m_cache[e.m_cache_index].m_valid);
      assert(m_cache[e.m_cache_index].m_block_addr ==
             m_config.block_addr(e.m_request->get_addr()));
      m_result_fifo.push(e.m_request);
      m_fragment_fifo.pop();
    }
  }
}

/// Place returning cache block into reorder buffer
void tex_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_TEX_FIFO) {
    assert(mf->get_original_mf());
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    e->second.pending_read--;

    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }

  // print out mf detailed information
  //printf("mf: %p, data_size: %d, addr: %lx, dynamic_fetch_mode=%d texture cache fill\n",
  //      mf, mf->get_data_size(), mf->get_addr(), mf->get_dynamic_fetch_mode());

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  assert(!m_rob.empty());
  mf->set_status(m_rob_status, time);

  unsigned rob_index = e->second.m_rob_index;
  rob_entry &r = m_rob.peek(rob_index);
  assert(!r.m_ready);
  r.m_ready = true;
  r.m_time = time;
  assert(r.m_block_addr == m_config.block_addr(mf->get_addr()));
}

void tex_cache::display_state(FILE *fp) const {
  fprintf(fp, "%s (texture cache) state:\n", m_name.c_str());
  fprintf(fp, "fragment fifo entries  = %u / %u\n", m_fragment_fifo.size(),
          m_fragment_fifo.capacity());
  fprintf(fp, "reorder buffer entries = %u / %u\n", m_rob.size(),
          m_rob.capacity());
  fprintf(fp, "request fifo entries   = %u / %u\n", m_request_fifo.size(),
          m_request_fifo.capacity());
  if (!m_rob.empty()) fprintf(fp, "reorder buffer contents:\n");
  for (int n = m_rob.size() - 1; n >= 0; n--) {
    unsigned index = (m_rob.next_pop_index() + n) % m_rob.capacity();
    const rob_entry &r = m_rob.peek(index);
    fprintf(fp, "tex rob[%3d] : %s ", index,
            (r.m_ready ? "ready  " : "pending"));
    if (r.m_ready)
      fprintf(fp, "@%6u", r.m_time);
    else
      fprintf(fp, "       ");
    fprintf(fp, "[idx=%4u]", r.m_index);
    r.m_request->print(fp, false);
  }
  if (!m_fragment_fifo.empty()) {
    fprintf(fp, "fragment fifo (oldest) :");
    fragment_entry &f = m_fragment_fifo.peek();
    fprintf(fp, "%s:          ", f.m_miss ? "miss" : "hit ");
    f.m_request->print(fp, false);
  }
}
/******************************************************************************************************************************************/
