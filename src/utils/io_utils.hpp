#ifndef _IO_UTILS_H
#define _IO_UTILS_H


#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>

#include "mxx/comm.hpp"
#include "mxx/shift.hpp"

template<typename SizeType, typename T>
static inline SizeType block_low(T rank,  T nproc,
                                 SizeType n){
  return (((SizeType)rank) * n) / ((SizeType)nproc);
}

template<typename SizeType, typename T>
static inline SizeType block_high(T rank, T nproc,
                                  SizeType n){
  return ((((SizeType)(rank + 1)) * n) / ((SizeType)nproc)) - 1;
  //return (((rank + 1) * n) / nproc) - 1;
}

template<typename SizeType, typename T>
static inline SizeType block_size(T rank, T nproc,
                                  SizeType n){
  return block_low<SizeType, T>(rank + 1, nproc, n)
    - block_low<SizeType, T>(rank, nproc, n);
}

template<typename SizeType, typename T>
static inline T block_owner(SizeType j, SizeType n,
                            T nproc){
  return (((nproc) * ((j) + 1) - 1) / (n));
}

void compute_offsets(const mxx::comm& comm,
                     std::string inFileName,
                     uint64_t& offsetStart,
                     uint64_t& offsetEnd);

void read_block(const mxx::comm& comm,
                std::string inFileName,
                uint64_t offsetStart,
                uint64_t offsetEnd,
               std::vector<std::string>& dataStore);

void read_block(const mxx::comm& comm,
                std::string inFileName,
                std::vector<std::string>& dataStore);

#endif
