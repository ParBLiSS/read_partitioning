#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP

#include <vector>
#include <string>

#include "mxx/comm.hpp"
#include "mxx/collective.hpp"

    // bug in mxx left_shift in specialization for std::vector
    template <typename T>
    std::vector<T> mxx_left_shift(const std::vector<T>& v, const mxx::comm& comm = mxx::comm())
    {
      // get datatype
      mxx::datatype dt = mxx::get_datatype<T>();

      // TODO: handle tags with MXX (get unique tag function)
      int tag = 15;
      // receive the size first
      std::vector<T> result;
      size_t right_size = mxx::left_shift(v.size(), comm);

      MPI_Request recv_req;
      // if not last processor
      // TODO: replace with comm.send/ comm.recv which automatically will resolve
      // to BIG MPI if message size is too large
      if (comm.rank() < comm.size()-1 && right_size > 0) {
        result.resize(right_size);
        MPI_Irecv(&result[0], right_size, dt.type(), comm.rank()+1, tag,
                  comm, &recv_req);
      }
      // if not first processor
      if (comm.rank() > 0 && v.size() > 0) {
        // send my most right element to the right
        MPI_Send(const_cast<T*>(&v[0]), v.size(), dt.type(), comm.rank()-1, tag, comm);
      }
      if (right_size > 0 && comm.rank() < comm.size()-1) {
        // wait for the async receive to finish
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
      }
      return result;
    }

    template<typename BVT, typename Predicate>
    void shiftStraddlingRegion(const mxx::comm& comm,
                               std::vector<BVT>&  local_rhpairs,
                               std::size_t& start_offset,
                               std::size_t& end_offset,
                               std::vector<BVT>& straddle_region,
                               Predicate bvt_predicate){
      // Assumes that the local_rhpairs has at least one element
      std::vector<BVT> snd_to_left, right_region;
      // find the starting segment of local_rhpairs that straddles
      //  with the processor on the left
      auto lastv = local_rhpairs.back();
      auto prevx = mxx::right_shift(lastv, comm);

      auto fwx_itr = local_rhpairs.begin();
      if(comm.rank() > 0){
        for(;fwx_itr != local_rhpairs.end(); fwx_itr++){
          if(!bvt_predicate(*fwx_itr, prevx))
            break;
        }
      }

      if(fwx_itr != local_rhpairs.begin()){
        auto osize = std::distance(local_rhpairs.begin(), fwx_itr);
        snd_to_left.resize(osize);
        std::copy(local_rhpairs.begin(), fwx_itr, snd_to_left.begin());
      }
      //wrong auto to_snd_size = snd_to_left.size();
      //wrong auto to_rcv_size = mxx::right_shift(to_snd_size, comm);
      // std::cout << snd_to_left.size() << std::endl;
      right_region = mxx_left_shift(snd_to_left, comm);
      start_offset = std::distance(local_rhpairs.begin(), fwx_itr);
      //kishore int soffset = (int)  (start_offset ==  local_rhpairs.size());
      //kishore auto total =  mxx::allreduce(soffset);
      //kishore if(comm.rank() == 0)
      //kishore   std::cout << "Total : "<< total << std::endl;

      // find the ending segment of local_rhpairs that straddles
      //  with the processor on the right
      //  - there will be at least one value
      auto rvx_itr = local_rhpairs.rbegin();
      if(comm.rank() < comm.size() - 1) {
        for(;rvx_itr != local_rhpairs.rend();rvx_itr++){
          if(!bvt_predicate(*rvx_itr, lastv))
            break;
        }
      }
      auto left_region_size = std::distance(local_rhpairs.rbegin(), rvx_itr);
      end_offset = local_rhpairs.size() - left_region_size;
      //std::cout << left_region_size;

      // construct straddling region from left and right region
      straddle_region.resize(left_region_size + right_region.size());
      std::copy(local_rhpairs.begin() + end_offset, local_rhpairs.end(),
                straddle_region.begin());
      std::copy(right_region.begin(), right_region.end(),
                straddle_region.begin() + left_region_size);

    }

    template<typename BVT, typename Predicate>
    void shiftStraddlingRegion(const mxx::comm& comm,
                               std::vector<BVT>& local_rhpairs,
                               std::vector<BVT>& straddle_region,
                               Predicate bvt_predicate){
      // Assumes that the local_rhpairs has at least one element
      std::vector<BVT> snd_to_left;
      // find the starting segment of local_rhpairs that straddles
      // with the processor on the left
      auto lastv = local_rhpairs.back();
      auto prevx = mxx::right_shift(lastv, comm);

      auto fwx_itr = local_rhpairs.begin();
      if(comm.rank() > 0){
        for(;fwx_itr != local_rhpairs.end(); fwx_itr++){
          if(!bvt_predicate(*fwx_itr, prevx))
            break;
        }
      }

      if(fwx_itr != local_rhpairs.begin()){
        auto osize = std::distance(local_rhpairs.begin(), fwx_itr);
        snd_to_left.resize(osize);
        std::copy(local_rhpairs.begin(), fwx_itr, snd_to_left.begin());
      }

      straddle_region = mxx_left_shift(snd_to_left, comm);

    }

#endif
