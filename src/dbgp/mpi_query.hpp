
#ifndef MPI_QUERY_H
#define MPI_QUERY_H

#include "mxx/collective.hpp"
#include <vector>

namespace dbgp{
  ///
  // @brief   Returns the displacements vector needed by MPI_Alltoallv.
  //
  // @param counts    The `counts` array needed by MPI_Alltoallv
  //
  // @return The displacements vector needed by MPI_Alltoallv.
  ///
  template<typename OffsetType>
  static inline
  std::vector<OffsetType> get_displacements(const std::vector<OffsetType>& counts)
  {
    // copy and do an exclusive prefix sum
    std::vector<OffsetType> result = counts;
    // excl_prefix_sum2(result.begin(), result.end());
    auto begin = result.begin();
    auto end = result.end();
    // set the total sum to zero
    OffsetType sum = 0;
    OffsetType tmp;
    // calculate the inclusive prefix sum
    while (begin != end) {
      tmp = sum;
      sum += *begin;
      *begin = tmp;
      ++begin;
    }
    return result;
  }

  // A template for local oracle
  //  Local oracle objects are used in the functions
  //   get_query_results and get_query_results_perm
  //  Local oracle objects has to satisfy the following
  // "contract"
  //   Oracle is expected to define the following types
  //     - size_type : type of indices
  //     - value_type : type of data queried
  //   Oracle is expected to define the following fns
  //     - get(queries, results) : local query results
  //     - owner(idx) : owining process of index idx
  //     - size() : size of the local data
  //     - comm() : communicator name
  //     - comm_rank() : rank of this processor within the comm
  //     - comm_size() : size of the communicator
  template<typename ValueType>
  class LocalOracle{
  public:
    typedef ValueType value_type;

    LocalOracle(){}
    virtual void get(std::vector<ValueType>& queries,
                     std::vector<ValueType>& results) const = 0;
    virtual std::size_t size() const = 0;
    virtual int owner(const ValueType& x) const = 0;
    virtual int comm_size() const = 0;
    virtual int comm_rank() const = 0;
    virtual const mxx::comm& comm() const = 0;
    virtual ~LocalOracle(){}
  };


  //
  // @brief   Prepare the data to be sent to answer distributed 'queries'
  //
  // @param oracle   An oracle that can answer local queries
  //   Oracle is expected to define
  //     - size_type : type of indices
  //     - value_type : type of data queried
  //   Oracle is expected to answer
  //     - comm() : communicator targ
  //     - comm_size() : size of the communicator
  //     - get(queries, results) : local query results
  //     - owner(idx) : owining process of index idx
  // @param queries  A local array of indices onto rdata
  //                 whose corresponding rdata entries are desired
  // @return sndcts   query  counts to send to each processor
  // @return sndqry   query values ordered by the processes
  //
  ///
  template<typename OracleType,
           typename T=int,
           typename DataType=typename OracleType::value_type>
  void prepare_query_send(const OracleType& oracle, //IN
                          const std::vector<DataType>& queries, // IN
                          std::vector<T>& sndcts, //OUT
                          std::vector<std::size_t>& sndqry // OUT
                          ){
    sndcts.resize(oracle.comm_size());
    sndqry.resize(queries.size());
    // initialize
    for(auto i = 0; i < sndcts.size(); i++)
      sndcts[i] = 0;

    //  Count the number of queries targeting each processor
    for(auto x: queries){
      int px = oracle.owner(x);
      sndcts[px] += 1;
    }
    // Order the query data to be sent w.r.t the owining processor
    std::vector<T> sndptr = get_displacements(sndcts);
    for(auto i = 0; i < queries.size(); i++){
      int px = oracle.owner(queries[i]);
      sndqry[sndptr[px]] = queries[i];
      sndptr[px] += 1;
    }
  }


  //
  // @brief   Get query results for rdata
  //
  // @param oracle   An oracle that can answer local queries
  //   Oracle is expected to define
  //     - size_type : type of indices
  //     - value_type : type of data queried
  //   Oracle is expected to answer
  //     - comm() : communicator targ
  //     - comm_size() : size of the communicator
  //     - get(queries, results) : local query results
  //     - owner(idx) : owining process of index idx
  //   Oracle can answer all the queries in the range
  //    - block_begin + [0, ..., oracle.size() - 1] locally.
  // @param queries  A local array of indices onto rdata
  //                 whose corresponding rdata entries are desired
  // @return results  Results from rdata for the local queries
  //
  // @return The displacements vector needed by MPI_Alltoallv.
  ///

  template<typename OracleType,
           typename DataType = typename OracleType::value_type>
  void get_query_results(const OracleType& oracle, // IN
                         const std::vector<DataType>& queries, // IN
                         std::vector<DataType>& results  // OUT
                         ){
    int nproc = oracle.comm_size();
    std::vector<std::size_t> sndqry(queries.size());
    std::vector<std::size_t>  sndcts(nproc), rcvcts(nproc);

    // 1. Prepare query data to send:
    prepare_query_send<OracleType, std::size_t>(oracle, queries, sndcts, sndqry);

    // 2. Send/Recieve the targetd queries
    //   2.1 all-to-all to specify how many to send to each proc.
    std::vector<std::size_t> rcvqry;
    rcvcts = mxx::all2all(sndcts, oracle.comm());
    //   2.2 receive queries targeted to this processor
    std::size_t rcvtotal = std::accumulate(rcvcts.begin(), rcvcts.end(), 0);
    rcvqry =  mxx::all2allv(sndqry, sndcts, oracle.comm());
    assert(rcvqry.size() == rcvtotal);
    sndqry.clear();

    // 3. Prepare query results corresponding to the recieved queries
    std::vector<DataType> snddata(rcvtotal);
    oracle.get(rcvqry, snddata);

    // 4. Communicate the results
    rcvcts.swap(sndcts);
    std::vector<DataType> rcvdata(queries.size());
    mxx::all2allv(snddata, sndcts, rcvdata, oracle.comm());

    // 5. Update the results in the input query order
    std::vector<std::size_t> rcvptr = get_displacements(rcvcts);
    results.resize(queries.size());
    for(auto i = 0; i < queries.size(); i++){
      auto x = queries[i];
      int px = oracle.owner(x);
      results[i] = rcvdata[rcvptr[px]];
      rcvptr[px] += 1;
    }
  }
}

#endif
