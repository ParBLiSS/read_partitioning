#ifndef MPI_QUERY_ORACLE_H
#define MPI_QUERY_ORACLE_H

#include <vector>
#include "mxx/comm.hpp"

namespace dbgp{


    template<typename ValueType>
    class VertexIDOracle {
    protected :
        const std::vector<ValueType>& local_data;
        std::vector<ValueType> proc_vert_begin;
        std::vector<std::size_t> proc_vert_sizes;
        const mxx::comm& _comm;
        int rank, nproc;
        ValueType default_value;
        std::size_t m_begin, m_end;
    public:
        typedef ValueType value_type;

        VertexIDOracle(const std::vector<ValueType>& ldat,
                       const std::vector<ValueType>& pvbs,
                       const std::vector<std::size_t>& pvsz,
                       ValueType dvalue = std::numeric_limits<ValueType>::max(),
                       const mxx::comm& cx = mxx::comm()):
            local_data(ldat), proc_vert_begin(pvbs),
            proc_vert_sizes(pvsz), _comm(cx){
            assert(local_data.size() > 0);
            proc_vert_begin.push_back(std::numeric_limits<ValueType>::max());
            default_value = dvalue;
            rank = _comm.rank();
            nproc = _comm.size();
            m_begin = std::accumulate(pvsz.begin(), pvsz.begin() + rank, 0);
            m_end = std::accumulate(pvsz.begin(), pvsz.begin() + rank + 1, 0);
        }

        virtual inline ValueType find(const ValueType& query) const{
            // binary_search
            auto qrx = std::lower_bound(local_data.begin(),
                                        local_data.end(), query);
            if(qrx != local_data.end() && *qrx == query)
                return std::distance(local_data.begin(), qrx) + m_begin;
            else
                return default_value;
        }

        virtual void get(std::vector<ValueType>& queries,
                         std::vector<ValueType>& results) const{
            results.resize(queries.size());
            for(std::size_t i = 0u; i < queries.size(); i++){
                results[i] = find(queries[i]);
            }
        }

        virtual inline std::size_t size() const{
            return local_data.size();
        }

        // the owning process
        virtual inline int owner(const ValueType& x) const{
            // binary search in the starting array
            if(x == default_value)
                return (int)comm_rank();
            auto stx = std::upper_bound(proc_vert_begin.begin(),
                                        proc_vert_begin.end(), x);
            if(stx != local_data.end())
                return ((int)std::distance(proc_vert_begin.begin(), stx)) - 1;
            else
                return comm_size() - 1;
        }

        virtual inline int comm_size() const{
            return nproc;
        }

        virtual inline int comm_rank() const{
            return rank;
        }

        virtual inline const mxx::comm& comm() const{
            return _comm;
        }

        inline bool in_range(const ValueType& sdx) const{
            return (local_data.size() > 0) &&
                (local_data.front() <= sdx) && (sdx <= local_data.back());
        }

    };

};

#endif
