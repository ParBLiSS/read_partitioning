#ifndef READ_PARTITION_HPP
#define READ_PARTITION_HPP


#include <map>
#include <string>

#include "mxx/comm.hpp"

namespace dbgp{
    void partitionReads(mxx::comm& comm, std::string inputFile, std::string partitionFile,
                        std::string rpFileName, std::string colorMappingFile,
                        std::string vertexListFile, std::string filteredKmersFile);
};

#endif
