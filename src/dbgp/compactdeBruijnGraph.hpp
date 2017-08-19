#ifndef _COMPACT_DEBRUIJN_GRAPH_
#define _COMPACT_DEBRUIJN_GRAPH_

#include <vector>
#include <string>
#include <cstdint>

#include "mxx/comm.hpp"

namespace dbgp{
  void compactInit();
  int compactDBG(mxx::comm& comm, std::string ipFileName,
                 uint64_t minKmerFreq, char dbgNodeWtType,
                 std::string outFileName, std::string filteredKmersFile,
                 std::string colorKmerMapFile, std::string colorVertexListFile);

};
#endif
