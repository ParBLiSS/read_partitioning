/*
 * Copyright (c) Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// @file    
// @ingroup 
// @author  
// @brief   
//
// Copyright (c) Georgia Institute of Technology. All Rights Reserved.
//

//Includes
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

//Own includes
#include "utils/mpi_utils.hpp"
#include "dbgp/timer.hpp"

//External includes
#include "extutils/logging.hpp"
#include "extutils/argvparser.hpp"
#include "mxx/reduction.hpp"
#include "mxx/utils.hpp"
#include "mxx/sort.hpp"
#include "mxx/comm.hpp"
#include "index/kmer_hash.hpp"
#include "index/kmer_index.hpp"
#include "debruijn/de_bruijn_node_trait.hpp"
#include "debruijn/de_bruijn_construct_engine.hpp"
#include "debruijn/de_bruijn_nodes_distributed.hpp"
#include "utils/kmer_utils.hpp"

INITIALIZE_EASYLOGGINGPP
using namespace CommandLineProcessing;


struct InputArgs{
  std::string ipFileName;
  std::string ajFilePrefix;
  std::string filteredKmersPrefix; //Stores (low frequency) kmers filtered out
  std::string colorVertexListPrefix; //Stores graph nodes in color space (before transformation to contiguous space)
  uint64_t minKmerFreq;
  char dbgNodeWtType; 

  InputArgs(){
    minKmerFreq = 5;
    dbgNodeWtType = 'c';
  }
};


//use for declarations
namespace dbgp{
    using vertexIdType = uint64_t;
    using weightType = uint32_t;

    static dbgp::vertexIdType NODE_ID_MISSING;
    static dbgp::weightType NODE_WEIGHT_MISSING;

    // init
    // Initialize MISSING vertex and weight static variables
    void init(){
       // The limits of type
       NODE_ID_MISSING = std::numeric_limits<dbgp::vertexIdType>::max();
       NODE_WEIGHT_MISSING = std::numeric_limits<dbgp::weightType>::max();
    }

    // adjacency list record type
    struct adjRecord {
        vertexIdType vertexU;
        vertexIdType vertexV;
        weightType edgeWeight;
        weightType vUWeight;

        adjRecord(vertexIdType ux,vertexIdType vx, weightType ew, weightType nw) {
            set(ux, vx, ew, nw);
        }
        adjRecord(){} // empty constructor for

        void set(vertexIdType ux, vertexIdType vx, weightType ew, weightType nw) {
            vertexU = ux; vertexV = vx; edgeWeight = ew; vUWeight = nw;
        }
        bool operator==(const adjRecord& other){
            return (vertexU == other.vertexU) &&
                (vertexV == other.vertexV);
        }
    };
    using adjRecordType = adjRecord;

    //Kmer size set to 31, and alphabets set to 4 nucleotides
    using Alphabet = bliss::common::DNA;
    using KmerType = bliss::common::Kmer<31, Alphabet, uint64_t>;

    //farm hash parameters
    template <typename KM>
    using CntIndexDistHashF = bliss::kmer::hash::farm<KM, true>;
    template <typename KM>
    using CntIndexStoreHashF = bliss::kmer::hash::farm<KM, false>;
    template <typename Key>
    using CntIndexMapParamsF = bliss::index::kmer::BimoleculeHashMapParams<Key, CntIndexDistHashF, CntIndexStoreHashF>;
    //BLISS internal data structure for storing de bruijn graph
    template <typename EdgeEnc>
    using NodeMapTypeF = typename
        bliss::de_bruijn::de_bruijn_nodes_distributed<
        dbgp::KmerType,
        bliss::de_bruijn::node::edge_counts<EdgeEnc, uint32_t>,
        dbgp::CntIndexMapParamsF>;

    //murmur hash parameters
    template <typename KM>
    using CntIndexDistHashM = bliss::kmer::hash::murmur<KM, true>;
    template <typename KM>
    using CntIndexStoreHashM = bliss::kmer::hash::murmur<KM, false>;
    template <typename Key>
    using CntIndexMapParamsM = bliss::index::kmer::BimoleculeHashMapParams<Key, CntIndexDistHashM, CntIndexStoreHashM>;
    //BLISS internal data structure for storing de bruijn graph
    template <typename EdgeEnc>
    using NodeMapTypeM = typename
        bliss::de_bruijn::de_bruijn_nodes_distributed<
        dbgp::KmerType,
        bliss::de_bruijn::node::edge_counts<EdgeEnc, uint32_t>,
        dbgp::CntIndexMapParamsM>;

    //Parser type, depends on the sequence file format
    //We restrict the usage to FASTA format
    template <typename baseIter>
    using SeqParser = typename bliss::io::FASTAParser<baseIter>;
    
};
MXX_CUSTOM_STRUCT(dbgp::adjRecordType, vertexU, vertexV, edgeWeight, vUWeight);


int parse_args(int argc, char* argv[], mxx::comm& comm,
               InputArgs& in_args){
  ArgvParser cmd;
  const char* ip_arg = "input_file";
  const char* aj_arg = "adj_list_prefix";
  const char* fk_arg = "filtered_kmers_prefix"; //Stores (low frequency) kmers filtered out
  const char* cv_arg = "color_vert_list_prefix"; //Stores graph nodes in color space (before transformation to contiguous space)
  const char* mk_arg = "min_kmer";
  const char* wt_arg = "db_node_wt_type";

  cmd.setIntroductoryDescription("Construct Plain DBG");

  cmd.addErrorCode(0, "Success");
  cmd.addErrorCode(1, "Error");

  cmd.setHelpOption("h", "help", "Print this help page");

  cmd.defineOption(ip_arg, "Input FASTA File",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(ip_arg, "i");

  cmd.defineOption(aj_arg, "Adjacency List Prefix",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(aj_arg, "a");

  cmd.defineOption(fk_arg, "Filtered Kmers File Prefix",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(fk_arg, "f");

  cmd.defineOption(cv_arg, "Color Vertex List File Prefix",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(cv_arg, "v");

  cmd.defineOption(mk_arg,
                   "(k+1)-mers below this frequency will be filtered out. Default is 5",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(mk_arg, "k");

  cmd.defineOption(wt_arg, "de Bruijn graph node weight type. Should be one of 'c' or 's', "
                   "where 'c' stands for count and 's' for sum",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(wt_arg, "w");

  int result = cmd.parse(argc, argv);

  //Make sure we get the right command line args
  if (result != ArgvParser::NoParserError) {
    if (!comm.rank()) std::cout << cmd.parseErrorDescription(result) << "\n";
    return 1;
  }

  if(cmd.foundOption(ip_arg)) {
    in_args.ipFileName = cmd.optionValue(ip_arg);
  } else {
    if(comm.rank() == 0)
      std::cout << "Required option missing: " << ip_arg  << std::endl;
    return 1;
  }

    if(cmd.foundOption(aj_arg)) {
        std::string outPfx = cmd.optionValue(aj_arg);
        std::stringstream outs;
        outs << outPfx << "_"
             << (comm.rank() < 10 ? "000" :
                 (comm.rank() < 100 ? "00" :
                  (comm.rank() < 1000 ? "0" : "")))
             << comm.rank() << ".txt";
        in_args.ajFilePrefix = outs.str();
    } else {
      if(comm.rank() == 0)
        std::cout << "Required option missing: " << aj_arg  << std::endl;
      return 1;
    }

  if(cmd.foundOption(fk_arg)) { //Stores (low frequency) kmers filtered out
      std::string outPfx = cmd.optionValue(fk_arg);
      std::stringstream outs;
      outs << outPfx << "_"
           << (comm.rank() < 10 ? "000" :
               (comm.rank() < 100 ? "00" :
                (comm.rank() < 1000 ? "0" : "")))
           << comm.rank() << ".txt";
      in_args.filteredKmersPrefix = outs.str();
  } else {
//    if(comm.rank() == 0)
//      std::cout << "Required option missing: " << fk_arg  << std::endl;
//    return 1;
  }

  if(cmd.foundOption(cv_arg)) { //Stores graph nodes in color space (before transformation to contiguous space)
      std::string outPfx = cmd.optionValue(cv_arg);
      std::stringstream outs;
      outs << outPfx << "_"
           << (comm.rank() < 10 ? "000" :
               (comm.rank() < 100 ? "00" :
                (comm.rank() < 1000 ? "0" : "")))
           << comm.rank() << ".txt";
      in_args.colorVertexListPrefix = outs.str();
  } else {
    if(comm.rank() == 0)
      std::cout << "Required option missing: " << cv_arg  << std::endl;
    return 1;
  }

  if(cmd.foundOption(mk_arg))
    in_args.minKmerFreq = std::stoi(cmd.optionValue(mk_arg));

    if(cmd.foundOption(wt_arg)) {
      in_args.dbgNodeWtType = cmd.optionValue(wt_arg)[0];
      if(in_args.dbgNodeWtType != 'c' && in_args.dbgNodeWtType != 's'){
        if(comm.rank() == 0)
            std::cout << in_args.dbgNodeWtType << " is invalid option of " << wt_arg
                      << ". Option should be one of 'c' or 's'. " << std::endl;
        return 1;
      }
    }

  LOG_IF(!comm.rank(), INFO) << "--------------------------------------" ;
  LOG_IF(!comm.rank(), INFO) << "Input File     : " << in_args.ipFileName;
  LOG_IF(!comm.rank(), INFO) << "Adj File Prefix: " << in_args.ajFilePrefix;
  LOG_IF(!comm.rank(), INFO) << "Filtered Kmers File Prefix : " << in_args.filteredKmersPrefix;
  LOG_IF(!comm.rank(), INFO) << "Color Vertex File Prefix   : " << in_args.colorVertexListPrefix;
  LOG_IF(!comm.rank(), INFO) << "Min kmer Freq  : " << in_args.minKmerFreq;
  LOG_IF(!comm.rank(), INFO) << "dbg N Wt Type  : " << in_args.dbgNodeWtType;
  LOG_IF(!comm.rank(), INFO) << "--------------------------------------" ;

  return 0;
}


//use for definitions
namespace dbgp{

  namespace graphGen
  {
    /**
     * @class                     dbgp::graphGen::deBruijnGraph
     * @brief                     Builds the edgelist of de Bruijn graph
     * @details                   Sequences are expected in FASTA format
     *                            Restrict the alphabet of DNA to {A,C,G,T}
     */
    class deBruijnGraph
    {
    public:


      void checkEdgeList(std::string &fileName, const mxx::comm &comm) {
        dbgp::Timer timer;
        std::vector<dbgp::adjRecordType> tmpEdgeList;
        std::vector<dbgp::adjRecordType> idxEdgeList;
        std::size_t selfEdgeCount = 0, pairEdgeCount = 0;
        bool isPairEdge;
        
{//Start scope for idx
        //Initialize the map for murmur hash
        bliss::de_bruijn::de_bruijn_engine<dbgp::NodeMapTypeM> idx(comm);

        //Build the de Bruijn graph as distributed map
        idx.template build_posix<dbgp::SeqParser, bliss::io::SequencesIterator>(fileName, comm);
        timer.end_section("finished building dB graph using BLISS and murmur hash");

        auto it = idx.cbegin();

        // Deriving data types of de Bruijn graph storage container
        using mapPairType       = typename std::iterator_traits<decltype(it)>::value_type;
        using constkmerType     = typename std::tuple_element<0, mapPairType>::type;
        using kmerType          = typename std::remove_const<constkmerType>::type;
        using edgeCountInfoType = typename std::tuple_element<1, mapPairType>::type;

        bliss::kmer::transform::lex_less<KmerType> minKmer;

        static_assert(std::is_same<typename kmerType::KmerWordType, uint64_t>::value,
                      "Kmer word type should be set to uint64_t");
        
        //Read the index and populate the edges inside edgeList
        for(; it != idx.cend(); it++) {
          auto sourceKmer = it->first;
          //Temporary storage for each kmer's neighbors in the graph
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vInNbrs;
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vOutNbrs;

          //Get incoming neighbors
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vInNbrs);
          bliss::de_bruijn::node::node_utils<kmerType,
                                             edgeCountInfoType>::get_in_neighbors(sourceKmer,
                                                                                  it->second,
                                                                                  vInNbrs);

          //Get outgoing neigbors
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vOutNbrs);
          bliss::de_bruijn::node::node_utils<kmerType,
                                             edgeCountInfoType>::get_out_neighbors(sourceKmer,
                                                                                   it->second,
                                                                                   vOutNbrs);

          //typename kmerType::KmerWordType* sourceVertexData = minKmer(sourceKmer).getData();
          auto s = minKmer(sourceKmer).getData()[0];
/*
          if ((s == 1229782938247303441) || (s == 922337203685477580) || (s == 1066452391761333444)) {
            LOG(INFO) << "Rank : " << comm.rank() << " s : " << bliss::utils::KmerUtils::toASCIIString(sourceKmer) << " s : " << s;
            LOG(INFO) << "Rank : " << comm.rank() << " " << *it;
          }
*/

          std::vector<dbgp::adjRecordType>().swap(idxEdgeList);
          for(auto &e : vInNbrs) {
              auto d = minKmer(e.first).getData()[0];
/*
              if ((s == 1229782938247303441) || (s == 922337203685477580) || (s == 1066452391761333444)) {
                auto tdist = std::distance(idx.cbegin(), it);
                LOG(INFO) << comm.rank() << " in " << tdist << " u: " << s << " v: " << d << " ew: " << e.second;
                LOG(INFO) << comm.rank() << " in " << tdist << " u: " << bliss::utils::KmerUtils::toASCIIString(sourceKmer) << " v: " << bliss::utils::KmerUtils::toASCIIString(e.first);
              }
*/
              if (s == d) {
//                LOG(INFO) << "Rank : " << comm.rank() << "  --> Self edge : " << s;
                selfEdgeCount++;
                continue;
              }
              idxEdgeList.emplace_back(s, d, e.second, (uint32_t) 0);
/*
              //Add real edge to tmpEdgeList
              tmpEdgeList.emplace_back(s, d, e.second, (uint32_t) 1);
              //Add test edge to tmpEdgeList
              tmpEdgeList.emplace_back(d, s, e.second, (uint32_t) 2);
*/
          }
          for(auto &e : vOutNbrs) {
              auto d = minKmer(e.first).getData()[0];
/*
              if ((s == 1229782938247303441) || (s == 922337203685477580) || (s == 1066452391761333444)) {
                auto tdist = std::distance(idx.cbegin(), it);
                LOG(INFO) << comm.rank() << " ou " << tdist << " u: " << s << " v: " << d << " ew: " << e.second;
                LOG(INFO) << comm.rank() << " ou " << tdist << " u: " << bliss::utils::KmerUtils::toASCIIString(sourceKmer) << " v: " << bliss::utils::KmerUtils::toASCIIString(e.first);
              }
*/
              if (s == d) {
//                LOG(INFO) << "Rank : " << comm.rank() << "  --> Self edge : " << s;
                selfEdgeCount++;
                continue;
              }
              isPairEdge = 0;
//              for(auto idxitr = idxEdgeList.begin(); idxitr != idxitr_end; idxitr++) {
              for(auto idxitr = idxEdgeList.begin(); idxitr != idxEdgeList.end(); idxitr++) {
                if ((idxitr->vertexU == s) && (idxitr->vertexV == d)) {
#if !NDEBUG
                    assert (isPairEdge == 0);
#endif
                    isPairEdge = 1;
                    pairEdgeCount++;
                    idxitr->edgeWeight += e.second;
                }
              }
              if (!isPairEdge) {
                idxEdgeList.emplace_back(s, d, e.second, (uint32_t) 0);
              }
/*
              //Add real edge to tmpEdgeList
              tmpEdgeList.emplace_back(s, d, e.second, (uint32_t) 1);
              //Add test edge to tmpEdgeList
              tmpEdgeList.emplace_back(d, s, e.second, (uint32_t) 2);
*/
          }
          for(auto idxitr = idxEdgeList.begin(); idxitr != idxEdgeList.end(); idxitr++) {
            //Add real edge to tmpEdgeList
            tmpEdgeList.emplace_back(idxitr->vertexU, idxitr->vertexV, idxitr->edgeWeight, (uint32_t) 1);
            //Add test edge to tmpEdgeList
            tmpEdgeList.emplace_back(idxitr->vertexV, idxitr->vertexU, idxitr->edgeWeight, (uint32_t) 2);
          }
        }
}//End scope for idx

        selfEdgeCount = mxx::reduce(selfEdgeCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "selfEdgeCount : " << selfEdgeCount;
        pairEdgeCount = mxx::reduce(pairEdgeCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "pairEdgeCount : " << pairEdgeCount;

        std::vector<dbgp::adjRecordType>(tmpEdgeList).swap(tmpEdgeList);
        comm.with_subset(
            tmpEdgeList.begin() != tmpEdgeList.end(), [&](const mxx::comm& comm){
                // sort by UID, VID, and NWT
                mxx::sort(tmpEdgeList.begin(), tmpEdgeList.end(),
                          [&](const dbgp::adjRecordType& ex,
                              const dbgp::adjRecordType& ey){
                              return (ex.vertexU < ey.vertexU) ||
                                  ((ex.vertexU == ey.vertexU) && (ex.vertexV < ey.vertexV)) ||
                                  ((ex.vertexU == ey.vertexU) && (ex.vertexV == ey.vertexV) && (ex.vUWeight < ey.vUWeight));
                          }, comm);
            });
        timer.end_section("finished sorting edges");

        uint64_t totalCount = 0, pairsCount = 0, singlesCount = 0, unknownCount=0;
        comm.with_subset(
            tmpEdgeList.begin() != tmpEdgeList.end(), [&](const mxx::comm& comm){
                dbgp::adjRecordType lastRecord(tmpEdgeList.back().vertexU, tmpEdgeList.back().vertexV,
                                               tmpEdgeList.back().edgeWeight, tmpEdgeList.back().vUWeight);
                dbgp::adjRecordType prevRecord = mxx::right_shift(lastRecord, comm);
                auto editr = tmpEdgeList.begin();
                if (comm.rank() == 0) {
                    if (editr->vUWeight == 1) {
                        totalCount++;
                    }
                    prevRecord.set(editr->vertexU, editr->vertexV, editr->edgeWeight, editr->vUWeight);
                    editr++;
                }
                //compactDBG has alternative implementation of below logic
                for(; editr != tmpEdgeList.end(); editr++){
                    if (editr->vUWeight == 1) {
                        totalCount++;
                    }
                    
                    if (prevRecord.vUWeight == 1 && editr->vUWeight == 1) {
                        singlesCount++;
                        auto tdist = std::distance(tmpEdgeList.begin(), editr);
                        LOG(INFO) << comm.rank() << " " << (tdist - 1) << " u: " << prevRecord.vertexU
                        << " v: " << prevRecord.vertexV << " ew: " << prevRecord.edgeWeight << " nw: " << prevRecord.vUWeight;
                        LOG(INFO) << comm.rank() << " " << tdist << " u: " << editr->vertexU
                        << " v: " << editr->vertexV << " ew: " << editr->edgeWeight << " nw: " << editr->vUWeight;
                    } else if (prevRecord.vUWeight == 1 && editr->vUWeight == 2) {
                        if (prevRecord.vertexU == editr->vertexU &&
                            prevRecord.vertexV == editr->vertexV &&
                            prevRecord.edgeWeight == editr->edgeWeight){
                            pairsCount++;
                        } else {
                            unknownCount++;
                        }
                    } else if (prevRecord.vUWeight == 2 && editr->vUWeight == 1) {
                        //Do nothing
                    } else {
                        singlesCount++;
                        auto tdist = std::distance(tmpEdgeList.begin(), editr);
                        LOG(INFO) << comm.rank() << " " << (tdist - 1) << " u: " << prevRecord.vertexU
                        << " v: " << prevRecord.vertexV << " ew: " << prevRecord.edgeWeight << " nw: " << prevRecord.vUWeight;
                        LOG(INFO) << comm.rank() << " " << tdist << " u: " << editr->vertexU
                        << " v: " << editr->vertexV << " ew: " << editr->edgeWeight << " nw: " << editr->vUWeight;
                    }
                    
                    prevRecord.set(editr->vertexU, editr->vertexV, editr->edgeWeight, editr->vUWeight);
                }
           });
        totalCount = mxx::reduce(totalCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "totalCount   : " << totalCount;
        pairsCount = mxx::reduce(pairsCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "pairsCount   : " << pairsCount;
        singlesCount = mxx::reduce(singlesCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "singlesCount : " << singlesCount;
        unknownCount = mxx::reduce(unknownCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "unknownCount : " << unknownCount;
        timer.end_section("finished analyzing for single edges");
      }


      /**
       * @brief       populates edge list vector
       * @param[in]   minKmerFreq
       * @param[in]   fileName
       * @param[out]  edgeList_1
       */
      template <typename ERT>
      void populateEdgeList( std::vector< ERT > &edgeList_1,
                             std::uint64_t &minKmerFreq, std::string &fileName,
                             std::string &filteredKmersFile,
                             const mxx::comm &comm) {
       dbgp::Timer timer;
       std::vector<ERT> tmpEdgeList;

        //Declare a vector to store filtered out kmers
        std::vector< uint64_t > delKmersList;
        uint64_t delKmersCount = 0;

#if !NDEBUG
        uint32_t nDegree = 0, eFreq = 0;
        std::vector<uint64_t> nodeDegreesBefore;
        nodeDegreesBefore.resize(9);
        std::vector<uint64_t> nodeDegreesAfter;
        nodeDegreesAfter.resize(9);
        std::vector<uint64_t> edgeFreqsBefore; //log scale
        edgeFreqsBefore.resize(30);
        std::vector<uint64_t> edgeFreqsAfter; //log scale
        edgeFreqsAfter.resize(30);
        uint64_t localWorkLoad, minLoad, maxLoad, avgLoad;
#endif
        uint64_t localEdgesCountBefore = 0, remoteEdgesCountBefore = 0;
        uint64_t localEdgesSumBefore = 0, remoteEdgesSumBefore = 0;
        uint64_t localEdgesCountAfter = 0, remoteEdgesCountAfter = 0;
        uint64_t localEdgesSumAfter = 0, remoteEdgesSumAfter = 0;

{//Start scope for idx
        std::size_t selfEdgeCount = 0, pairEdgeCount = 0;
        bool isPairEdge;

       //Initialize the map for murmur hash
       bliss::de_bruijn::de_bruijn_engine<dbgp::NodeMapTypeM> idx(comm);

       //Build the de Bruijn graph as distributed map
       idx.template build_posix<dbgp::SeqParser, bliss::io::SequencesIterator>(fileName, comm);
       timer.end_section("finished building dB graph using BLISS and murmur hash");

        auto it = idx.cbegin();

        // Deriving data types of de Bruijn graph storage container
        using mapPairType       = typename std::iterator_traits<decltype(it)>::value_type;
        using constkmerType     = typename std::tuple_element<0, mapPairType>::type;
        using kmerType          = typename std::remove_const<constkmerType>::type;
        using edgeCountInfoType = typename std::tuple_element<1, mapPairType>::type;

        bliss::kmer::transform::lex_less<KmerType> minKmer;

        static_assert(std::is_same<typename kmerType::KmerWordType, uint64_t>::value,
                      "Kmer word type should be set to uint64_t");
        
        //Read the index and populate the edges inside edgeList
        for(; it != idx.cend(); it++) {
          auto sourceKmer = it->first;
          //Temporary storage for each kmer's neighbors in the graph
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vInNbrs;
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vOutNbrs;
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vGoodInNbrs;
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vGoodOutNbrs;

          //Get incoming neighbors
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vInNbrs);
          bliss::de_bruijn::node::node_utils<kmerType,
                                             edgeCountInfoType>::get_in_neighbors(sourceKmer,
                                                                                  it->second,
                                                                                  vInNbrs);

          //Get outgoing neigbors
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vOutNbrs);
          bliss::de_bruijn::node::node_utils<kmerType,
                                             edgeCountInfoType>::get_out_neighbors(sourceKmer,
                                                                                   it->second,
                                                                                   vOutNbrs);

          //typename kmerType::KmerWordType* sourceVertexData = minKmer(sourceKmer).getData();
          auto s = minKmer(sourceKmer).getData()[0];
          
          bliss::kmer::hash::murmur<KmerType, true> tmpDistHash(ceilLog2(comm.size()));
          auto kmerRank = tmpDistHash(minKmer(sourceKmer)) % comm.size();
          assert (kmerRank == comm.rank());

          //Filter out low frequency (k+1)-mers
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vGoodInNbrs);
          for(auto &e : vInNbrs) {
              auto d = minKmer(e.first).getData()[0];
              if (s == d) {
//                LOG(INFO) << "Rank : " << comm.rank() << "  --> Self edge : " << s;
                selfEdgeCount++;
                continue;
              }

            if ((uint64_t)e.second >= minKmerFreq) {
              vGoodInNbrs.emplace_back(e);
            }

            kmerRank = tmpDistHash(minKmer(e.first)) % comm.size();
            if (kmerRank == comm.rank()) {
              localEdgesCountBefore++;
              localEdgesSumBefore += (uint64_t) e.second;
            } else {
              remoteEdgesCountBefore++;
              remoteEdgesSumBefore += (uint64_t) e.second;
            }
          }
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vGoodOutNbrs);
          for(auto &e : vOutNbrs) {
              auto d = minKmer(e.first).getData()[0];
              if (s == d) {
//                LOG(INFO) << "Rank : " << comm.rank() << "  --> Self edge : " << s;
                selfEdgeCount++;
                continue;
              }

            if ((uint64_t)e.second >= minKmerFreq) {
              isPairEdge = 0;
              for(auto idxitr = vGoodInNbrs.begin(); idxitr != vGoodInNbrs.end(); idxitr++) {
                if (minKmer(idxitr->first).getData()[0] == d) {
#if !NDEBUG
                    assert (isPairEdge == 0);
#endif
                    isPairEdge = 1;
                    pairEdgeCount++;
                    idxitr->second += e.second;
                }
              }
              if (!isPairEdge) {
                vGoodOutNbrs.emplace_back(e);
              }
            }

            kmerRank = tmpDistHash(minKmer(e.first)) % comm.size();
            if (kmerRank == comm.rank()) {
              localEdgesCountBefore++;
              localEdgesSumBefore += (uint64_t) e.second;
            } else {
              remoteEdgesCountBefore++;
              remoteEdgesSumBefore += (uint64_t) e.second;
            }
          }

#if !NDEBUG
        nDegree = vInNbrs.size() + vOutNbrs.size();
        assert (nDegree < 9);
        (nodeDegreesBefore[nDegree])++;
        nDegree = vGoodInNbrs.size() + vGoodOutNbrs.size();
        assert (nDegree < 9);
        (nodeDegreesAfter[nDegree])++;
        for(auto &e : vInNbrs) {
            eFreq = (uint32_t) std::log2(e.second);
            assert (eFreq < 30);
            (edgeFreqsBefore[eFreq])++;
        }
        for(auto &e : vOutNbrs) {
            eFreq = (uint32_t) std::log2(e.second);
            assert (eFreq < 30);
            (edgeFreqsBefore[eFreq])++;
        }
        for(auto &e : vGoodInNbrs) {
            eFreq = (uint32_t) std::log2(e.second);
            assert (eFreq < 30);
            (edgeFreqsAfter[eFreq])++;
        }
        for(auto &e : vGoodOutNbrs) {
            eFreq = (uint32_t) std::log2(e.second);
            assert (eFreq < 30);
            (edgeFreqsAfter[eFreq])++;
        }
        
#endif

          if (vGoodInNbrs.size() == 0 && vGoodOutNbrs.size() == 0) {
            
#if !NDEBUG
            if (!filteredKmersFile.empty()) {
                //Artifact of filtering out low frequency (k+1)-mers
                delKmersList.emplace_back(s);
            }
#endif
            delKmersCount++;
            
          } else {
            for(auto &e : vGoodInNbrs) {
              auto d = minKmer(e.first).getData()[0];
//              //Add edge to edgeList_1
//              edgeList_1.emplace_back(s, d, e.second, NODE_WEIGHT_MISSING);
              //Add real edge to tmpEdgeList
              tmpEdgeList.emplace_back(s, d, e.second, (uint32_t) 1);
              //Add test edge to tmpEdgeList
              tmpEdgeList.emplace_back(d, s, e.second, (uint32_t) 2);
              
              kmerRank = tmpDistHash(minKmer(e.first)) % comm.size();
              if (kmerRank == comm.rank()) {
                localEdgesCountAfter++;
                localEdgesSumAfter += (uint64_t) e.second;
              } else {
                remoteEdgesCountAfter++;
                remoteEdgesSumAfter += (uint64_t) e.second;
              }
            }

            for(auto &e : vGoodOutNbrs) {
              auto d = minKmer(e.first).getData()[0];
//              //Add edge to edgeList_1
//              edgeList_1.emplace_back(s, d, e.second, NODE_WEIGHT_MISSING);
              //Add real edge to tmpEdgeList
              tmpEdgeList.emplace_back(s, d, e.second, (uint32_t) 1);
              //Add test edge to tmpEdgeList
              tmpEdgeList.emplace_back(d, s, e.second, (uint32_t) 2);
              
              kmerRank = tmpDistHash(minKmer(e.first)) % comm.size();
              if (kmerRank == comm.rank()) {
                localEdgesCountAfter++;
                localEdgesSumAfter += (uint64_t) e.second;
              } else {
                remoteEdgesCountAfter++;
                remoteEdgesSumAfter += (uint64_t) e.second;
              }
            }
          }
        }
        
        selfEdgeCount = mxx::reduce(selfEdgeCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "selfEdgeCount : " << selfEdgeCount;
        pairEdgeCount = mxx::reduce(pairEdgeCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "pairEdgeCount : " << pairEdgeCount;
}//End scope for idx
        //Compact edgeList_1 and delKmersList
        std::vector<ERT>(edgeList_1).swap(edgeList_1);
        std::vector<ERT>(tmpEdgeList).swap(tmpEdgeList);
        std::vector<uint64_t>(delKmersList).swap(delKmersList);
        timer.end_section("finished populating tmpEdgeList and delKmersList");
        
#if !NDEBUG
        //Gather and print statistics
        //nodeDegreesBefore.resize(9);
        nodeDegreesBefore = mxx::reduce(nodeDegreesBefore, 0, comm);
        //nodeDegreesAfter.resize(9);
        nodeDegreesAfter = mxx::reduce(nodeDegreesAfter, 0, comm);
        //edgeFreqsBefore.resize(30);
        edgeFreqsBefore = mxx::reduce(edgeFreqsBefore, 0, comm);
        //edgeFreqsAfter.resize(30);
        edgeFreqsAfter = mxx::reduce(edgeFreqsAfter, 0, comm);
        
        for(auto xitr = 0; xitr < nodeDegreesBefore.size(); xitr++) {
            LOG_IF(comm.rank() == 0, INFO) << "Node degrees before [" << xitr << "] : " << nodeDegreesBefore[xitr];
        }
        for(auto xitr = 0; xitr < nodeDegreesAfter.size(); xitr++) {
            LOG_IF(comm.rank() == 0, INFO) << "Node degrees after  [" << xitr << "] : " << nodeDegreesAfter[xitr];
        }
        for(auto xitr = 0; xitr < edgeFreqsBefore.size(); xitr++) {
            LOG_IF(comm.rank() == 0, INFO) << "Edge frequencies before [" << xitr << "] : " << edgeFreqsBefore[xitr];
        }
        for(auto xitr = 0; xitr < edgeFreqsAfter.size(); xitr++) {
            LOG_IF(comm.rank() == 0, INFO) << "Edge frequencies after  [" << xitr << "] : " << edgeFreqsAfter[xitr];
        }
        localWorkLoad = tmpEdgeList.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>() , comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>() , comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of tmpEdgeList. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
        localWorkLoad = delKmersCount;
//        localWorkLoad = delKmersList.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>() , comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>() , comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of delKmersList. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif
        localEdgesCountBefore = mxx::reduce(localEdgesCountBefore, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Cnt of local  edges before : " << localEdgesCountBefore;
        remoteEdgesCountBefore = mxx::reduce(remoteEdgesCountBefore, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Cnt of remote edges before : " << remoteEdgesCountBefore;
        localEdgesSumBefore = mxx::reduce(localEdgesSumBefore, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Sum of local  edges before : " << localEdgesSumBefore;
        remoteEdgesSumBefore = mxx::reduce(remoteEdgesSumBefore, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Sum of remote edges before : " << remoteEdgesSumBefore;
        localEdgesCountAfter = mxx::reduce(localEdgesCountAfter, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Cnt of local  edges after  : " << localEdgesCountAfter;
        remoteEdgesCountAfter = mxx::reduce(remoteEdgesCountAfter, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Cnt of remote edges after  : " << remoteEdgesCountAfter;
        localEdgesSumAfter = mxx::reduce(localEdgesSumAfter, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Sum of local  edges after  : " << localEdgesSumAfter;
        remoteEdgesSumAfter = mxx::reduce(remoteEdgesSumAfter, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Sum of remote edges after  : " << remoteEdgesSumAfter;
        
#if !NDEBUG
        if (!filteredKmersFile.empty()) {
            //Write filtered (low frequency) kmers to a file
            std::ofstream outFilePtr(filteredKmersFile);
            for(auto xitr = delKmersList.begin(); xitr != delKmersList.end(); xitr++) {
                outFilePtr << *xitr;
                outFilePtr << std::endl;
            }
            outFilePtr.close();
            std::vector<uint64_t>().swap(delKmersList);
        }
#endif
        timer.end_section("finished writing filteredKmersFile");

        comm.with_subset(
            tmpEdgeList.begin() != tmpEdgeList.end(), [&](const mxx::comm& comm){
                // sort by UID, VID, and NWT
                mxx::sort(tmpEdgeList.begin(), tmpEdgeList.end(),
                          [&](const dbgp::adjRecordType& ex,
                              const dbgp::adjRecordType& ey){
                              return (ex.vertexU < ey.vertexU) ||
                                  ((ex.vertexU == ey.vertexU) && (ex.vertexV < ey.vertexV)) ||
                                  ((ex.vertexU == ey.vertexU) && (ex.vertexV == ey.vertexV) && (ex.vUWeight < ey.vUWeight));
                          }, comm);
            });
        timer.end_section("finished sorting edges");

        uint64_t totalCount = 0, pairsCount = 0, singlesCount = 0, unknownCount=0;
        comm.with_subset(
            tmpEdgeList.begin() != tmpEdgeList.end(), [&](const mxx::comm& comm){
                dbgp::adjRecordType lastRecord(tmpEdgeList.back().vertexU, tmpEdgeList.back().vertexV,
                                               tmpEdgeList.back().edgeWeight, tmpEdgeList.back().vUWeight);
                dbgp::adjRecordType prevRecord = mxx::right_shift(lastRecord, comm);
                auto editr = tmpEdgeList.begin();
                if (comm.rank() == 0) {
                    if (editr->vUWeight == 1) {
                        totalCount++;
                    }
                    prevRecord.set(editr->vertexU, editr->vertexV, editr->edgeWeight, editr->vUWeight);
                    editr++;
                }
                //compactDBG has alternative implementation of below logic
                for(; editr != tmpEdgeList.end(); editr++){
                    if (editr->vUWeight == 1) {
                        totalCount++;
                    }
                    
                    if (prevRecord.vUWeight == 1 && editr->vUWeight == 1) {
                        //Do nothing
                        singlesCount++;
                        auto tdist = std::distance(tmpEdgeList.begin(), editr);
                        LOG(INFO) << comm.rank() << " " << (tdist - 1) << " u: " << prevRecord.vertexU
                        << " v: " << prevRecord.vertexV << " ew: " << prevRecord.edgeWeight << " nw: " << prevRecord.vUWeight;
                        LOG(INFO) << comm.rank() << " " << tdist << " u: " << editr->vertexU
                        << " v: " << editr->vertexV << " ew: " << editr->edgeWeight << " nw: " << editr->vUWeight;
                    } else if (prevRecord.vUWeight == 1 && editr->vUWeight == 2) {
                        if (prevRecord.vertexU == editr->vertexU &&
                            prevRecord.vertexV == editr->vertexV &&
                            prevRecord.edgeWeight == editr->edgeWeight){
                            pairsCount++;
                            edgeList_1.emplace_back(editr->vertexU, editr->vertexV,
                                                    editr->edgeWeight, NODE_WEIGHT_MISSING);
                        } else {
                            //Do nothing
                            unknownCount++;
                        }
                    } else if (prevRecord.vUWeight == 2 && editr->vUWeight == 1) {
                        //Do nothing
                    } else {
                        //Do nothing
                        singlesCount++;
                        auto tdist = std::distance(tmpEdgeList.begin(), editr);
                        LOG(INFO) << comm.rank() << " " << (tdist - 1) << " u: " << prevRecord.vertexU
                        << " v: " << prevRecord.vertexV << " ew: " << prevRecord.edgeWeight << " nw: " << prevRecord.vUWeight;
                        LOG(INFO) << comm.rank() << " " << tdist << " u: " << editr->vertexU
                        << " v: " << editr->vertexV << " ew: " << editr->edgeWeight << " nw: " << editr->vUWeight;
                    }
                    
                    prevRecord.set(editr->vertexU, editr->vertexV, editr->edgeWeight, editr->vUWeight);
                }
           });
        totalCount = mxx::reduce(totalCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "totalCount   : " << totalCount;
        pairsCount = mxx::reduce(pairsCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "pairsCount   : " << pairsCount;
        singlesCount = mxx::reduce(singlesCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "singlesCount : " << singlesCount;
        unknownCount = mxx::reduce(unknownCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "unknownCount : " << unknownCount;
        std::vector<ERT>().swap(tmpEdgeList);
        std::vector<ERT>(edgeList_1).swap(edgeList_1);
        timer.end_section("finished analyzing bad edges");


        localEdgesCountBefore = 0, remoteEdgesCountBefore = 0;
        localEdgesSumBefore = 0, remoteEdgesSumBefore = 0;
        localEdgesCountAfter = 0, remoteEdgesCountAfter = 0;
        localEdgesSumAfter = 0, remoteEdgesSumAfter = 0;
/*
{//Start scope for idx
        std::size_t selfEdgeCount = 0, pairEdgeCount = 0;
        bool isPairEdge;

       //Initialize the map for farm hash
       bliss::de_bruijn::de_bruijn_engine<dbgp::NodeMapTypeF> idx(comm);

       //Build the de Bruijn graph as distributed map
       idx.template build_posix<dbgp::SeqParser, bliss::io::SequencesIterator>(fileName, comm);
       timer.end_section("finished building dB graph using BLISS and farm hash");

        auto it = idx.cbegin();

        // Deriving data types of de Bruijn graph storage container
        using mapPairType       = typename std::iterator_traits<decltype(it)>::value_type;
        using constkmerType     = typename std::tuple_element<0, mapPairType>::type;
        using kmerType          = typename std::remove_const<constkmerType>::type;
        using edgeCountInfoType = typename std::tuple_element<1, mapPairType>::type;

        bliss::kmer::transform::lex_less<KmerType> minKmer;

        static_assert(std::is_same<typename kmerType::KmerWordType, uint64_t>::value,
                      "Kmer word type should be set to uint64_t");
        
        //Read the index and populate the edges inside edgeList
        for(; it != idx.cend(); it++) {
          auto sourceKmer = it->first;
          //Temporary storage for each kmer's neighbors in the graph
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vInNbrs;
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vOutNbrs;
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vGoodInNbrs;
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>> vGoodOutNbrs;

          //Get incoming neighbors
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vInNbrs);
          bliss::de_bruijn::node::node_utils<kmerType,
                                             edgeCountInfoType>::get_in_neighbors(sourceKmer,
                                                                                  it->second,
                                                                                  vInNbrs);

          //Get outgoing neigbors
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vOutNbrs);
          bliss::de_bruijn::node::node_utils<kmerType,
                                             edgeCountInfoType>::get_out_neighbors(sourceKmer,
                                                                                   it->second,
                                                                                   vOutNbrs);

          //typename kmerType::KmerWordType* sourceVertexData = minKmer(sourceKmer).getData();
          auto s = minKmer(sourceKmer).getData()[0];
          
          bliss::kmer::hash::farm<KmerType, true> tmpDistHash(ceilLog2(comm.size()));
          auto kmerRank = tmpDistHash(minKmer(sourceKmer)) % comm.size();
          assert (kmerRank == comm.rank());

          //Filter out low frequency (k+1)-mers
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vGoodInNbrs);
          for(auto &e : vInNbrs) {
              auto d = minKmer(e.first).getData()[0];
              if (s == d) {
//                LOG(INFO) << "Rank : " << comm.rank() << "  --> Self edge : " << s;
                selfEdgeCount++;
                continue;
              }

            if ((uint64_t)e.second >= minKmerFreq) {
              vGoodInNbrs.emplace_back(e);
            }

            kmerRank = tmpDistHash(minKmer(e.first)) % comm.size();
            if (kmerRank == comm.rank()) {
              localEdgesCountBefore++;
              localEdgesSumBefore += (uint64_t) e.second;
            } else {
              remoteEdgesCountBefore++;
              remoteEdgesSumBefore += (uint64_t) e.second;
            }
          }
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vGoodOutNbrs);
          for(auto &e : vOutNbrs) {
              auto d = minKmer(e.first).getData()[0];
              if (s == d) {
//                LOG(INFO) << "Rank : " << comm.rank() << "  --> Self edge : " << s;
                selfEdgeCount++;
                continue;
              }

            if ((uint64_t)e.second >= minKmerFreq) {
              isPairEdge = 0;
              for(auto idxitr = vGoodInNbrs.begin(); idxitr != vGoodInNbrs.end(); idxitr++) {
                if (minKmer(idxitr->first).getData()[0] == d) {
                    assert (isPairEdge == 0);
                    isPairEdge = 1;
                    pairEdgeCount++;
                    idxitr->second += e.second;
                }
              }
              if (!isPairEdge) {
                vGoodOutNbrs.emplace_back(e);
              }
            }

            kmerRank = tmpDistHash(minKmer(e.first)) % comm.size();
            if (kmerRank == comm.rank()) {
              localEdgesCountBefore++;
              localEdgesSumBefore += (uint64_t) e.second;
            } else {
              remoteEdgesCountBefore++;
              remoteEdgesSumBefore += (uint64_t) e.second;
            }
          }

          if (vGoodInNbrs.size() == 0 && vGoodOutNbrs.size() == 0) {
            //Nothing to do
          } else {
            for(auto &e : vGoodInNbrs) {
              kmerRank = tmpDistHash(minKmer(e.first)) % comm.size();
              if (kmerRank == comm.rank()) {
                localEdgesCountAfter++;
                localEdgesSumAfter += (uint64_t) e.second;
              } else {
                remoteEdgesCountAfter++;
                remoteEdgesSumAfter += (uint64_t) e.second;
              }
            }

            for(auto &e : vGoodOutNbrs) {
              kmerRank = tmpDistHash(minKmer(e.first)) % comm.size();
              if (kmerRank == comm.rank()) {
                localEdgesCountAfter++;
                localEdgesSumAfter += (uint64_t) e.second;
              } else {
                remoteEdgesCountAfter++;
                remoteEdgesSumAfter += (uint64_t) e.second;
              }
            }
          }
        }

        selfEdgeCount = mxx::reduce(selfEdgeCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "selfEdgeCount : " << selfEdgeCount;
        pairEdgeCount = mxx::reduce(pairEdgeCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "pairEdgeCount : " << pairEdgeCount;
}//End scope for idx
*/
        timer.end_section("finished analyzing for farm hash");

        localEdgesCountBefore = mxx::reduce(localEdgesCountBefore, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Cnt of local  edges before : " << localEdgesCountBefore;
        remoteEdgesCountBefore = mxx::reduce(remoteEdgesCountBefore, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Cnt of remote edges before : " << remoteEdgesCountBefore;
        localEdgesSumBefore = mxx::reduce(localEdgesSumBefore, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Sum of local  edges before : " << localEdgesSumBefore;
        remoteEdgesSumBefore = mxx::reduce(remoteEdgesSumBefore, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Sum of remote edges before : " << remoteEdgesSumBefore;
        localEdgesCountAfter = mxx::reduce(localEdgesCountAfter, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Cnt of local  edges after  : " << localEdgesCountAfter;
        remoteEdgesCountAfter = mxx::reduce(remoteEdgesCountAfter, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Cnt of remote edges after  : " << remoteEdgesCountAfter;
        localEdgesSumAfter = mxx::reduce(localEdgesSumAfter, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Sum of local  edges after  : " << localEdgesSumAfter;
        remoteEdgesSumAfter = mxx::reduce(remoteEdgesSumAfter, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Sum of remote edges after  : " << remoteEdgesSumAfter;
      }
    };
  };


    void writeVertexList(mxx::comm& comm,
                         std::vector<dbgp::adjRecordType>& cEdges,
                         std::string t2FileName){
        //Write out graph nodes in color space (vertices in vertList)
        comm.with_subset(
            cEdges.begin() != cEdges.end(), [&](const mxx::comm& comm){
                auto lastVertex = cEdges.back().vertexU;
                auto prevVertex = mxx::right_shift(lastVertex, comm);
                if(comm.rank() == 0) prevVertex = NODE_ID_MISSING;
                std::ofstream outFilePtr(t2FileName);
                for(auto ed: cEdges){
                    if(ed.vertexU != prevVertex){
                        outFilePtr << ed.vertexU << std::endl;
                        prevVertex = ed.vertexU;
                    }
                }
                outFilePtr.close();
            });
    }


    template<typename EdgeIterator>
    void emitEdgeList(EdgeIterator edgeBegin, EdgeIterator edgeEnd,
                      std::ofstream& outFilePtr, char dbgNodeWtType){
        assert (edgeBegin != edgeEnd);
        dbgp::weightType totalVertexWt = 0;
        for(auto xitr = edgeBegin; xitr != edgeEnd; xitr++){
            if (dbgNodeWtType == 'c') {
                //junction node
                if (xitr->vUWeight == NODE_WEIGHT_MISSING) {
                    xitr->vUWeight = 1;
                }
                //non-junction node has correct weight
            } else {
                totalVertexWt += xitr->edgeWeight;
            }
        }
        for(auto xitr = edgeBegin; xitr != edgeEnd; xitr++){
            //dbgNodeWtType is 's'
            if (dbgNodeWtType == 's') {
                //junction node
                if (xitr->vUWeight == NODE_WEIGHT_MISSING) {
                    xitr->vUWeight = totalVertexWt;
                } else {
                    //weights of edges in colors are captured twice
                    assert (xitr->vUWeight % 2 == 0);
                    xitr->vUWeight = (xitr->vUWeight / 2) + totalVertexWt;
                }
            }
        }
        assert (edgeBegin->vUWeight != NODE_WEIGHT_MISSING);
        outFilePtr << edgeBegin->vUWeight;
        for(auto xitr = edgeBegin; xitr != edgeEnd; xitr++){
            outFilePtr << " " << xitr->vertexV << " " << xitr->edgeWeight;
        }
        outFilePtr << std::endl;
    }


    void writeAdjacencyList(mxx::comm& comm,
                            std::vector<dbgp::adjRecordType>& cEdges,
                            std::string outFileName,
                            std::size_t totalVertexCount,
                            char dbgNodeWtType){
        comm.with_subset(
            cEdges.begin() != cEdges.end(), [&](const mxx::comm& comm){
                // count edges
                std::size_t nEdges = 0;
                nEdges = mxx::allreduce(cEdges.size(), comm);
                assert (nEdges % 2 == 0);
                nEdges = nEdges / 2; // nEdges should be an even number

                auto vertuCompare = [&](const dbgp::adjRecordType& x,
                                        const dbgp::adjRecordType& y){
                    return x.vertexU == y.vertexU;
                };

                // shift region
                uint64_t startOffset, endOffset;
                std::vector<dbgp::adjRecordType> straddleRegion;
                shiftStraddlingRegion(comm, cEdges,
                                      startOffset, endOffset,
                                      straddleRegion,
                                      vertuCompare);

                auto edgItr = cEdges.begin() + startOffset;
                auto prevItr = edgItr;

                std::ofstream outFilePtr(outFileName);
                // rank 0 should output total no. of edges
                if(comm.rank() == 0){
                    outFilePtr << totalVertexCount;
                    outFilePtr << " " << nEdges;
                    outFilePtr << " " << "11";
                    outFilePtr << std::endl;
                }
                //cEdges.begin() + endOffset <= cEdges.end()
                for(; edgItr != cEdges.begin() + endOffset; edgItr++){
                    if((*edgItr).vertexU == (*prevItr).vertexU)
                        continue;
                    emitEdgeList(prevItr, edgItr, outFilePtr, dbgNodeWtType);
                    prevItr = edgItr;
                }
                assert (edgItr == cEdges.begin() + endOffset);
                assert (prevItr != edgItr);
                emitEdgeList(prevItr, edgItr, outFilePtr, dbgNodeWtType);
                if (comm.rank() != comm.size() - 1) {
                    // write out straddling region
                    emitEdgeList(straddleRegion.begin(), straddleRegion.end(), outFilePtr, dbgNodeWtType);
                }
                outFilePtr.close();
            });
    }


    void translateVertexIds(mxx::comm& comm,
                            std::vector<dbgp::adjRecordType>& cEdges,
                            std::string outFileName,
                            std::string t2FileName,
                            char dbgNodeWtType){

        dbgp::Timer timer;
/*
        // sort edges
        sortEdges(comm, cEdges);
        timer.end_section("finished sorting edges");
        // unique sorted vertex list
        std::vector<dbgp::vertexIdType> vertList;
        constructVertexList(comm, cEdges, vertList);
        timer.end_section("finished constructing vertex list");
        writeVertexList(comm, vertList, t2FileName);
        timer.end_section("finished writing vertex list");
*/

        // redistribute equally
        mxx::distribute_inplace(cEdges, comm);
        timer.end_section("finished redistributing edges");

        // update vertex V's
        comm.with_subset(
            cEdges.begin() != cEdges.end(), [&](const mxx::comm& comm){
                // sort by VID
                mxx::sort(cEdges.begin(), cEdges.end(),
                          [&](const dbgp::adjRecordType& ex,
                              const dbgp::adjRecordType& ey){
                              return (ex.vertexV < ey.vertexV);
                          }, comm);
            });
        timer.end_section("finished sorting edges by Vs");
        comm.with_subset(
            cEdges.begin() != cEdges.end(), [&](const mxx::comm& comm){
                std::size_t vertIdx = 0;
                auto lastVertex = cEdges.back().vertexV;
                auto prevVertex = mxx::right_shift(lastVertex, comm);
                if(comm.rank() == 0) prevVertex = NODE_ID_MISSING;
                auto savePrevVertex = prevVertex;
                for(auto ed: cEdges){
                    if(ed.vertexV != prevVertex){
                        vertIdx++;
                        prevVertex = ed.vertexV;
                    }
                }

                std::size_t nVertPfxSum = 0;
                std::vector<std::size_t> vertSizes;
                vertSizes  = mxx::allgather(vertIdx, comm);
                for(auto i = 0; i < comm.rank(); i++){
                    nVertPfxSum += vertSizes[i];
                }
                
                prevVertex = savePrevVertex;
                for(auto editr = cEdges.begin(); editr != cEdges.end(); editr++){
                    if(editr->vertexV != prevVertex){
                        nVertPfxSum++;
                        prevVertex = editr->vertexV;
                    }
                    editr->vertexV = nVertPfxSum;
                }
           });
        timer.end_section("finished updating vertex Vs");

        // update vertex U's
        comm.with_subset(
            cEdges.begin() != cEdges.end(), [&](const mxx::comm& comm){
                // sort by UID and then by VID
                mxx::sort(cEdges.begin(), cEdges.end(),
                          [&](const dbgp::adjRecordType& ex,
                              const dbgp::adjRecordType& ey){
                              return (ex.vertexU < ey.vertexU) ||
                                  ((ex.vertexU == ey.vertexU) &&
                                   (ex.vertexV < ey.vertexV));
                          }, comm);
            });
        timer.end_section("finished sorting edges by Us");
//        writeVertexList(comm, cEdges, t2FileName);
        timer.end_section("finished writing vertex list");        
        std::size_t totalVertexCount = 0;
        comm.with_subset(
            cEdges.begin() != cEdges.end(), [&](const mxx::comm& comm){
                std::size_t vertIdx = 0;
                auto lastVertex = cEdges.back().vertexU;
                auto prevVertex = mxx::right_shift(lastVertex, comm);
                if(comm.rank() == 0) prevVertex = NODE_ID_MISSING;
                auto savePrevVertex = prevVertex;
                for(auto ed: cEdges){
                    if(ed.vertexU != prevVertex){
                        vertIdx++;
                        prevVertex = ed.vertexU;
                    }
                }

                std::size_t nVertPfxSum = 0;
                std::vector<std::size_t> vertSizes;
                vertSizes  = mxx::allgather(vertIdx, comm);
                for(auto i = 0; i < comm.rank(); i++){
                    nVertPfxSum += vertSizes[i];
                }
                for(auto i = 0; i < comm.size(); i++){
                    totalVertexCount += vertSizes[i];
                }
                
                prevVertex = savePrevVertex;
                for(auto editr = cEdges.begin(); editr != cEdges.end(); editr++){
                    if(editr->vertexU != prevVertex){
                        nVertPfxSum++;
                        prevVertex = editr->vertexU;
                    }
                    editr->vertexU = nVertPfxSum;
                }
           });
        timer.end_section("finished updating vertex Us");

/*
        // update vertex V from local maps
        comm.with_subset(
            cEdges.begin() != cEdges.end(), [&](const mxx::comm& comm){
                std::map<dbgp::vertexIdType, uint64_t> localVerts;
                //TODO: Is (nVertPfxSum+1) below correct?
                buildLocalVertexMap(comm, cEdges, (nVertPfxSum+1),
                                    vertList, localVerts);
                for(auto editr = cEdges.begin(); editr != cEdges.end(); editr++){
                    if(editr->vertexV != NODE_ID_MISSING){
                        auto vitx = localVerts.find(editr->vertexV);
                        if(vitx != localVerts.end()){
                            editr->vertexV = vitx->second;
                        }
                    }
                }
           });
         timer.end_section("finished updating vertex Vs");
*/
//        // write Adjacency List
//        writeAdjacencyList(comm, cEdges, outFileName, totalVertexCount, dbgNodeWtType);
        timer.end_section("finished writing adjacency list");
    }
};


int main(int argc, char* argv[])
{

  // Initialize the MPI library:.
  MPI_Init(&argc, &argv);

  //Initialize the communicator
  mxx::comm comm;

  //Print mpi rank distribution
  mxx::print_node_distribution();

  // COMMAND LINE ARGUMENTS
  LOG_IF(!comm.rank(), INFO) << "Start Constructing Plain DBG";

  //Parse command line arguments
  InputArgs cargs;

  if(parse_args(argc, argv, comm, cargs)) return 1;

  dbgp::init(); // initialize static variables
  
  std::vector< dbgp::adjRecordType > dbgEdges;

//  dbgp::graphGen::deBruijnGraph g0;
//  g0.checkEdgeList(cargs.ipFileName, comm);

  //Construct graph
  LOG_IF(!comm.rank(), INFO) << "Begin populating de Bruijn graph edges";
  //Object of the graph generator class
  dbgp::graphGen::deBruijnGraph g;
  // Populate the edges from the DBG
  g.populateEdgeList(dbgEdges, cargs.minKmerFreq, cargs.ipFileName, cargs.filteredKmersPrefix, comm);
  LOG_IF(!comm.rank(), INFO) << "End populating de Bruijn graph edges";
#if !NDEBUG
                // count edges
                std::size_t nEdges = 0;
                nEdges = mxx::allreduce(dbgEdges.size(), comm);
                LOG_IF(!comm.rank(), INFO) << "Total edge count : " << nEdges;
#endif

  LOG_IF(!comm.rank(), INFO) << "Begin translating vertex IDs";
  // change from kmer based node ids to 1...|V|, and write out adjacency list
  dbgp::translateVertexIds(comm, dbgEdges, cargs.ajFilePrefix, cargs.colorVertexListPrefix, cargs.dbgNodeWtType);
  LOG_IF(!comm.rank(), INFO) << "End translating vertex IDs";

  MPI_Finalize();
  return(0);
}


