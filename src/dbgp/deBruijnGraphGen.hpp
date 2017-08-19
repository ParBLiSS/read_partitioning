/*
 * Copyright 2016 Georgia Institute of Technology
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

/**
 * @file    deBruijnGraphGen.hpp
 * @ingroup graphGen
 * @author  Chirag Jain <cjain7@gatech.edu>
 *          Sriram P C <srirampc@gmail.com>
 * @brief   Builds the edgelist for de bruijn graph using BLISS library.
 *
 * Copyright (c) 2015 Georgia Institute of Technology. All Rights Reserved.
 */

#ifndef DE_BRUIJN_GEN_HPP
#define DE_BRUIJN_GEN_HPP

//Includes
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

//Own includes
#include "dbgp/timer.hpp"
#include "utils/commonfuncs.hpp"
#include "utils/mpi_utils.hpp"
#include "dbgp/recordTypes.hpp"

//External includes
#include "debruijn/de_bruijn_node_trait.hpp"
#include "debruijn/de_bruijn_construct_engine.hpp"
#include "debruijn/de_bruijn_nodes_distributed.hpp"
#include "mxx/sort.hpp"
#include "mxx/comm.hpp"
#include "mxx/reduction.hpp"
#include "utils/kmer_utils.hpp"


namespace dbgp
{
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


      /**
       * @brief       populates edge list vectors
       * @param[in]   minKmerFreq
       * @param[in]   fileName
       * @param[out]  edgeList_1, edgeList_2, and edgeList_3
       */
      template <typename ERT>
      //edgeList_1 - Edge list for junction nodes including interface edges
      //edgeList_2 - Edge list for non-junction nodes excluding interface edges
      //edgeList_3 - Edge list for interface edges
      void populateEdgeList( std::vector< ERT > &edgeList_1,
                             std::vector< ERT > &edgeList_2,
                             std::vector< ERT > &edgeList_3,
                             std::uint64_t &minKmerFreq, std::string &fileName,
                             std::string &filteredKmersFile,
                             const mxx::comm &comm) {
       dbgp::Timer timer;
       dbgp::weightType edge_weight_missing = std::numeric_limits<dbgp::weightType>::max();

        //Declare an edge list vector to save edges temporarily
        //   bool = false if junction node-edge.
        //   bool = true if non-junction node-edge
        //   bool is captured using (second) kmerType[62] out of kmerType[63:0]
        std::vector< ERT > tmpEdgeList;
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
        uint64_t jnnNodeCount = 0, nonNodeCount = 0;
        uint64_t localWorkLoad, minLoad, maxLoad, avgLoad;
#endif

{//Start scope for idx
#if !NDEBUG
        std::size_t selfEdgeCount = 0, pairEdgeCount = 0;
#endif
        bool isPairEdge;

        //Initialize the map
       bliss::de_bruijn::de_bruijn_engine<dbgp::NodeMapType> idx(comm);

        //Build the de Bruijn graph as distributed map
       idx.template build_posix<dbgp::SeqParser, bliss::io::SequencesIterator>(fileName, comm);
       timer.end_section("finished building dB graph using BLISS");

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
//          if ((s == 2182137596314813776) || (s == 3434791461203729892)) {
//            LOG(INFO) << "Rank : " << comm.rank() << " s : " << bliss::utils::KmerUtils::toASCIIString(sourceKmer) << " s : " << s;
//            LOG(INFO) << "Rank : " << comm.rank() << " " << *it;
//          }


          //Filter out low frequency (k+1)-mers
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vGoodInNbrs);
          for(auto &e : vInNbrs) {
              auto d = minKmer(e.first).getData()[0];
//              if ((s == 2182137596314813776) || (s == 3434791461203729892)) {
//                auto tdist = std::distance(idx.cbegin(), it);
//                LOG(INFO) << comm.rank() << " in " << tdist << " u: " << s << " v: " << d << " ew: " << e.second;
//                LOG(INFO) << comm.rank() << " in " << tdist << " u: " << bliss::utils::KmerUtils::toASCIIString(sourceKmer) << " v: " << bliss::utils::KmerUtils::toASCIIString(e.first);
//              }
              if (s == d) {
//                LOG(INFO) << "Rank : " << comm.rank() << "  --> Self edge : " << s;
#if !NDEBUG
                selfEdgeCount++;
#endif
                continue;
              }
//              if (d == 546780989349264598) {// || s == 258550613331770582) {
//                LOG(INFO) << comm.rank() << " s: " << s << " a-d-i: " << d << " w: " << (uint64_t)e.second;
//              }

            if ((uint64_t)e.second >= minKmerFreq) {
              vGoodInNbrs.emplace_back(e);
            }
          }
          std::vector<std::pair<kmerType, typename edgeCountInfoType::CountType>>().swap(vGoodOutNbrs);
          for(auto &e : vOutNbrs) {
              auto d = minKmer(e.first).getData()[0];
//              if ((s == 2182137596314813776) || (s == 3434791461203729892)) {
//                auto tdist = std::distance(idx.cbegin(), it);
//                LOG(INFO) << comm.rank() << " ou " << tdist << " u: " << s << " v: " << d << " ew: " << e.second;
//                LOG(INFO) << comm.rank() << " ou " << tdist << " u: " << bliss::utils::KmerUtils::toASCIIString(sourceKmer) << " v: " << bliss::utils::KmerUtils::toASCIIString(e.first);
//              }
              if (s == d) {
//                LOG(INFO) << "Rank : " << comm.rank() << "  --> Self edge : " << s;
#if !NDEBUG
                selfEdgeCount++;
#endif
                continue;
              }
//              if (d == 546780989349264598) {// || s == 258550613331770582) {
//                LOG(INFO) << comm.rank() << " s: " << s << " a-d-o: " << d << " w: " << (uint64_t)e.second;
//              }

            if ((uint64_t)e.second >= minKmerFreq) {
              isPairEdge = 0;
              for(auto idxitr = vGoodInNbrs.begin(); idxitr != vGoodInNbrs.end(); idxitr++) {
                if (minKmer(idxitr->first).getData()[0] == d) {
#if !NDEBUG
                    assert (isPairEdge == 0);
                    pairEdgeCount++;
#endif
                    isPairEdge = 1;
                    idxitr->second += e.second;
                }
              }
              if (!isPairEdge) {
                vGoodOutNbrs.emplace_back(e);
              }
            }
          }
/*
          if (s == 546780989349264598 || s == 471797961708590738) {
            for(auto &e : vInNbrs) {
                auto d = minKmer(e.first).getData()[0];
                LOG(INFO) << comm.rank() << " s: " << s << " a-d-i: " << d << " w: " << (uint64_t)e.second;
            }
            for(auto &e : vOutNbrs) {
                auto d = minKmer(e.first).getData()[0];
                LOG(INFO) << comm.rank() << " s: " << s << " a-d-o: " << d << " w: " << (uint64_t)e.second;
            }
            for(auto &e : vGoodInNbrs) {
                auto d = minKmer(e.first).getData()[0];
                LOG(INFO) << comm.rank() << " s: " << s << " g-d-i: " << d << " w: " << (uint64_t)e.second;
            }
            for(auto &e : vGoodOutNbrs) {
                auto d = minKmer(e.first).getData()[0];
                LOG(INFO) << comm.rank() << " s: " << s << " g-d-o: " << d << " w: " << (uint64_t)e.second;
            }
            LOG(INFO) << " ";
          }
*/

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

          //Distinguish between junction and non-junction nodes (edges)
          //Non-junction nodes (edges) undergo chain compaction
          //bool = false if junction node-edge. bool=true if non-junction node-edge
          //bool is captured using (second) kmerType[62] out of kmerType[63:0]
          if (vGoodInNbrs.size() == 0 && vGoodOutNbrs.size() == 0) {
            
#if !NDEBUG
            if (!filteredKmersFile.empty()) {
                //Artifact of filtering out low frequency (k+1)-mers
                delKmersList.emplace_back(s);
            }
#endif
            delKmersCount++;
            
          } else if ((vGoodInNbrs.size() == 0 && vGoodOutNbrs.size() == 1) ||
            (vGoodInNbrs.size() == 1 && vGoodOutNbrs.size() == 0) ||
            (vGoodInNbrs.size() == 1 && vGoodOutNbrs.size() == 1)) {
            //Non-junction node
#if !NDEBUG
            nonNodeCount++;
#endif
            //Push the edges to our edgeList
            for(auto &e : vGoodInNbrs) {
                auto d = minKmer(e.first).getData()[0];
//#if !NDEBUG
//            assert (d & 0x4000000000000000 == 0);
//#endif
//                //Generate bool = true using bit manipulation. Set (second) kmerType[62]
//                d = d | 0x4000000000000000;
                tmpEdgeList.emplace_back(s, d, e.second);
            }

            //Same procedure for the outgoing edges
            for(auto &e : vGoodOutNbrs) {
              auto d = minKmer(e.first).getData()[0];
//#if !NDEBUG
//            assert (d & 0x4000000000000000 == 0);
//#endif
//              //Generate bool = true using bit manipulation. Set (second) kmerType[62]
//              d = d | 0x4000000000000000;
              tmpEdgeList.emplace_back(s, d, e.second);
            }
          } else {
            //Junction node
#if !NDEBUG
            jnnNodeCount++;
#endif
            //Push the edges to our edgeList
            for(auto &e : vGoodInNbrs) {
              auto d = minKmer(e.first).getData()[0];
//#if !NDEBUG
//            assert (s & 0x4000000000000000 == 0);
//#endif
              //Flip the edge corresponding to junction node
              tmpEdgeList.emplace_back(d, s, edge_weight_missing);
              //Add edge to edgeList_1
              edgeList_1.emplace_back(s, d, e.second);
//              if ((s == 2182137596314813776) && (d == 3434791461203729892)) {
//                auto tdist = std::distance(idx.cbegin(), it);
//                LOG(INFO) << comm.rank() << " in " << tdist << " u: " << s << " v: " << d << " ew: " << e.second;
//                LOG(INFO) << comm.rank() << " in " << tdist << " u: " << bliss::utils::KmerUtils::toASCIIString(sourceKmer) << " v: " << bliss::utils::KmerUtils::toASCIIString(e.first);
//              }
            }

            //Same procedure for the outgoing edges
            for(auto &e : vGoodOutNbrs) {
              auto d = minKmer(e.first).getData()[0];
//#if !NDEBUG
//            assert (s & 0x4000000000000000 == 0);
//#endif
              //Flip the edge corresponding to junction node
              tmpEdgeList.emplace_back(d, s, edge_weight_missing);
              //Add edge to edgeList_1
              edgeList_1.emplace_back(s, d, e.second);
//              if ((d == 2182137596314813776) && (s == 3434791461203729892)) {
//                auto tdist = std::distance(idx.cbegin(), it);
//                LOG(INFO) << comm.rank() << " ou " << tdist << " u: " << s << " v: " << d << " ew: " << e.second;
//                LOG(INFO) << comm.rank() << " ou " << tdist << " u: " << bliss::utils::KmerUtils::toASCIIString(sourceKmer) << " v: " << bliss::utils::KmerUtils::toASCIIString(e.first);
//              }
            }
          }
        }
#if !NDEBUG
        selfEdgeCount = mxx::reduce(selfEdgeCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "selfEdgeCount : " << selfEdgeCount;
        pairEdgeCount = mxx::reduce(pairEdgeCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "pairEdgeCount : " << pairEdgeCount;
#endif
}//End scope for idx
        //Compact edgeList_1 and tmpEdgeList
        std::vector<ERT>(edgeList_1).swap(edgeList_1);
        std::vector<ERT>(tmpEdgeList).swap(tmpEdgeList);
        std::vector<uint64_t>(delKmersList).swap(delKmersList);
        timer.end_section("finished populating edgeList_1");
        
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
        //jnnNodeCount = 0, nonNodeCount = 0;
        jnnNodeCount = mxx::reduce(jnnNodeCount, 0, comm);
        nonNodeCount = mxx::reduce(nonNodeCount, 0, comm);
        
        LOG_IF(comm.rank() == 0, INFO) << "Junction node count     : " << jnnNodeCount;
        LOG_IF(comm.rank() == 0, INFO) << "Non-junction node count : " << nonNodeCount;
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
        localWorkLoad = edgeList_1.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>() , comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>() , comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of edgeList_1. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
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

        auto edge_compare = [](const ERT& x,
                               const ERT& y){
            return (x.vertexU < y.vertexU) ||
                   ((x.vertexU == y.vertexU) && (x.vertexV < y.vertexV)) ||
                   ((x.vertexU == y.vertexU) && (x.vertexV == y.vertexV) && (x.edgeWeight < y.edgeWeight));

/*
            if (x.vertexU < y.vertexU) {
                return true;
            } else if (x.vertexU > y.vertexU) {
                return false;
            } else {
                if ((x.vertexV & 0x3FFFFFFFFFFFFFFF) == (y.vertexV & 0x3FFFFFFFFFFFFFFF)) {
                    return (x.vertexV < y.vertexV);
                } else {
                    return ((x.vertexV & 0x3FFFFFFFFFFFFFFF) < (y.vertexV & 0x3FFFFFFFFFFFFFFF));
                }
            }
*/
        };
        //Sort tmpEdgeList and process the sorted list
        mxx::sort(tmpEdgeList.begin(), tmpEdgeList.end(),
                  edge_compare, comm);
/*
        for (auto titr = edgeList_1.begin(); titr != edgeList_1.end(); titr++) {
            if (titr->vertexU == 546780989349264598) {
                auto tdist = std::distance(edgeList_1.begin(), titr);
                LOG(INFO) << tdist << " es: " << titr->vertexU << " ed: " << titr->vertexV << " w: " << titr->edgeWeight;
            }
            if (titr->vertexV == 546780989349264598) {
                auto tdist = std::distance(edgeList_1.begin(), titr);
                LOG(INFO) << tdist << " ed: " << titr->vertexV << " es: " << titr->vertexU << " w: " << titr->edgeWeight;
            }
            if (titr->vertexU == 471797961708590738) {
                auto tdist = std::distance(edgeList_1.begin(), titr);
                LOG(INFO) << tdist << " es: " << titr->vertexU << " ed: " << titr->vertexV << " w: " << titr->edgeWeight;
            }
            if (titr->vertexV == 471797961708590738) {
                auto tdist = std::distance(edgeList_1.begin(), titr);
                LOG(INFO) << tdist << " ed: " << titr->vertexV << " es: " << titr->vertexU << " w: " << titr->edgeWeight;
            }
        }
        for (auto titr = tmpEdgeList.begin(); titr != tmpEdgeList.end(); titr++) {
            if (titr->vertexU == 546780989349264598) {
                auto tdist = std::distance(tmpEdgeList.begin(), titr);
                LOG(INFO) << tdist << " ts: " << titr->vertexU << " td: " << titr->vertexV << " w: " << titr->edgeWeight;
            }
            if (titr->vertexV == 546780989349264598) {
                auto tdist = std::distance(tmpEdgeList.begin(), titr);
                LOG(INFO) << tdist << " td: " << titr->vertexV << " ts: " << titr->vertexU << " w: " << titr->edgeWeight;
            }
            if (titr->vertexU == 471797961708590738) {
                auto tdist = std::distance(tmpEdgeList.begin(), titr);
                LOG(INFO) << tdist << " ts: " << titr->vertexU << " td: " << titr->vertexV << " w: " << titr->edgeWeight;
            }
            if (titr->vertexV == 471797961708590738) {
                auto tdist = std::distance(tmpEdgeList.begin(), titr);
                LOG(INFO) << tdist << " td: " << titr->vertexV << " ts: " << titr->vertexU << " w: " << titr->edgeWeight;
            }
        }
*/
/*
        //Send the last tuple on the current processor to the processor's right neighbor
        ERT tmpTuple              = tmpEdgeList.back();
        ERT lastTupleFromLeftProc = mxx::right_shift(tmpTuple, comm);

        ERT prevTuple, currTuple, pushTuple;
        auto it_1 = tmpEdgeList.begin();
        if (comm.rank() != 0) {
          prevTuple = lastTupleFromLeftProc;
        } else{
          prevTuple = *it_1;
          it_1++;
        }
*/

/*
        //Validate that ...
#if !NDEBUG
        uint32_t valCnt = 1;
        uint64_t cntValCnt = 0;
        auto it_save = tmpEdgeList.begin();
        for(; it_1 != tmpEdgeList.end(); it_1++) {
          currTuple = *it_1;
//          //A junction node is followed by a non-junction node when
//          if (prevTuple.vertexU == currTuple.vertexU) {
//            if ((prevTuple.vertexV & 0x4000000000000000) == 0) {
//              assert ((currTuple.vertexV & 0x4000000000000000) == 0x4000000000000000);
//            }
//          }
          if (prevTuple.vertexU == currTuple.vertexU) {
            valCnt++;
          } else {
            if (valCnt > 4) {
                LOG(INFO) << "valCnt : " << valCnt;
                for (; it_save != it_1; it_save++) {
                    if ((*it_save).vertexV & 0x4000000000000000 == 0) {
                        LOG(INFO) << "jnn Edge";
                    } else {
                        LOG(INFO) << "non Edge";
                    }
                }
                cntValCnt++;
            }
            valCnt = 1;
            it_save = it_1;
          }
//          //There are NO more than 4 (consecutive) nodes with identical vertexU
//          assert (valCnt <= 4);
          prevTuple = currTuple;
        }
        cntValCnt = mxx::reduce(cntValCnt, 0, comm);
        LOG_IF(comm.rank() == 0, INFO) << "cntValCnt : " << cntValCnt;

        //Reset the iterators
        it_1 = tmpEdgeList.begin();
        if (comm.rank() != 0) {
          prevTuple = lastTupleFromLeftProc;
        } else{
          prevTuple = *it_1;
          it_1++;
        }
#endif
*/

/*
        for(; it_1 != tmpEdgeList.end(); it_1++) {
          currTuple = *it_1;
          if ((currTuple.vertexV & 0x4000000000000000) == 0x4000000000000000) {
              //currTuple corresponds to non-junction node
              if ((prevTuple.vertexV & 0x4000000000000000) == 0) {
                //prevTuple corresponds to junction node
                //Check if currTuple == prevTuple
                if ((currTuple.vertexU == prevTuple.vertexU) &&
                    (currTuple.vertexV & 0x3FFFFFFFFFFFFFFF == prevTuple.vertexV)) {
                  //currTuple corresponds to interface edge
                  //Assert currTuple.edgeWeight == prevTuple.edgeWeight
                  //edgeList_3 is NOT being used
                  //edgeList_3.emplace_back(); //Add to list of interface edges
                } else {
                //Reset (second) kmerType[62]
                pushTuple = currTuple;
                pushTuple.vertexV = pushTuple.vertexV & 0x3FFFFFFFFFFFFFFF;
                edgeList_2.emplace_back(pushTuple.vertexU, pushTuple.vertexV, pushTuple.edgeWeight); //Add to list of non-junction
                                                 //nodes excluding interface edges
                }
              } else {
                //prevTuple also corresponds to non-junction node
                //Reset (second) kmerType[62]
                pushTuple = currTuple;
                pushTuple.vertexV = pushTuple.vertexV & 0x3FFFFFFFFFFFFFFF;
                edgeList_2.emplace_back(pushTuple.vertexU, pushTuple.vertexV, pushTuple.edgeWeight); //Add to list of non-junction
                                                 //nodes excluding interface edges
              }
          } else {
            //currTuple corresponds to junction node
            //Nothing to do
          }
          prevTuple = currTuple;
        }
*/
        
        ERT prevTuple, currTuple;
        auto vertuCompare = [&](const ERT& x, const ERT& y){
            return x.vertexU == y.vertexU;
        };
        // shift region
        uint64_t startOffset, endOffset;
        std::vector<ERT> straddleRegion;
        shiftStraddlingRegion(comm, tmpEdgeList, startOffset, endOffset,
                              straddleRegion, vertuCompare);
        
        //Validate that ...
        auto it_1 = tmpEdgeList.begin() + startOffset;
#if !NDEBUG
        uint32_t valCnt = 1;
        prevTuple = *it_1;
        it_1++;
        for(; it_1 != tmpEdgeList.begin() + endOffset; it_1++) {
          currTuple = *it_1;
          if ((prevTuple.vertexU == currTuple.vertexU) &&
              (prevTuple.vertexV == currTuple.vertexV)) {
            valCnt++;
          } else {
            valCnt = 1;
          }
          //There are NO more than 2 (consecutive) identical edges
          assert (valCnt <= 2);
          prevTuple = currTuple;
        }
        //There are NO more than 2 (consecutive) identical edges
        assert (valCnt <= 2);

        if (straddleRegion.size() > 1) {
          valCnt = 1;
          it_1 = straddleRegion.begin();
          prevTuple = *it_1;
          it_1++;
          for(; it_1 != straddleRegion.end(); it_1++) {
            currTuple = *it_1;
            if ((prevTuple.vertexU == currTuple.vertexU) &&
                (prevTuple.vertexV == currTuple.vertexV)) {
              valCnt++;
            } else {
              valCnt = 1;
            }
            //There are NO more than 2 (consecutive) identical edges
            assert (valCnt <= 2);
            prevTuple = currTuple;
          }
          //There are NO more than 2 (consecutive) identical edges
          assert (valCnt <= 2);
        }
#endif

/*
        uint32_t sameNodeCount = 1, sameEdgeCount = 1, saveSameEdgeCount = 1;
#if !NDEBUG
        uint64_t specialNodeCountOne = 0;
        uint64_t specialNodeCountTwo = 0;
#endif
        it_1 = tmpEdgeList.begin() + startOffset;
        prevTuple = *it_1;
        it_1++;
        for(; it_1 != tmpEdgeList.begin() + endOffset; it_1++) {
          currTuple = *it_1;
          if ((prevTuple.vertexU == currTuple.vertexU) &&
              (prevTuple.vertexV == currTuple.vertexV)) {
            sameEdgeCount++;
          } else {
            if (sameEdgeCount == 1) {
              edgeList_2.emplace_back(prevTuple.vertexU, prevTuple.vertexV, prevTuple.edgeWeight);
            }
            saveSameEdgeCount = sameEdgeCount;
            sameEdgeCount = 1;
          }
          if (prevTuple.vertexU == currTuple.vertexU) {
            sameNodeCount++;
          } else {
            if (saveSameEdgeCount == 2 && sameNodeCount == 2) {
#if !NDEBUG
        specialNodeCountOne++;
#endif
            }
            if (saveSameEdgeCount == 2 && sameNodeCount == 4) {
#if !NDEBUG
        specialNodeCountTwo++;
#endif
            }
            saveSameEdgeCount = 1;
            sameNodeCount = 1;
          }
          prevTuple = currTuple;
        }
*/

        uint32_t sameEdgeCount = 1;
        uint32_t sameNodeCount = 1;
        std::vector<ERT> saveSpecialEdges;
        it_1 = tmpEdgeList.begin() + startOffset;
        prevTuple = *it_1;
        it_1++;
        for(; it_1 != tmpEdgeList.begin() + endOffset; it_1++) {
          currTuple = *it_1;
          if ((prevTuple.vertexU == currTuple.vertexU) &&
              (prevTuple.vertexV == currTuple.vertexV)) {
            sameEdgeCount++;
            assert (prevTuple.edgeWeight != edge_weight_missing);
            saveSpecialEdges.emplace_back(prevTuple.vertexU, prevTuple.vertexV, prevTuple.edgeWeight);
          } else {
            if ((sameEdgeCount == 1) && (prevTuple.edgeWeight != edge_weight_missing)) {
              edgeList_2.emplace_back(prevTuple.vertexU, prevTuple.vertexV, prevTuple.edgeWeight);
            }
            sameEdgeCount = 1;
          }
          if (prevTuple.vertexU == currTuple.vertexU) {
            sameNodeCount++;
          } else {
            if (sameNodeCount == (2 * saveSpecialEdges.size())) {
                for(auto sitr = saveSpecialEdges.begin(); sitr != saveSpecialEdges.end(); sitr++) {
                    edgeList_1.emplace_back(sitr->vertexU, sitr->vertexV, sitr->edgeWeight);
                }
            }
            std::vector<ERT>().swap(saveSpecialEdges);
            sameNodeCount = 1;
          }
          prevTuple = currTuple;
        }
        if ((sameEdgeCount == 1) && (prevTuple.edgeWeight != edge_weight_missing)) {
          edgeList_2.emplace_back(prevTuple.vertexU, prevTuple.vertexV, prevTuple.edgeWeight);
        }
            if (sameNodeCount == (2 * saveSpecialEdges.size())) {
                for(auto sitr = saveSpecialEdges.begin(); sitr != saveSpecialEdges.end(); sitr++) {
                    edgeList_1.emplace_back(sitr->vertexU, sitr->vertexV, sitr->edgeWeight);
                }
            }
            std::vector<ERT>().swap(saveSpecialEdges);
            sameNodeCount = 1;

        if (straddleRegion.size() < 1) {
          //Do nothing
        } else if (straddleRegion.size() == 1) {
          assert (0);
        } else {
          sameEdgeCount = 1;
          it_1 = straddleRegion.begin();
          prevTuple = *it_1;
          it_1++;
          for(; it_1 != straddleRegion.end(); it_1++) {
            currTuple = *it_1;
            if ((prevTuple.vertexU == currTuple.vertexU) &&
                (prevTuple.vertexV == currTuple.vertexV)) {
              sameEdgeCount++;
            assert (prevTuple.edgeWeight != edge_weight_missing);
            saveSpecialEdges.emplace_back(prevTuple.vertexU, prevTuple.vertexV, prevTuple.edgeWeight);
            } else {
              if ((sameEdgeCount == 1) && (prevTuple.edgeWeight != edge_weight_missing)) {
                edgeList_2.emplace_back(prevTuple.vertexU, prevTuple.vertexV, prevTuple.edgeWeight);
              }
              sameEdgeCount = 1;
            }
          if (prevTuple.vertexU == currTuple.vertexU) {
            sameNodeCount++;
          } else {
            if (sameNodeCount == (2 * saveSpecialEdges.size())) {
                for(auto sitr = saveSpecialEdges.begin(); sitr != saveSpecialEdges.end(); sitr++) {
                    edgeList_1.emplace_back(sitr->vertexU, sitr->vertexV, sitr->edgeWeight);
                }
            }
            std::vector<ERT>().swap(saveSpecialEdges);
            sameNodeCount = 1;
          }
            prevTuple = currTuple;
          }
          if ((sameEdgeCount == 1) && (prevTuple.edgeWeight != edge_weight_missing)) {
            edgeList_2.emplace_back(prevTuple.vertexU, prevTuple.vertexV, prevTuple.edgeWeight);
          }
            if (sameNodeCount == (2 * saveSpecialEdges.size())) {
                for(auto sitr = saveSpecialEdges.begin(); sitr != saveSpecialEdges.end(); sitr++) {
                    edgeList_1.emplace_back(sitr->vertexU, sitr->vertexV, sitr->edgeWeight);
                }
            }
            std::vector<ERT>().swap(saveSpecialEdges);
            sameNodeCount = 1;
        }        
        //Compact edgeList_2 and edgeList_1
        std::vector<ERT>(edgeList_2).swap(edgeList_2);
        std::vector<ERT>(edgeList_1).swap(edgeList_1);
        mxx::distribute_inplace(edgeList_1, comm); // distribute equally
        mxx::distribute_inplace(edgeList_2, comm); // distribute equally
/*        
        for(auto eitr = edgeList_1.begin(); eitr != edgeList_1.end(); eitr++) {
              if ((eitr->vertexU == 2182137596314813776) || (eitr->vertexU == 3434791461203729892)) {
                auto tdist = std::distance(edgeList_1.begin(), eitr);
                LOG(INFO) << comm.rank() << " e1 " << tdist << " u: " << eitr->vertexU << " v: " << eitr->vertexV << " ew: " << eitr->edgeWeight;
              }
        }
        for(auto eitr = edgeList_2.begin(); eitr != edgeList_2.end(); eitr++) {
              if ((eitr->vertexU == 2182137596314813776) || (eitr->vertexU == 3434791461203729892)) {
                auto tdist = std::distance(edgeList_2.begin(), eitr);
                LOG(INFO) << comm.rank() << " e2 " << tdist << " u: " << eitr->vertexU << " v: " << eitr->vertexV << " ew: " << eitr->edgeWeight;
              }
              if ((eitr->vertexV == 2182137596314813776) || (eitr->vertexV == 3434791461203729892)) {
                auto tdist = std::distance(edgeList_2.begin(), eitr);
                LOG(INFO) << comm.rank() << " e2 " << tdist << " u: " << eitr->vertexU << " v: " << eitr->vertexV << " ew: " << eitr->edgeWeight;
              }
        }
*/

#if !NDEBUG
/*
        specialNodeCountOne = mxx::reduce(specialNodeCountOne, 0, comm);
        specialNodeCountTwo = mxx::reduce(specialNodeCountTwo, 0, comm);
        LOG_IF(comm.rank() == 0, INFO) << "specialNodeCountOne : " << specialNodeCountOne;
        LOG_IF(comm.rank() == 0, INFO) << "specialNodeCountTwo : " << specialNodeCountTwo;
*/
        localWorkLoad = edgeList_1.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>() , comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>() , comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of edgeList_1. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
        localWorkLoad = edgeList_2.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>() , comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>() , comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of edgeList_2. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
        localWorkLoad = edgeList_3.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>() , comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>() , comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of edgeList_3. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
        localWorkLoad = delKmersList.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>() , comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>() , comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of delKmersList. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif
/*
        for (auto titr = edgeList_2.begin(); titr != edgeList_2.end(); titr++) {
            if (titr->vertexU == 546780989349264598) {
                auto tdist = std::distance(edgeList_2.begin(), titr);
                LOG(INFO) << tdist << " es: " << titr->vertexU << " ed: " << titr->vertexV << " w: " << titr->edgeWeight;
            }
            if (titr->vertexV == 546780989349264598) {
                auto tdist = std::distance(edgeList_2.begin(), titr);
                LOG(INFO) << tdist << " ed: " << titr->vertexV << " es: " << titr->vertexU << " w: " << titr->edgeWeight;
            }
            if (titr->vertexU == 471797961708590738) {
                auto tdist = std::distance(edgeList_2.begin(), titr);
                LOG(INFO) << tdist << " es: " << titr->vertexU << " ed: " << titr->vertexV << " w: " << titr->edgeWeight;
            }
            if (titr->vertexV == 471797961708590738) {
                auto tdist = std::distance(edgeList_2.begin(), titr);
                LOG(INFO) << tdist << " ed: " << titr->vertexV << " es: " << titr->vertexU << " w: " << titr->edgeWeight;
            }
        }
*/
        timer.end_section("finished populating edgeList_2");
      }

    };
  }
}

#endif
