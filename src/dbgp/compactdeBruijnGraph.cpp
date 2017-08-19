
#include <vector>
#include "mxx/comm.hpp"
#include "utils/mpi_utils.hpp"
#include "utils/io_utils.hpp"
#include "dbgp/recordTypes.hpp"

#include "coloring/labelProp_utils.hpp"
#include "coloring/labelProp.hpp"

#include "dbgp/deBruijnGraphGen.hpp"

namespace dbgp{

    static dbgp::colorType NODE_COLOR_MISSING;
    static dbgp::vertexIdType NODE_ID_MISSING;
    static dbgp::weightType NODE_WEIGHT_MISSING;

    using CCLTupleIds = conn::coloring::cclTupleIds;
    // init
    //   Initialize MISSING color, vertex and weight static variables
    //
    void compactInit(){
        // The limits of type
       NODE_COLOR_MISSING = std::numeric_limits<dbgp::colorType>::max();
       NODE_ID_MISSING = std::numeric_limits<dbgp::vertexIdType>::max();
       NODE_WEIGHT_MISSING = std::numeric_limits<dbgp::weightType>::max();
    }

/*
    // updateColorIds
    //   Updates the color value to the min. node id having the same color
    // nodeColors : tuples output by the CCL algorithm
    void updateColorIds(mxx::comm& comm,
                        std::vector<dbgp::cclNodeType>& nodeColors){
        comm.with_subset(
            nodeColors.begin() != nodeColors.end(), [&](const mxx::comm& comm){
                // if not sorted by color, sort by color
                mxx::sort(nodeColors.begin(), nodeColors.end(),
                          conn::utils::TpleComp2Layers<
                              CCLTupleIds::Pc, CCLTupleIds::nId>(), comm);
            });

        comm.with_subset(
            nodeColors.begin() != nodeColors.end(), [&](const mxx::comm& comm){
                // update color ids to the node with the least value
                // -- get the shifted values from prev processor
                auto lastColor = std::get<CCLTupleIds::Pc>(nodeColors.back());
                auto lastUpdNode = std::get<CCLTupleIds::nId>(nodeColors.back());
                for(auto xitr = nodeColors.rbegin(); xitr != nodeColors.rend();
                    xitr++){
                    if(lastColor != std::get<CCLTupleIds::Pc>(nodeColors.back()))
                        break;
                    lastUpdNode = std::get<CCLTupleIds::nId>(nodeColors.back());
                }
                auto shiftColor = mxx::right_shift(lastColor, comm); // TODO: is shift ok ?
                auto shiftUpdNode = mxx::right_shift(lastUpdNode, comm); // TODO: is shift ok ?

                // -- Update the color id to the min. node id of the same color
                auto prevColor = shiftColor;
                auto prevUpdNode = shiftUpdNode;
                if(comm.rank() == 0){
                    prevColor = std::get<CCLTupleIds::Pc>(nodeColors.front());
                    prevUpdNode = std::get<CCLTupleIds::nId>(nodeColors.front());
                }
                for(auto ncitr = nodeColors.begin();ncitr != nodeColors.end(); ncitr++){
                    if(std::get<CCLTupleIds::Pc>(*ncitr) == prevColor){
                        std::get<CCLTupleIds::Pc>(*ncitr) = prevUpdNode;
                    } else {
                        prevColor = std::get<CCLTupleIds::Pc>(*ncitr);
                        prevUpdNode = std::get<CCLTupleIds::nId>(*ncitr);
                        std::get<CCLTupleIds::Pc>(*ncitr) = prevUpdNode;
                    }
                }
            });
    }
*/

    // runCCL
    //
    // edges (IN) - Input graph as a distributed list of edges
    // nodeColors (OUT) - Tuples values returned by CCL algorithm
    void runCCL(mxx::comm& comm,
                std::vector<dbgp::edgeRecordType>& edges,
                std::vector<dbgp::cclNodeType>& nodeColors,
                std::vector<dbgp::colorType>& graphColors) {
        dbgp::Timer timer;

        dbgp::cclType cclInstance(edges, comm);
        comm.with_subset(edges.size() > 0, [&](const mxx::comm& comm){
                //std::vector< std::tuple< dbgp::vertexIdType,
                //                         dbgp::vertexIdType,
                //                         dbgp::weightType > >
                //    edgesTuples(edges.size());
                //for(std::size_t i = 0; i < edges.size();i++)
                //    edgesTuples[i] = std::make_tuple(edges[i].vertexU,
                //                                     edges[i].vertexV,
                //                                     edges[i].edgeWeight);
                //dbgp::cclType cclInstance(edgesTuples, comm);
                cclInstance.compute(nodeColors);
            });

        //ccl already does this
        //// update the color to min node id
        //updateColorIds(comm, nodeColors);
        
#if !NDEBUG
        uint64_t localWorkLoad, minLoad, maxLoad, avgLoad;
        localWorkLoad = edges.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>() , comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>() , comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of nonEdges. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
        localWorkLoad = nodeColors.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>() , comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>() , comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of nonNodeColors. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
        
        //Validate that ...
        for(auto xitr = nodeColors.begin(); xitr != nodeColors.end(); xitr++) {
            //Color corresponds to minimum ID
            assert (std::get<CCLTupleIds::Pc>(*xitr) <= std::get<CCLTupleIds::nId>(*xitr));
//            //Pc and Pn are identical
//            assert (std::get<CCLTupleIds::Pc>(*xitr) == std::get<CCLTupleIds::Pn>(*xitr));
        }
        
        //TODO: For below computation, we assume that a chain spans across at most 2 processors
        std::size_t componentCount = cclInstance.computeComponentCount();
        LOG_IF(comm.rank() == 0, INFO) << "CCL chain count: " << componentCount;
        std::vector<uint64_t> chainCounts; //log scale
        chainCounts.resize(32);
        comm.with_subset(nodeColors.begin() !=  nodeColors.end(), [&](const mxx::comm& comm) {
            mxx::sort(nodeColors.begin(), nodeColors.end(), conn::utils::TpleComp2Layers<CCLTupleIds::Pc, CCLTupleIds::nId>(), comm);
            auto xitr = nodeColors.begin();
            dbgp::colorType currColor, prevColor, saveColor;
            dbgp::vertexIdType currNode, prevNode, saveNode;
            prevColor = std::get<CCLTupleIds::Pc>(*xitr);
            prevNode  = std::get<CCLTupleIds::nId>(*xitr);
            saveColor = prevColor;
            saveNode  = prevNode;
            xitr++;
            uint32_t saveChainLength = 1;
            //Defer processing the first chain/color
            for(; xitr != nodeColors.end(); xitr++) {
                currColor = std::get<CCLTupleIds::Pc>(*xitr);
                currNode  = std::get<CCLTupleIds::nId>(*xitr);
                if (currColor != prevColor) {
                    break;
                } else {
                    if (currNode != prevNode) {
                        saveChainLength++;
                    }
                }
                prevColor = currColor;
                prevNode  = currNode;
            }
            prevColor = currColor; xitr++;
            prevNode  = currNode;
            uint32_t currChainLength = 1;
            //Process subsequent chains/colors
            for(; xitr != nodeColors.end(); xitr++) {
                currColor = std::get<CCLTupleIds::Pc>(*xitr);
                currNode  = std::get<CCLTupleIds::nId>(*xitr);
                if (currColor != prevColor) {
                    //Chain length must be at least 2
                    assert ((uint32_t) std::log2(currChainLength) > 0);
                    assert ((uint32_t) std::log2(currChainLength) < 32);
                    (chainCounts[(uint32_t) std::log2(currChainLength)])++;
                    currChainLength = 1;
                    graphColors.emplace_back(prevColor);
                } else {
                    if (currNode != prevNode) {
                        currChainLength++;
                    }
                }
                prevColor = currColor;
                prevNode  = currNode;
            }
            //Process first chain/color
            dbgp::colorType lastColorFromLeftProc = mxx::right_shift(currColor, comm);
            dbgp::vertexIdType lastNodeFromLeftProc = mxx::right_shift(currNode, comm);
            uint32_t lastLengthFromLeftProc = mxx::right_shift(currChainLength, comm);
            if (comm.rank() == (comm.size() -1)) {
                assert ((uint32_t) std::log2(currChainLength) > 0);
                assert ((uint32_t) std::log2(currChainLength) < 32);
                (chainCounts[(uint32_t) std::log2(currChainLength)])++;
                graphColors.emplace_back(currColor);
            }
            if (comm.rank() == 0) {
                lastColorFromLeftProc = NODE_COLOR_MISSING;
                lastNodeFromLeftProc = NODE_ID_MISSING;
                lastLengthFromLeftProc = 0;
            }
            if (lastColorFromLeftProc == saveColor) {
                saveChainLength += lastLengthFromLeftProc;
                if (lastNodeFromLeftProc == saveNode) {
                    saveChainLength--;
                }
                assert ((uint32_t) std::log2(saveChainLength) > 0);
                assert ((uint32_t) std::log2(saveChainLength) < 32);
                (chainCounts[(uint32_t) std::log2(saveChainLength)])++;
                graphColors.emplace_back(saveColor);
            } else {
                assert ((uint32_t) std::log2(saveChainLength) > 0);
                assert ((uint32_t) std::log2(saveChainLength) < 32);
                (chainCounts[(uint32_t) std::log2(saveChainLength)])++;
                graphColors.emplace_back(saveColor);
                if (comm.rank() != 0) {
                    assert ((uint32_t) std::log2(lastLengthFromLeftProc) > 0);
                    assert ((uint32_t) std::log2(lastLengthFromLeftProc) < 32);
                    (chainCounts[(uint32_t) std::log2(lastLengthFromLeftProc)])++;
                    graphColors.emplace_back(lastColorFromLeftProc);
                }
            }
            
            std::vector<dbgp::colorType>(graphColors).swap(graphColors);
            dbgp::colorType firstColorFromRightProc = mxx::left_shift(std::get<CCLTupleIds::Pc>(nodeColors.front()), comm);
            if (comm.rank() == (comm.size() - 1)) {
                firstColorFromRightProc = NODE_COLOR_MISSING;
            }
            if ((lastColorFromLeftProc == std::get<CCLTupleIds::Pc>(nodeColors.front())) &&
                (std::get<CCLTupleIds::Pc>(nodeColors.front()) == std::get<CCLTupleIds::Pc>(nodeColors.back())) &&
                (std::get<CCLTupleIds::Pc>(nodeColors.back()) == firstColorFromRightProc)){
                LOG(INFO) << "A chain spans across more than 2 processors on rank " << comm.rank();
            }
        });
        chainCounts = mxx::reduce(chainCounts, 0, comm);
        componentCount = 0;
        for(auto xitr = 0; xitr < chainCounts.size(); xitr++) {
            LOG_IF(comm.rank() == 0, INFO) << "Chain counts [" << xitr << "] : " << chainCounts[xitr];
            componentCount += chainCounts[xitr];
        }
        LOG_IF(comm.rank() == 0, INFO) << "DBGP chain count: " << componentCount;
#endif

        timer.end_section("finished coloring nonEdges");
    }

    // emitXferTuples
    //
    // nonEdges  (IN) - edges (u, v) in DBG such that both u and v are non-junction nodes
    // nodeColors (IN) - coloring results from the CCL algorithm
    // colorXferTuples (OUT) - Xfer tuples merging the two inputs
    void emitXferTuples(mxx::comm& comm,
                        std::vector<dbgp::edgeRecordType>& nonEdges,
                        std::vector<dbgp::cclNodeType>& nodeColors,
                        std::vector<dbgp::xferRecordType>& colorXferTuples){
        auto clsize = std::distance(nodeColors.begin(), nodeColors.end());
        auto nlsize = std::distance(nonEdges.begin(), nonEdges.end());

        // Sort by node ids
        // In nodeColors, a node appears as many times as the degree of the node
        // We will collapse all the copies into one instance
        comm.with_subset(nodeColors.begin() !=  nodeColors.end(),
            [&](const mxx::comm& comm){
                mxx::sort(nodeColors.begin(), nodeColors.end(),
                          conn::utils::TpleComp<CCLTupleIds::nId>(), comm);
            });

        comm.with_subset(
            clsize + nlsize > 0, [&](const mxx::comm& comm){
                colorXferTuples.resize(clsize + nlsize);
                auto yitr = colorXferTuples.begin();

                // Shift the last node from the previous processor
                auto lastNode = std::get<CCLTupleIds::nId>(nodeColors.back());
                auto prevNode = mxx::right_shift(lastNode, comm);
                if(comm.rank() == 0) prevNode = NODE_ID_MISSING;

                // (2) Emit <color, node, NA, MISSING>
                for(auto xcp : nodeColors) {
                    if(prevNode == std::get<CCLTupleIds::nId>(xcp))
                        continue;
                    (*yitr).set(std::get<CCLTupleIds::Pc>(xcp),
                                std::get<CCLTupleIds::nId>(xcp),
                                NODE_ID_MISSING, NODE_WEIGHT_MISSING);
                    prevNode = std::get<CCLTupleIds::nId>(xcp);
                    yitr++;
                }

                // (3) Emit <MISSING, u, v, weight> tuple
                for(auto ned : nonEdges){
                    (*yitr).set(NODE_COLOR_MISSING,
                                ned.vertexU, ned.vertexV, ned.edgeWeight);
                    yitr++;
                }
                if(colorXferTuples.begin() != yitr){
                    auto rsize = std::distance(colorXferTuples.begin(), yitr);
                    colorXferTuples.resize(rsize);
                    // compact this vector
                    std::vector<dbgp::xferRecordType>(colorXferTuples).swap(colorXferTuples);
                }
            });
    }

    void writeColorKmerMapping(mxx::comm& comm,
                               std::vector<dbgp::cclNodeType>& nodeColors,
                               std::string t1FileName){
        auto clsize = std::distance(nodeColors.begin(), nodeColors.end());
        //Assume nodeColors is sorted by node ID from previous step
        comm.with_subset(clsize > 0, [&](const mxx::comm& comm){
                // Shift the last node from the previous processor
                auto lastNode = std::get<CCLTupleIds::nId>(nodeColors.back());
                auto prevNode = mxx::right_shift(lastNode, comm);
                if(comm.rank() == 0) prevNode = NODE_ID_MISSING;

                // write to file as "color vertex" pair
                // a vertex/node may appear multiple times in nodeColors
                // we will only write one instance
                std::ofstream outFilePtr(t1FileName);
                for(auto xcp : nodeColors) {
                    if(prevNode == std::get<CCLTupleIds::nId>(xcp))
                        continue;
                    outFilePtr << std::get<CCLTupleIds::Pc>(xcp) << " " << std::get<CCLTupleIds::nId>(xcp) << std::endl;
                    prevNode = std::get<CCLTupleIds::nId>(xcp);
                }
                outFilePtr.close();
            });
    }

    // updateXferColor
    //
    // colorXferTuples (IN/OUT) - Xfer tuples
    void updateXferColor(mxx::comm& comm,
                         std::vector<dbgp::xferRecordType>& colorXferTuples){

        comm.with_subset(
            colorXferTuples.begin() !=  colorXferTuples.end(),
            [&](const mxx::comm& comm){
                // (4) Sort the list by node id and then by color
                // NODE_COLOR_MISSING has the highest possible value
                mxx::sort(colorXferTuples.begin(), colorXferTuples.end(),
                          [](const dbgp::xferRecordType& x,
                             const dbgp::xferRecordType& y){
                              return (x.xferUID < y.xferUID) ||
                                  ((x.xferUID == y.xferUID) &&
                                   (x.xferColor < y.xferColor));
                          },
                          comm);
            });

// Below implementation [due to Sriram] should work correctly as well
/*
        comm.with_subset(colorXferTuples.begin() !=  colorXferTuples.end(),
            [&](const mxx::comm& comm){
                // (5) Update color in (4)

                // (tuples with missing colors are at the end,
                //   since they have MAX values)
                //    - Shift the last non-missing color from prev processor
                auto lastColor = NODE_COLOR_MISSING;
                for(auto xitr = colorXferTuples.rbegin();
                    xitr != colorXferTuples.rend(); xitr++){
                    lastColor = (*xitr).xferColor;
                    if(lastColor != NODE_COLOR_MISSING)
                        break;
                }
                auto shiftColor = mxx::right_shift(lastColor, comm); // TODO: is shift ok ?

                //  - Update, whom ever has the missing color
                auto prevColor = (comm.rank() == 0) ?
                    NODE_COLOR_MISSING : shiftColor;
                for(auto yitr = colorXferTuples.begin();
                    yitr != colorXferTuples.end(); yitr++){
                    auto currColor = (*yitr).xferColor;
                    if((*yitr).xferColor == NODE_COLOR_MISSING){
                        (*yitr).xferColor = prevColor;
                    } else {
                        prevColor = currColor;
                    }
                }
            });
*/

        //Checked below implementation against all possible cases
        comm.with_subset(colorXferTuples.begin() !=  colorXferTuples.end(),
            [&](const mxx::comm& comm){
                // (5) Update color in (4)
                auto currXferRPtr = colorXferTuples.begin();
                auto prevXferRPtr = currXferRPtr;
                currXferRPtr++; //Skip the first element for now
                for(;currXferRPtr != colorXferTuples.end(); currXferRPtr++) {
                    if ((*prevXferRPtr).xferUID == (*currXferRPtr).xferUID &&
                        (*currXferRPtr).xferColor == NODE_COLOR_MISSING) {
                        (*currXferRPtr).xferColor = (*prevXferRPtr).xferColor;
                    }
                    prevXferRPtr = currXferRPtr;
                }

                //Handle the first element (two elements) now
                //There are at most three colorXferTuples with identical xferUID
                //Further, the xferColor of the first (of at most three) colorXferTuple is known
                auto xitr = colorXferTuples.back();
                dbgp::xferRecordType lastXferRecord(xitr.xferColor, xitr.xferUID, xitr.xferVID, xitr.xferWeight);
                dbgp::xferRecordType prevXferRecord = mxx::right_shift(lastXferRecord, comm);
                if (comm.rank() != 0) {
                    currXferRPtr = colorXferTuples.begin();
                    if (prevXferRecord.xferUID == (*currXferRPtr).xferUID &&
                        (*currXferRPtr).xferColor == NODE_COLOR_MISSING) {
                        //Assert (*prevXferRPtr).xferColor != NODE_COLOR_MISSING
                        (*currXferRPtr).xferColor =  prevXferRecord.xferColor;
                    }
                    currXferRPtr++;
                    if (prevXferRecord.xferUID == (*currXferRPtr).xferUID &&
                        (*currXferRPtr).xferColor == NODE_COLOR_MISSING) {
                        //Assert (*prevXferRPtr).xferColor != NODE_COLOR_MISSING
                        (*currXferRPtr).xferColor =  prevXferRecord.xferColor;
                    }
                }
            });
#if !NDEBUG
        for(auto currXferRPtr = colorXferTuples.begin(); currXferRPtr != colorXferTuples.end(); currXferRPtr++) {
            if ((*currXferRPtr).xferColor == NODE_COLOR_MISSING) {
                LOG(INFO) << comm.rank() << " " << std::distance(colorXferTuples.begin(), currXferRPtr)
                          << " " << colorXferTuples.size() << std::endl;
            }
        }
        comm.barrier();
        //xferColor of all colorXferTuples must be valid at this point
        for(auto currXferRPtr = colorXferTuples.begin(); currXferRPtr != colorXferTuples.end(); currXferRPtr++) {
            assert ((*currXferRPtr).xferColor != NODE_COLOR_MISSING);
        }
        std::uint64_t abnormalCount = 0;
        for(auto currXferRPtr = colorXferTuples.begin(); currXferRPtr != colorXferTuples.end(); currXferRPtr++) {
            if (currXferRPtr->xferVID == NODE_ID_MISSING) {
                assert (currXferRPtr->xferWeight == NODE_WEIGHT_MISSING);
            } else {
                if (currXferRPtr->xferWeight < 2) {
                    abnormalCount++;
                }
            }
        }
        abnormalCount = mxx::reduce(abnormalCount, 0, std::plus<std::uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "Abnormal weight count : " << abnormalCount;
#endif
    }

    void updateXferWeights(mxx::comm& comm,
                           std::vector<dbgp::xferRecordType>& colorXferTuples,
                           char dbgNodeWtType){
        comm.with_subset(
            colorXferTuples.begin() !=  colorXferTuples.end(), [&](const mxx::comm& comm){
                // (7) Sort (6) by color and then by weight
                mxx::sort(colorXferTuples.begin(), colorXferTuples.end(),
                          [&](const dbgp::xferRecordType& x,
                             const dbgp::xferRecordType& y){
                              return (x.xferColor < y.xferColor) ||
                                  ((x.xferColor == y.xferColor) &&
                                   (x.xferWeight < y.xferWeight));
                          },
                          comm);
            });

// Below implementation [due to Sriram] has atleast one issue
/*
        comm.with_subset(
            colorXferTuples.begin() !=  colorXferTuples.end(),
            [&](const mxx::comm& comm){
                // (8) Compute cumulative weight
                // (tuples with missing weights are at the end, since they have MAX values)
                dbgp::weightType runWeight = (dbgp::weightType)0;
                auto xitr = colorXferTuples.rbegin();
                for(; xitr != colorXferTuples.rend(); xitr++)
                    if((*xitr).xferWeight < NODE_WEIGHT_MISSING) break;
                //
                for(; xitr != colorXferTuples.rend(); xitr++){
                    auto lastWeight = (*xitr).xferWeight;
                    if(lastWeight == NODE_WEIGHT_MISSING) break;
                    runWeight += lastWeight;
                }
                auto shiftWeight = mxx::right_shift(runWeight, comm); // TODO: is shift ok ?

                //  - Accumulate weights
                runWeight = (comm.rank() ==  0) ?
                    (dbgp::weightType)0 : shiftWeight;
                auto yitr = colorXferTuples.begin();
                auto prevWeight = (*yitr).xferWeight;
                for(;yitr != colorXferTuples.end(); yitr++){
                    auto currWeight = (*yitr).xferWeight;
                    if(currWeight == NODE_WEIGHT_MISSING){
                        (*yitr).xferWeight = runWeight;
                    } else if (prevWeight == NODE_WEIGHT_MISSING){
                        runWeight = currWeight;
                    } else {
                        runWeight += currWeight;
                    }
                    prevWeight = currWeight;
                }
            });
*/

            comm.with_subset(
            colorXferTuples.begin() !=  colorXferTuples.end(),
            [&](const mxx::comm& comm){
                // (8) Compute cumulative weight
                // (tuples with missing weights are at the end, since they have MAX values)
                auto currXferRPtr = colorXferTuples.begin();
                auto prevXferRPtr = currXferRPtr;
/*
                auto saveXferRPtr = currXferRPtr;
                auto xitr = colorXferTuples.back();
                dbgp::xferRecordType lastXferRecord(xitr.xferColor, xitr.xferUID, xitr.xferVID, xitr.xferWeight);
*/
                dbgp::weightType runWeight;
                bool tmpBool = false; //Becomes TRUE on encountering the first element with missing weight
/*
                if ((*currXferRPtr).xferWeight == NODE_WEIGHT_MISSING) {
                    (*currXferRPtr).xferWeight = 0;
                    runWeight = 0;
                    tmpBool = true;
                    saveXferRPtr = currXferRPtr;
                } else {
                    if (dbgNodeWtType == 'c') {
                        runWeight = 1;
                    } else {
                        runWeight = (*currXferRPtr).xferWeight;
                    }
                }
                currXferRPtr++;
                for(; currXferRPtr != colorXferTuples.end(); currXferRPtr++) {
                    if (tmpBool == false && (*currXferRPtr).xferWeight == NODE_WEIGHT_MISSING) {
                        tmpBool = true;
                        saveXferRPtr = currXferRPtr;
                    }
                    if ((*currXferRPtr).xferWeight == NODE_WEIGHT_MISSING) {
#if !NDEBUG
                        assert (runWeight < NODE_WEIGHT_MISSING);
#endif
                        (*currXferRPtr).xferWeight = runWeight;
                        runWeight = 0;
                    } else {
                        if (dbgNodeWtType == 'c') {
                            runWeight++;
                        } else {
                            runWeight += (*currXferRPtr).xferWeight;
                        }
                    }
                    prevXferRPtr = currXferRPtr;
                }
                //This implementation requires that there is at least one element with missing weight per rank
                //TODO: Cast as a prefix sum problem
                if (comm.rank() !=  0) {
                    assert(tmpBool);
                }

                //Handle the first element with missing weight
                if (lastXferRecord.xferWeight == NODE_WEIGHT_MISSING) {
                    lastXferRecord.xferColor = NODE_COLOR_MISSING;
                }
                lastXferRecord.xferWeight = runWeight;
                dbgp::xferRecordType prevXferRecord = mxx::right_shift(lastXferRecord, comm);
                if (comm.rank() !=  0) {
                    if (prevXferRecord.xferColor == NODE_COLOR_MISSING) {
                        //Nothing to do
                    } else {
#if !NDEBUG
                        assert (((*saveXferRPtr).xferWeight + prevXferRecord.xferWeight) < NODE_WEIGHT_MISSING);
#endif
                        (*saveXferRPtr).xferWeight += prevXferRecord.xferWeight;
                    }
                }
*/                
                if (tmpBool == false && currXferRPtr->xferWeight == NODE_WEIGHT_MISSING) {
                    tmpBool = true;
                }
                currXferRPtr++;
                //Process first color later
                for(; currXferRPtr != colorXferTuples.end(); currXferRPtr++) {
                    if (tmpBool == false && currXferRPtr->xferWeight == NODE_WEIGHT_MISSING) {
                        tmpBool = true;
                    }
                    if (prevXferRPtr->xferColor != currXferRPtr->xferColor) {
                        break;
                    }
                    prevXferRPtr = currXferRPtr;
                }
                for(; currXferRPtr != colorXferTuples.end(); currXferRPtr++) {
                    if (tmpBool == false && currXferRPtr->xferWeight == NODE_WEIGHT_MISSING) {
                        tmpBool = true;
                    }
                    if (prevXferRPtr->xferColor != currXferRPtr->xferColor) {
                        runWeight = 0;
                    }
                    if (currXferRPtr->xferWeight == NODE_WEIGHT_MISSING) {
#if !NDEBUG
                        assert (runWeight < NODE_WEIGHT_MISSING);
#endif
                        currXferRPtr->xferWeight = runWeight;
                    } else {
                        if (dbgNodeWtType == 'c') {
                            runWeight++;
                        } else {
                            runWeight += currXferRPtr->xferWeight;
                        }
                    }
                    prevXferRPtr = currXferRPtr;
                }
                //This implementation requires that there is at least one element with missing weight per rank
                //TODO: Cast as a prefix sum problem
                if (comm.rank() !=  0) {
                    assert(tmpBool);
                }
                //Process first color
                auto xitr = colorXferTuples.back();
                dbgp::xferRecordType lastXferRecord(xitr.xferColor, xitr.xferUID, xitr.xferVID, runWeight);
                dbgp::xferRecordType prevXferRecord = mxx::right_shift(lastXferRecord, comm);
                runWeight = prevXferRecord.xferWeight;
                currXferRPtr = colorXferTuples.begin();
                prevXferRPtr = currXferRPtr;
                if (comm.rank() != 0) {
                    if (prevXferRecord.xferColor != currXferRPtr->xferColor) {
                        runWeight = 0;
                    }
                    if (currXferRPtr->xferWeight == NODE_WEIGHT_MISSING) {
#if !NDEBUG
                        assert (runWeight < NODE_WEIGHT_MISSING);
#endif
                        currXferRPtr->xferWeight = runWeight;
                    } else {
                        if (dbgNodeWtType == 'c') {
                            runWeight++;
                        } else {
                            runWeight += currXferRPtr->xferWeight;
                        }
                    }
                } else {
                    if (dbgNodeWtType == 'c') {
                        runWeight = 1;
                    } else {
                        runWeight = currXferRPtr->xferWeight;
                    }
                }
                currXferRPtr++;
                for(; currXferRPtr != colorXferTuples.end(); currXferRPtr++) {
                    if (prevXferRPtr->xferColor != currXferRPtr->xferColor) {
                        break;
                    }
                    if (currXferRPtr->xferWeight == NODE_WEIGHT_MISSING) {
#if !NDEBUG
                        assert (runWeight < NODE_WEIGHT_MISSING);
#endif
                        currXferRPtr->xferWeight = runWeight;
                    } else {
                        if (dbgNodeWtType == 'c') {
                            runWeight++;
                        } else {
                            runWeight += currXferRPtr->xferWeight;
                        }
                    }
                    prevXferRPtr = currXferRPtr;
                }
            });
#if !NDEBUG
        //xferWeight of all colorXferTuples must be valid at this point
        for(auto currXferRPtr = colorXferTuples.begin(); currXferRPtr != colorXferTuples.end(); currXferRPtr++) {
            assert ((*currXferRPtr).xferWeight != NODE_WEIGHT_MISSING);
        }
#endif
    }

    // updateWeights
    //
    // nonEdges   : (u, v) : both u and v are non-junction nodes; for all (u,v),
    //              (v,u) exist in nonEdges
    // nodeColors : colors by the CCL algorithm
    // nodeTuples : updated weights of non-junction nodes
    void updateWeights(mxx::comm& comm,
                       std::vector<dbgp::edgeRecordType>& nonEdges,
                       std::vector<dbgp::cclNodeType>& nodeColors,
                       std::vector<dbgp::vertexRecordType>& nodeTuples,
                       std::string t1FileName,
                       char dbgNodeWtType){
        dbgp::Timer timer;
        std::vector<dbgp::xferRecordType> colorXferTuples;

        // Emit color xfer tuples (tuples used to xfer the color info)
        emitXferTuples(comm, nonEdges, nodeColors, colorXferTuples);
        timer.end_section("finished emitting color transfer tuples");
        
        // write color kmer mapping
        writeColorKmerMapping(comm, nodeColors, t1FileName);
        // clear node colors and edges
        std::vector<dbgp::edgeRecordType>().swap(nonEdges);
        std::vector<dbgp::cclNodeType>().swap(nodeColors);
        mxx::distribute_inplace(colorXferTuples, comm); // distribute equally
        timer.end_section("finished writing color kmer mapping");

        // Update color values
        updateXferColor(comm, colorXferTuples);
        timer.end_section("finished updating transfer colors");

        // update Weights
        updateXferWeights(comm, colorXferTuples, dbgNodeWtType);
        timer.end_section("finished updating transfer weights");

        // filter colorXferTuples to nodeTuples
        nodeTuples.resize(colorXferTuples.size());
        auto ntr = nodeTuples.begin();
            std::uint64_t abnormalCount = 0;
        for(auto tx : colorXferTuples){
            if(tx.xferVID == NODE_ID_MISSING){
                (*ntr).set(tx.xferUID, tx.xferColor, tx.xferWeight);
                ntr++;
                    if (tx.xferWeight < 2) {
                        abnormalCount++;
                    }
            }
        }
            abnormalCount = mxx::reduce(abnormalCount, 0, std::plus<uint64_t>(), comm);
            LOG_IF(comm.rank() == 0, INFO) << "Abnormal weight count : " << abnormalCount;
        auto nsize = std::distance(nodeTuples.begin(), ntr);
        nodeTuples.resize(nsize);
        // Compact this vector
        std::vector<dbgp::vertexRecordType>(nodeTuples).swap(nodeTuples);
        mxx::distribute_inplace(nodeTuples, comm); // distribute equally
        timer.end_section("finished updating weights");
    }

    // pcEmitXferTuples
    //
    // jnnEdges   (IN) : (u, v) : u is a junction nodes; for all (u,v),
    // nonWtNodes (IN) : non-jn nodes with accumulated weights the CCL algorithm
    // xferTuple  (OUT): merging of the inputs
    void pcEmitXferTuples(mxx::comm& comm,
                          std::vector<dbgp::edgeRecordType>& jnnEdges,
                          std::vector<dbgp::vertexRecordType>& nonWtNodes,
                          std::vector<dbgp::xferRecordType>& xferTuples){
#if !NDEBUG
        comm.with_subset(nonWtNodes.size() > 0, [&](const mxx::comm& comm){
            mxx::sort(nonWtNodes.begin(), nonWtNodes.end(),
                      [&](const dbgp::vertexRecordType& x, const dbgp::vertexRecordType& y){
                      return (x.vertexID < y.vertexID);
            }, comm);
            //TODO: Not checking across process boundaries
            auto citr = nonWtNodes.begin();
            auto pitr = citr;
            citr++;
//            std::uint64_t abnormalCount = 0;
            for (; citr != nonWtNodes.end(); citr++) {
                assert (pitr->vertexID != citr->vertexID);
                pitr = citr;
/*
                if (citr->vertexWeight < 2) {
                    abnormalCount++;
                }
*/
/*
                if (citr->vertexColor == 25195748317960630) {
                    auto tdist = std::distance(nonWtNodes.begin(), citr);
                    LOG(INFO) << comm.rank() << tdist << " id: " << citr->vertexID << " c: "
                    << citr->vertexColor << " w: " << citr->vertexWeight;
                }
*/
            }
/*
            abnormalCount = mxx::reduce(abnormalCount, 0, std::plus<uint64_t>(), comm);
            LOG_IF(comm.rank() == 0, INFO) << "Abnormal weight count : " << abnormalCount;
            std::size_t totalCount = nonWtNodes.size();
            totalCount = mxx::reduce(totalCount, 0, std::plus<std::size_t>(), comm);
            LOG_IF(comm.rank() == 0, INFO) << "Total weight count    : " << totalCount;
*/
        });

        comm.with_subset(jnnEdges.size() > 0, [&](const mxx::comm& comm){
            mxx::sort(jnnEdges.begin(), jnnEdges.end(),
                      [&](const dbgp::edgeRecordType& x, const dbgp::edgeRecordType& y){
                      return (x.vertexU < y.vertexU) ||
                      ((x.vertexU == y.vertexU) && (x.vertexV < y.vertexV));
            }, comm);
            //TODO: Not checking across process boundaries
            auto citr = jnnEdges.begin();
            auto pitr = citr;
            citr++;
            for (; citr != jnnEdges.end(); citr++) {
                assert ((pitr->vertexU != citr->vertexU) ||
                        (pitr->vertexV != citr->vertexV));
                pitr = citr;
            }
        });
#endif

        comm.with_subset(jnnEdges.size() + nonWtNodes.size() > 0, [&](const mxx::comm& comm){
                xferTuples.resize(jnnEdges.size() + nonWtNodes.size());
                auto xitr = xferTuples.begin();
                for(auto jedge : jnnEdges){
#if !NDEBUG
                    assert(jedge.vertexV != NODE_ID_MISSING);
                    assert(jedge.vertexU != NODE_ID_MISSING);
#endif
                    (*xitr).set(NODE_COLOR_MISSING, jedge.vertexU,
                                jedge.vertexV, jedge.edgeWeight);
                    xitr++;
                }
                for(auto nvert : nonWtNodes){
#if !NDEBUG
                    assert(nvert.vertexID != NODE_ID_MISSING);
                    assert(nvert.vertexColor != NODE_COLOR_MISSING);
#endif
                    (*xitr).set(nvert.vertexColor, NODE_ID_MISSING,
                                nvert.vertexID, nvert.vertexWeight);
                    xitr++;
                }
                auto nsize = std::distance(xferTuples.begin(), xitr);
                xferTuples.resize(nsize);
                std::vector<dbgp::xferRecordType>(xferTuples).swap(xferTuples);
            });

    }


    void generateCompactedGraph(mxx::comm& comm,
                               std::vector<dbgp::xferRecordType>& xferTuples,
                               std::vector<dbgp::adjRecordType>& cmptEdges,
                               std::vector<dbgp::colorType>& graphColors){

         comm.with_subset(xferTuples.begin() !=  xferTuples.end(), [&](const mxx::comm& comm){
//Below implementation is due to Sriram
/*
                 // find the last tuple with same VID as the one at the back
                 auto lastColor = xferTuples.back().xferColor;
                 auto lastUpdNode = xferTuples.back().xferVID;
                 for(auto xitr = xferTuples.rbegin(); xitr != xferTuples.rend(); xitr++){
                     if((*xitr).xferVID != lastUpdNode)
                         break;
                     lastColor = xferTuples.back().xferColor;
                     lastUpdNode = xferTuples.back().xferVID;
                 }
                 auto shiftColor = mxx::right_shift(lastColor, comm); // TODO: is shift ok ?
                 auto shiftUpdNode = mxx::right_shift(lastUpdNode, comm); // TODO: is shift ok ?

                 // update the non-jn vertex to the color vert
                 auto xitr = xferTuples.begin();
                 auto prevVID = (*xitr).xferVID;
                 auto prevColor = (*xitr).xferColor;
                 if(comm.rank() > 0 && shiftUpdNode == prevVID){
                     prevVID = shiftUpdNode; prevColor = shiftColor;
                 }

                 cmptEdges.resize(2 * xferTuples.size());
                 auto exitr = cmptEdges.begin();
                 for(;xitr != xferTuples.end();xitr++){
                     if(prevColor != NODE_COLOR_MISSING &&
                        (*xitr).xferColor == NODE_COLOR_MISSING){
                         // Update the vertex to color
                         assert(prevVID == NODE_ID_MISSING);
                         (*xitr).xferVID = prevColor;
                         // edge entries;
                         (*exitr).set((*xitr).xferUID, (*xitr).xferVID, (*xitr).xferWeight);
                         exitr++;
                         (*exitr).set((*xitr).xferVID, (*xitr).xferUID, (*xitr).xferWeight);
                         exitr++;
                     } else if((*xitr).xferColor != NODE_COLOR_MISSING){
                         // (*xitr).xferUID = (*xfer)
                         assert((*xitr).xferUID == NODE_ID_MISSING);
                         prevColor = (*xitr).xferColor;
                         prevVID = (*xitr).xferVID;
                         // node entries;
                         (*exitr).set((*xitr).xferVID, NODE_ID_MISSING, (*xitr).xferWeight);
                         exitr++;
                     } else {
                         // technically, this should not happen
                         assert(true);
                     }
                 }
                 auto nsize = std::distance(cmptEdges.begin(), exitr);
                 cmptEdges.resize(nsize);
                 std::vector<dbgp::edgeRecordType>(cmptEdges).swap(cmptEdges);
*/

                 std::vector<dbgp::adjRecordType> tmpEdgeList;
                 auto bxitr = xferTuples.back();
                 dbgp::xferRecordType lastXR(bxitr.xferColor, bxitr.xferUID, bxitr.xferVID, bxitr.xferWeight);
                 if (bxitr.xferColor == NODE_COLOR_MISSING) {
                     lastXR.set(NODE_COLOR_MISSING, NODE_ID_MISSING, NODE_ID_MISSING, NODE_WEIGHT_MISSING);
                 }
                 dbgp::xferRecordType prevXR = mxx::right_shift(lastXR, comm);
                 if (comm.rank() == 0) {
                     prevXR.set(NODE_COLOR_MISSING, NODE_ID_MISSING, NODE_ID_MISSING, NODE_WEIGHT_MISSING);
                 }
                 auto prevVID = prevXR.xferVID;
                 auto prevColor = prevXR.xferColor;
                 auto prevWeight = prevXR.xferWeight;
                 
#if !NDEBUG
                 std::uint32_t valCnt = 0;
//                 std::uint64_t cntValCnt = 0;
//                 std::vector<dbgp::xferRecordType> tmpXferTuples;
                 for(auto xitr = xferTuples.begin(); xitr != xferTuples.end(); xitr++){
                    if ((*xitr).xferUID == NODE_ID_MISSING) {
/*
                        if (valCnt > 2) {
                            cntValCnt++;
                            for(auto titr = tmpXferTuples.begin(); titr != tmpXferTuples.end(); titr++){
                                LOG_IF(comm.rank() == 0, INFO) << " " << (*titr).xferColor << " " << (*titr).xferUID
                                << " " << (*titr).xferVID << " " << (*titr).xferWeight;
                            }
                            LOG_IF(comm.rank() == 0, INFO) << " ";
                        }
*/
                        assert (valCnt < 3);
                        valCnt = 1;
                        prevVID = (*xitr).xferVID;
                        prevColor = (*xitr).xferColor;
                        prevWeight = (*xitr).xferWeight;
//                        std::vector<dbgp::xferRecordType>().swap(tmpXferTuples);
//                        tmpXferTuples.emplace_back((*xitr).xferColor, (*xitr).xferUID, (*xitr).xferVID, (*xitr).xferWeight);
                    } else {
                        assert ((*xitr).xferColor == NODE_COLOR_MISSING);
                        if ((*xitr).xferVID == prevVID) {
                            valCnt++;
//                            tmpXferTuples.emplace_back((*xitr).xferColor, (*xitr).xferUID, (*xitr).xferVID, (*xitr).xferWeight);
                        } else {
/*
                            if (valCnt > 2) {
                                cntValCnt++;
                                for(auto titr = tmpXferTuples.begin(); titr != tmpXferTuples.end(); titr++){
                                    LOG_IF(comm.rank() == 0, INFO) << " " << (*titr).xferColor << " " << (*titr).xferUID
                                    << " " << (*titr).xferVID << " " << (*titr).xferWeight;
                                }
                                LOG_IF(comm.rank() == 0, INFO) << " ";
                            }
*/
                            assert (valCnt < 3);
                            valCnt = 0;
//                            std::vector<dbgp::xferRecordType>().swap(tmpXferTuples);
                        }
                    }
//                    assert (valCnt < 3);
                 }
//                 cntValCnt = mxx::reduce(cntValCnt, 0, comm);
//                 LOG_IF(comm.rank() == 0, INFO) << "cntValCnt : " << cntValCnt;
                 prevVID = prevXR.xferVID;
                 prevColor = prevXR.xferColor;
                 prevWeight = prevXR.xferWeight;
#endif
                 
                 for(auto xitr = xferTuples.begin(); xitr != xferTuples.end(); xitr++){
                    if ((*xitr).xferColor == NODE_COLOR_MISSING) {
//              if ((xitr->xferUID == 2182137596314813776) || (xitr->xferUID == 3434791461203729892)) {
//                auto tdist = std::distance(xferTuples.begin(), xitr);
//                LOG(INFO) << comm.rank() << " tT " << (tdist - 1) << " u: " << NODE_ID_MISSING << " v: " << prevVID << " c: " << prevColor << " w: " << prevWeight;
//                LOG(INFO) << comm.rank() << " tT " << tdist << " u: " << xitr->xferUID << " v: " << xitr->xferVID << " c: " << xitr->xferColor << " w: " << xitr->xferWeight;
//              }
                        if ((*xitr).xferVID == prevVID) {
/*
            if (xitr->xferUID == 1944663780267557337) {
                auto tdist = std::distance(xferTuples.begin(), xitr);
                LOG(INFO) << comm.rank() << tdist << " s: " << xitr->xferUID << " d: " << xitr->xferVID << " c: " << prevColor;
            }
*/
                            //Edge from non-junction node
                            tmpEdgeList.emplace_back(prevColor, (*xitr).xferUID, (*xitr).xferWeight, prevWeight);
                            //Edge from junction node
                            tmpEdgeList.emplace_back((*xitr).xferUID, prevColor, (*xitr).xferWeight, NODE_WEIGHT_MISSING);
#if !NDEBUG
                            graphColors.emplace_back(prevColor);
#endif
                        } else {
                            //Edge from junction node
                            tmpEdgeList.emplace_back((*xitr).xferUID, (*xitr).xferVID, (*xitr).xferWeight, NODE_WEIGHT_MISSING);
                        }
                    } else {
                        //Assert (*xitr).xferUID == NODE_ID_MISSING
                        prevVID = (*xitr).xferVID;
                        prevColor = (*xitr).xferColor;
                        prevWeight = (*xitr).xferWeight;
                    }
                 }
                 std::vector<dbgp::adjRecordType>(tmpEdgeList).swap(tmpEdgeList);
                 mxx::distribute_inplace(tmpEdgeList, comm); // distribute equally
#if !NDEBUG
                 std::vector<dbgp::colorType>(graphColors).swap(graphColors);
                 mxx::distribute_inplace(graphColors, comm); // distribute equally
#endif

/*
        for(auto eitr = tmpEdgeList.begin(); eitr != tmpEdgeList.end(); eitr++) {
              if ((eitr->vertexU == 2182137596314813776) || (eitr->vertexU == 3434791461203729892)) {
                auto tdist = std::distance(tmpEdgeList.begin(), eitr);
                LOG(INFO) << comm.rank() << " tE " << tdist << " u: " << eitr->vertexU << " v: " << eitr->vertexV << " ew: " << eitr->edgeWeight << " nw: " << eitr->vUWeight;
              }
              if ((eitr->vertexV == 2182137596314813776) || (eitr->vertexV == 3434791461203729892)) {
                auto tdist = std::distance(tmpEdgeList.begin(), eitr);
                LOG(INFO) << comm.rank() << " tE " << tdist << " u: " << eitr->vertexU << " v: " << eitr->vertexV << " ew: " << eitr->edgeWeight << " nw: " << eitr->vUWeight;
              }
        }
*/

//        comm.with_subset(
//            tmpEdgeList.begin() != tmpEdgeList.end(), [&](const mxx::comm& comm){
                // sort by UID and VID
                mxx::sort(tmpEdgeList.begin(), tmpEdgeList.end(),
                          [&](const dbgp::adjRecordType& ex,
                              const dbgp::adjRecordType& ey){
                              return (ex.vertexU < ey.vertexU) ||
                                  ((ex.vertexU == ey.vertexU) && (ex.vertexV < ey.vertexV));
                          }, comm);
//            });

                 std::uint32_t multiplicity = 1;
                 auto titr = tmpEdgeList.begin();
                 auto ptitr = titr;
                 auto stitr = titr;
                 titr++;
                 //TODO: Not validating multiplicity for boundary case
                 for (; titr != tmpEdgeList.end(); titr++) {
                    if ((ptitr->vertexU != titr->vertexU) || (ptitr->vertexV != titr->vertexV)) {
                        stitr = ptitr;
                        break;
                    } else {
                        titr->edgeWeight += ptitr->edgeWeight;
                        assert (ptitr->vUWeight == titr->vUWeight);
                    }
                    ptitr = titr;
                 }
                 ptitr++; titr++;
                 for (; titr != tmpEdgeList.end(); titr++) {
                    if ((ptitr->vertexU == titr->vertexU) && (ptitr->vertexV == titr->vertexV)) {
                        titr->edgeWeight += ptitr->edgeWeight;
                        assert (ptitr->vUWeight == titr->vUWeight);
                        multiplicity++;
                    } else {
                        assert (multiplicity < 3);
                        cmptEdges.emplace_back(ptitr->vertexU, ptitr->vertexV, ptitr->edgeWeight, ptitr->vUWeight);
                        multiplicity = 1;
                    }
                    ptitr = titr;
                 }
                 if (comm.rank() == comm.size() - 1) {
                    assert (multiplicity < 3);
                    cmptEdges.emplace_back(ptitr->vertexU, ptitr->vertexV, ptitr->edgeWeight, ptitr->vUWeight);
                 }

                 auto btitr = tmpEdgeList.back();
                 dbgp::adjRecordType lastTR(btitr.vertexU, btitr.vertexV, btitr.edgeWeight, btitr.vUWeight);
                 dbgp::adjRecordType prevTR = mxx::right_shift(lastTR, comm);
                 if (comm.rank() == 0) {
                        cmptEdges.emplace_back(stitr->vertexU, stitr->vertexV, stitr->edgeWeight, stitr->vUWeight);
                 } else {
                    if ((prevTR.vertexU == stitr->vertexU) && (prevTR.vertexV == stitr->vertexV)) {
                        stitr->edgeWeight += prevTR.edgeWeight;
                        assert (prevTR.vUWeight == stitr->vUWeight);
                        cmptEdges.emplace_back(stitr->vertexU, stitr->vertexV, stitr->edgeWeight, stitr->vUWeight);
                    } else {
                        cmptEdges.emplace_back(prevTR.vertexU, prevTR.vertexV, prevTR.edgeWeight, prevTR.vUWeight);
                        cmptEdges.emplace_back(stitr->vertexU, stitr->vertexV, stitr->edgeWeight, stitr->vUWeight);
                    }
                 }
                 std::vector<dbgp::adjRecordType>(cmptEdges).swap(cmptEdges);

            });
    }


    // propagateColor
    //
    // jnnEdges   (IN): (u, v) : u is a junction nodes; for all (u,v),
    //            (OUT) : empty array (cleared for releasing memory)
    // nonWtNodes (IN): non-jn nodes with accumulated weights the CCL algorithm
    //            (OUT) : empty array (cleared for releasing memory)
    // cmptEdges (OUT): Edges of compacted graph
    void propagateColor(mxx::comm& comm,
                        std::vector<dbgp::edgeRecordType>& jnnEdges,
                        std::vector<dbgp::vertexRecordType>& nonWtNodes,
                        std::vector<dbgp::adjRecordType>& cEdges,
                        std::vector<dbgp::colorType>& graphColors){

        dbgp::Timer timer;
        std::uint64_t jnnEdgesSize = (std::uint64_t) jnnEdges.size();
        // 1: emit xfer records
        std::vector<dbgp::xferRecordType> xferTuples;
        pcEmitXferTuples(comm, jnnEdges, nonWtNodes, xferTuples);
        // clear input after generating the xfer tuples (for memory prupose)
        std::vector<dbgp::edgeRecord>().swap(jnnEdges);
        std::vector<dbgp::vertexRecordType>().swap(nonWtNodes);
        timer.end_section("finished emitting pc transfer tuples");

/*
        for(auto eitr = xferTuples.begin(); eitr != xferTuples.end(); eitr++) {
              if ((eitr->xferUID == 2182137596314813776) || (eitr->xferUID == 3434791461203729892)) {
                auto tdist = std::distance(xferTuples.begin(), eitr);
                LOG(INFO) << comm.rank() << " xT " << tdist << " u: " << eitr->xferUID << " v: " << eitr->xferVID << " c: " << eitr->xferColor << " w: " << eitr->xferWeight;
              }
              if ((eitr->xferVID == 2182137596314813776) || (eitr->xferVID == 3434791461203729892)) {
                auto tdist = std::distance(xferTuples.begin(), eitr);
                LOG(INFO) << comm.rank() << " xT " << tdist << " u: " << eitr->xferUID << " v: " << eitr->xferVID << " c: " << eitr->xferColor << " w: " << eitr->xferWeight;
              }
        }
*/

        mxx::distribute_inplace(xferTuples, comm); // distribute equally
        // 2:sort by xferVID and then by color
        comm.with_subset(xferTuples.begin() !=  xferTuples.end(), [&](const mxx::comm& comm){
                mxx::sort(xferTuples.begin(), xferTuples.end(),
                          [&](const dbgp::xferRecordType& x,
                              const dbgp::xferRecordType& y){
                              return (x.xferVID < y.xferVID) ||
                                  ((x.xferVID == y.xferVID) &&
                                   (x.xferColor < y.xferColor));
                          },
                          comm);
            });

        // 3:update the non-junction vertex to color vertex
        generateCompactedGraph(comm, xferTuples, cEdges, graphColors);

/*
        for(auto eitr = cEdges.begin(); eitr != cEdges.end(); eitr++) {
              if ((eitr->vertexU == 2182137596314813776) || (eitr->vertexU == 3434791461203729892)) {
                auto tdist = std::distance(cEdges.begin(), eitr);
                LOG(INFO) << comm.rank() << " cE " << tdist << " u: " << eitr->vertexU << " v: " << eitr->vertexV << " ew: " << eitr->edgeWeight << " nw: " << eitr->vUWeight;
              }
              if ((eitr->vertexV == 2182137596314813776) || (eitr->vertexV == 3434791461203729892)) {
                auto tdist = std::distance(cEdges.begin(), eitr);
                LOG(INFO) << comm.rank() << " cE " << tdist << " u: " << eitr->vertexU << " v: " << eitr->vertexV << " ew: " << eitr->edgeWeight << " nw: " << eitr->vUWeight;
              }
        }
*/

        timer.end_section("finished generating compacted graph");
    }


      void checkEdgeList(mxx::comm& comm, std::vector<dbgp::adjRecordType>& cEdges,
                         char dbgNodeWtType) {
        dbgp::Timer timer;
        std::vector<dbgp::adjRecordType> tmpEdgeList;
        
        for(auto editr = cEdges.begin(); editr != cEdges.end(); editr++){
/*
            if (editr->vertexU == 25195748317960630) {
                auto tdist = std::distance(cEdges.begin(), editr);
                LOG(INFO) << comm.rank() << tdist << " es: " << editr->vertexU << " ed: " << editr->vertexV << " w: " << editr->edgeWeight;
            }
            if (editr->vertexV == 25195748317960630) {
                auto tdist = std::distance(cEdges.begin(), editr);
                LOG(INFO) << comm.rank() << tdist << " ed: " << editr->vertexV << " es: " << editr->vertexU << " w: " << editr->edgeWeight;
            }
*/
              //Add real edge to tmpEdgeList
              tmpEdgeList.emplace_back(editr->vertexU, editr->vertexV, (uint32_t) 1, editr->vUWeight);
              //Add test edge to tmpEdgeList
              tmpEdgeList.emplace_back(editr->vertexV, editr->vertexU, (uint32_t) 2, editr->vUWeight);
        }
        std::vector<dbgp::adjRecordType>(tmpEdgeList).swap(tmpEdgeList);
        
        comm.with_subset(
            tmpEdgeList.begin() != tmpEdgeList.end(), [&](const mxx::comm& comm){
                // sort by UID, VID, and CWT
                mxx::sort(tmpEdgeList.begin(), tmpEdgeList.end(),
                          [&](const dbgp::adjRecordType& ex,
                              const dbgp::adjRecordType& ey){
                              return (ex.vertexU < ey.vertexU) ||
                                  ((ex.vertexU == ey.vertexU) && (ex.vertexV < ey.vertexV)) ||
                                  ((ex.vertexU == ey.vertexU) && (ex.vertexV == ey.vertexV) && (ex.edgeWeight < ey.edgeWeight));
                          }, comm);
            });
        timer.end_section("finished sorting edges in checkEdgeList");

/*
        for (auto titr = tmpEdgeList.begin(); titr != tmpEdgeList.end(); titr++) {
            if (titr->vertexU == 25195748317960630) {
                auto tdist = std::distance(tmpEdgeList.begin(), titr);
                LOG(INFO) << comm.rank() << tdist << " ts: " << titr->vertexU << " td: " << titr->vertexV << " w: " << titr->edgeWeight;
            }
            if (titr->vertexV == 25195748317960630) {
                auto tdist = std::distance(tmpEdgeList.begin(), titr);
                LOG(INFO) << comm.rank() << tdist << " td: " << titr->vertexV << " ts: " << titr->vertexU << " w: " << titr->edgeWeight;
            }
        }
*/

        uint64_t totalCount = 0, pairsCount = 0, singlesCount = 0, unknownCount=0;
        uint64_t jnnSinglesCount = 0, nonSinglesCount = 0;
        comm.with_subset(
            tmpEdgeList.begin() != tmpEdgeList.end(), [&](const mxx::comm& comm){
                dbgp::adjRecordType lastRecord(tmpEdgeList.back().vertexU, tmpEdgeList.back().vertexV,
                                               tmpEdgeList.back().edgeWeight, tmpEdgeList.back().vUWeight);
                dbgp::adjRecordType prevRecord = mxx::right_shift(lastRecord, comm);
                auto editr = tmpEdgeList.begin();
                if (comm.rank() == 0) {
                    if (editr->edgeWeight == 1) {
                        totalCount++;
                    }
                    prevRecord.set(editr->vertexU, editr->vertexV, editr->edgeWeight, editr->vUWeight);
                    editr++;
                }
                
                for(; editr != tmpEdgeList.end(); editr++){
                    if (editr->edgeWeight == 1) {
                        totalCount++;
                    }
                    
                    if (prevRecord.edgeWeight == 1) {
                        if (editr->edgeWeight == 1) {
                            singlesCount++;
                            if (prevRecord.vUWeight == NODE_WEIGHT_MISSING) {
//                                LOG(INFO) << comm.rank() << " single jnnEdge";
                                jnnSinglesCount++;
                            } else {
//                                LOG(INFO) << comm.rank() << " single nonEdge";
                                nonSinglesCount++;
                            }
                            auto tdist = std::distance(tmpEdgeList.begin(), editr);
                            LOG(INFO) << comm.rank() << " " << (tdist - 1) << " u: " << prevRecord.vertexU
                            << " v: " << prevRecord.vertexV << " ew: " << prevRecord.edgeWeight << " nw: " << prevRecord.vUWeight;
                            LOG(INFO) << comm.rank() << " " << tdist << " u: " << editr->vertexU
                            << " v: " << editr->vertexV << " ew: " << editr->edgeWeight << " nw: " << editr->vUWeight;
                        } else {
                            if (prevRecord.vertexU == editr->vertexU &&
                                prevRecord.vertexV == editr->vertexV){
                                pairsCount++;
                            } else {
                                unknownCount++;
                            }
                        }
                    }
                    if (prevRecord.edgeWeight == 2) {
                        if (editr->edgeWeight == 2) {
                            auto tdist = std::distance(tmpEdgeList.begin(), editr);
                            LOG(INFO) << comm.rank() << " " << (tdist - 1) << " u: " << prevRecord.vertexU
                            << " v: " << prevRecord.vertexV << " ew: " << prevRecord.edgeWeight << " nw: " << prevRecord.vUWeight;
                            LOG(INFO) << comm.rank() << " " << tdist << " u: " << editr->vertexU
                            << " v: " << editr->vertexV << " ew: " << editr->edgeWeight << " nw: " << editr->vUWeight;
                        }
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
        jnnSinglesCount = mxx::reduce(jnnSinglesCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "jnnSinglesCount : " << jnnSinglesCount;
        nonSinglesCount = mxx::reduce(nonSinglesCount, 0, std::plus<uint64_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "nonSinglesCount : " << nonSinglesCount;
        timer.end_section("finished analyzing for single edges");
      }


    void checkGraphCycles(mxx::comm& comm, std::vector<dbgp::colorType>& graphColors){
        comm.with_subset(
            graphColors.begin() != graphColors.end(), [&](const mxx::comm& comm){
                mxx::sort(graphColors.begin(), graphColors.end(),
                          [&](const dbgp::colorType& cx,
                              const dbgp::colorType& cy){
                              return (cx < cy);}, comm);
                std::uint32_t multiplicity = 1;
                std::uint64_t cycleCount = 0;
                dbgp::colorType prevColor, currColor, nextColor;
                prevColor = mxx::right_shift(graphColors.back(), comm);
                if (comm.rank() == 0) {
                    prevColor = NODE_COLOR_MISSING;
                }
                auto citr = graphColors.begin();
                currColor = *citr;
                if (prevColor != currColor) {
                    prevColor = currColor;
                    citr++;
                }
                for (; citr != graphColors.end(); citr++) {
                    currColor = *citr;
                    if (prevColor == currColor) {
                        multiplicity++;
                    } else {
                        if (multiplicity == 1) {
                            cycleCount++;
//                            auto cdist = std::distance(graphColors.begin(), citr);
//                            LOG(INFO) << comm.rank() << " " << (cdist - 1) << " prevColor: " << prevColor << " currColor: " << currColor;
                        }
                        assert (multiplicity < 4);
                        multiplicity = 1;
                    }
                    prevColor = currColor;
                }
                nextColor = mxx::left_shift(graphColors.front(), comm);
                if (comm.rank() == (comm.size()-1)) {
                    nextColor = NODE_COLOR_MISSING;
                }
                assert(currColor == graphColors.back());
                if (currColor != nextColor) {
                    if (multiplicity == 1) {
                        cycleCount++;
//                        auto cdist = std::distance(graphColors.begin(), citr);
//                        LOG(INFO) << comm.rank() << " " << (cdist - 1) << " currColor: " << currColor << " nextColor: " << nextColor;
                    }
                    assert (multiplicity < 4);
                }
                cycleCount = mxx::reduce(cycleCount, 0, std::plus<uint64_t>(), comm);
                LOG_IF(comm.rank() == 0, INFO) << "cycleCount   : " << cycleCount;
        });
    }


/*
    void sortEdges(mxx::comm& comm,
                   std::vector<dbgp::adjRecordType>& cEdges){
        // redistribute equally
        mxx::distribute_inplace(cEdges, comm);
        
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
    }

    void constructVertexList(mxx::comm& comm,
                             std::vector<dbgp::adjRecordType>& cEdges,
                             std::vector<dbgp::vertexIdType>& vertList){
        comm.with_subset(
            cEdges.begin() != cEdges.end(), [&](const mxx::comm& comm){
                vertList.resize(cEdges.size());
                // find vertex ids
                std::size_t vertIdx = 0;
                auto lastVertex = cEdges.back().vertexU;
                auto prevVertex = mxx::right_shift(lastVertex, comm);
                if(comm.rank() == 0) prevVertex = NODE_ID_MISSING;
                for(auto ed: cEdges){
                    if(ed.vertexU != prevVertex){
                        vertList[vertIdx] = ed.vertexU;
                        vertIdx++;
                        prevVertex = ed.vertexU;
                    }
                }
                vertList.resize(vertIdx);
                std::vector<dbgp::vertexIdType>(vertList).swap(vertList);
            });
    }

    void writeVertexList(mxx::comm& comm,
                         std::vector<dbgp::vertexIdType>& vertList,
                         std::string t2FileName){
        //Write out graph nodes in color space (vertices in vertList)
        comm.with_subset(
            vertList.begin() != vertList.end(), [&](const mxx::comm& comm){
                std::ofstream outFilePtr(t2FileName);
                for(auto vitr = vertList.begin(); vitr != vertList.end(); vitr++){
                    outFilePtr << *vitr << std::endl;
                }
                outFilePtr.close();
            });
    }

    void buildLocalVertexMap(const mxx::comm& comm,
                             std::vector<dbgp::adjRecordType>& cEdges,
                             std::size_t startVertId,
                             std::vector<dbgp::vertexIdType>& vertList,
                             std::map<dbgp::vertexIdType, uint64_t>& localVerts) {
        std::vector<dbgp::vertexIdType> kmerVertexes;
        kmerVertexes.resize(cEdges.size());
        auto kvitr = kmerVertexes.begin();
        //TODO: There may be multiple copies of a vertexV
        for(auto eitr = cEdges.begin();eitr != cEdges.end();
            eitr++, kvitr++){
            *kvitr = (*eitr).vertexV;
        }
        auto nsize = std::distance(kmerVertexes.begin(), kvitr);
        kmerVertexes.resize(nsize);
        std::vector<dbgp::vertexIdType>(kmerVertexes).swap(kmerVertexes);
        KmerVertIDMapType kvmap(comm);
        std::vector<std::pair<dbgp::vertexIdType, uint64_t>> keyPairs(vertList.size());
        auto itrx = keyPairs.begin();
        uint64_t idx = 0;
        for(auto vx : vertList){
            *itrx = std::make_pair(vx, idx + startVertId); idx++;
        }
        kvmap.insert(keyPairs);
        auto translatedVertexes = kvmap.find(kmerVertexes);
        localVerts.insert(translatedVertexes.begin(), translatedVertexes.end());
    }
*/

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

/*
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
                } else {
                    //non-boundary nodes in colors are captured twice
                    assert (xitr->vUWeight % 2 == 0);
                    xitr->vUWeight = (xitr->vUWeight / 2) + 1;
                    //unsimplified : ((xitr->vUWeight - 2) / 2) + 2
                }
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
//                //All ranks will output metadata
//                dbgp::vertexIdType toVertexID;
//                if (comm.rank() != (comm.size() - 1)) {
//                    toVertexID = straddleRegion.back().vertexU;
//                } else {
//                    toVertexID = cEdges.back().vertexU;
//                }
                    outFilePtr << totalVertexCount;
                    outFilePtr << " " << nEdges;
                    outFilePtr << " " << "11";
//                    outFilePtr << " " << (edgItr->vertexU - 1); //fromVertexID
//                    outFilePtr << " " << (toVertexID - 1); //toVertexID
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
*/

    void emitEdgeList(dbgp::prtRecordType& nodeToWrite, std::ofstream& outFilePtr,
                      std::uint32_t minNodeWeight, std::uint32_t divNodeWeight, char dbgNodeWtType){
        dbgp::weightType totalVertexWt = 0;
        
        // skip dummy values
        if (nodeToWrite.vertexU == NODE_ID_MISSING) return;
        
        if (dbgNodeWtType == 'c') {
            //junction node
            if (nodeToWrite.vUWeight == NODE_WEIGHT_MISSING) {
                nodeToWrite.vUWeight = 1;
            } else {
                //non-boundary nodes in colors are captured twice
                assert (nodeToWrite.vUWeight % 2 == 0);
                nodeToWrite.vUWeight = (nodeToWrite.vUWeight / 2) + 1;
                //unsimplified : ((nodeToWrite.vUWeight - 2) / 2) + 2
            }
        } else {
            //dbgNodeWtType is 's'
            for(auto xitr = nodeToWrite.edgeWeight.begin(); xitr != nodeToWrite.edgeWeight.end(); xitr++){
                if (*xitr != NODE_WEIGHT_MISSING) {
                    totalVertexWt += *xitr;
                }
            }
            //junction node
            if (nodeToWrite.vUWeight == NODE_WEIGHT_MISSING) {
                nodeToWrite.vUWeight = totalVertexWt;
            } else {
                //weights of edges in colors are captured twice
                assert (nodeToWrite.vUWeight % 2 == 0);
                nodeToWrite.vUWeight = (nodeToWrite.vUWeight / 2) + totalVertexWt;
            }
        }

        assert (nodeToWrite.vUWeight != NODE_WEIGHT_MISSING);
#if !NDEBUG
        if (nodeToWrite.vUWeight < divNodeWeight) {
            nodeToWrite.vUWeight = (dbgp::weightType) (nodeToWrite.vUWeight/minNodeWeight);
        } else {
            nodeToWrite.vUWeight = (dbgp::weightType) (nodeToWrite.vUWeight/divNodeWeight);
        }
        assert (nodeToWrite.vUWeight);
#endif
        
        outFilePtr << nodeToWrite.vUWeight;
        auto xitr = nodeToWrite.edgeWeight.begin();
        for(auto yitr = nodeToWrite.vertexV.begin(); yitr != nodeToWrite.vertexV.end(); yitr++, xitr++){
            if (*yitr != NODE_ID_MISSING) {
                outFilePtr << " " << *yitr << " " << *xitr;
            }
        }
        outFilePtr << std::endl;
    }

    void computeWeightMetrics(const mxx::comm& comm, std::vector<dbgp::prtRecordType>& listToWrite,
//                              std::size_t totalNodes, std::size_t totalEdges, char dbgNodeWtType){
                              std::uint32_t& minNodeWeight, std::size_t& sumNodeWeight, char dbgNodeWtType){
        std::size_t sumNodeWt = 0, sumEdgeWt = 0;
        dbgp::weightType maxNodeWt = 0, maxEdgeWt = 0;
        dbgp::weightType minNodeWt, minEdgeWt;
        minNodeWt = std::numeric_limits<dbgp::weightType>::max();
        minEdgeWt = std::numeric_limits<dbgp::weightType>::max();
        
        for(const auto &nodeToWrite : listToWrite) {
            dbgp::weightType totalVertexWt = 0;
        
            // skip dummy values
            if (nodeToWrite.vertexU == NODE_ID_MISSING) continue;
        
            if (dbgNodeWtType == 'c') {
                //junction node
                if (nodeToWrite.vUWeight == NODE_WEIGHT_MISSING) {
                    totalVertexWt = 1;
                } else {
                    //non-boundary nodes in colors are captured twice
                    assert (nodeToWrite.vUWeight % 2 == 0);
                    totalVertexWt = (nodeToWrite.vUWeight / 2) + 1;
                    //unsimplified : ((nodeToWrite.vUWeight - 2) / 2) + 2
                }
            } else {
                //dbgNodeWtType is 's'
                for(auto xitr = nodeToWrite.edgeWeight.begin(); xitr != nodeToWrite.edgeWeight.end(); xitr++){
                    if (*xitr != NODE_WEIGHT_MISSING) {
                        totalVertexWt += *xitr;
                    }
                }
                //junction node
                if (nodeToWrite.vUWeight == NODE_WEIGHT_MISSING) {
                    //nothing to do
                } else {
                    //weights of edges in colors are captured twice
                    assert (nodeToWrite.vUWeight % 2 == 0);
                    totalVertexWt = (nodeToWrite.vUWeight / 2) + totalVertexWt;
                }
            }

            assert (totalVertexWt != NODE_WEIGHT_MISSING);
            sumNodeWt += (std::size_t) totalVertexWt;
            if (totalVertexWt < minNodeWt) {
                minNodeWt = totalVertexWt;
            }
            if (totalVertexWt > maxNodeWt) {
                maxNodeWt = totalVertexWt;
            }
            
            for(auto xitr = nodeToWrite.edgeWeight.begin(); xitr != nodeToWrite.edgeWeight.end(); xitr++){
                if (*xitr != NODE_WEIGHT_MISSING) {
                    sumEdgeWt += (std::size_t) *xitr;
                    if (*xitr < minEdgeWt) {
                        minEdgeWt = *xitr;
                    }
                    if (*xitr > maxEdgeWt) {
                        maxEdgeWt = *xitr;
                    }
                }
            }
        }
            
//            LOG_IF(comm.rank() == 0, INFO) << "Total Nodes : " << totalNodes;
//            LOG_IF(comm.rank() == 0, INFO) << "Total Edges : " << totalEdges;
            minNodeWt = mxx::allreduce(minNodeWt, mxx::min<dbgp::weightType>(), comm);
            LOG_IF(comm.rank() == 0, INFO) << "MIN Node Wt : " << minNodeWt;
            maxNodeWt = mxx::allreduce(maxNodeWt, mxx::max<dbgp::weightType>(), comm);
            LOG_IF(comm.rank() == 0, INFO) << "MAX Node Wt : " << maxNodeWt;
            sumNodeWt = mxx::allreduce(sumNodeWt, std::plus<std::size_t>(), comm);
            LOG_IF(comm.rank() == 0, INFO) << "SUM Node Wt : " << sumNodeWt;
            minEdgeWt = mxx::reduce(minEdgeWt, 0, mxx::min<dbgp::weightType>(), comm);
            LOG_IF(comm.rank() == 0, INFO) << "MIN Edge Wt : " << minEdgeWt;
            maxEdgeWt = mxx::reduce(maxEdgeWt, 0, mxx::max<dbgp::weightType>(), comm);
            LOG_IF(comm.rank() == 0, INFO) << "MAX Edge Wt : " << maxEdgeWt;
            sumEdgeWt = mxx::reduce(sumEdgeWt, 0, std::plus<std::size_t>(), comm);
            LOG_IF(comm.rank() == 0, INFO) << "SUM Edge Wt : " << sumEdgeWt;
            minNodeWeight = (std::uint32_t) minNodeWt;
            sumNodeWeight = sumNodeWt;
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

                std::vector<dbgp::prtRecordType> listToWrite;
                std::vector<dbgp::vertexIdType> tmpVxList;
                std::vector<dbgp::weightType> tmpWtList;
                
                auto edgItr = cEdges.begin() + startOffset;
                auto prevItr = edgItr;
                for(; edgItr != cEdges.begin() + endOffset; edgItr++){
                    if(edgItr->vertexU == prevItr->vertexU)
                        continue;
                    //emitEdgeList(prevItr, edgItr, outFilePtr, dbgNodeWtType);
                    for (auto citr = prevItr; citr != edgItr; citr++) {
                        tmpVxList.emplace_back(citr->vertexV);
                        tmpWtList.emplace_back(citr->edgeWeight);
                    }
                    listToWrite.emplace_back(prevItr->vertexU,tmpVxList,tmpWtList,prevItr->vUWeight);
                    std::vector<dbgp::vertexIdType>().swap(tmpVxList);
                    std::vector<dbgp::weightType>().swap(tmpWtList);
                    prevItr = edgItr;
                }
                assert (edgItr == cEdges.begin() + endOffset);
                assert (prevItr != edgItr);
                    //emitEdgeList(prevItr, edgItr, outFilePtr, dbgNodeWtType);
                    for (auto citr = prevItr; citr != edgItr; citr++) {
                        tmpVxList.emplace_back(citr->vertexV);
                        tmpWtList.emplace_back(citr->edgeWeight);
                    }
                    listToWrite.emplace_back(prevItr->vertexU,tmpVxList,tmpWtList,prevItr->vUWeight);
                    std::vector<dbgp::vertexIdType>().swap(tmpVxList);
                    std::vector<dbgp::weightType>().swap(tmpWtList);
                if (comm.rank() != comm.size() - 1) {
                    // write out straddling region
                    //emitEdgeList(straddleRegion.begin(), straddleRegion.end(), outFilePtr, dbgNodeWtType);
                    for (auto citr = straddleRegion.begin(); citr != straddleRegion.end(); citr++) {
                        tmpVxList.emplace_back(citr->vertexV);
                        tmpWtList.emplace_back(citr->edgeWeight);
                    }
                    listToWrite.emplace_back(straddleRegion.front().vertexU,tmpVxList,tmpWtList,straddleRegion.front().vUWeight);
                    std::vector<dbgp::vertexIdType>().swap(tmpVxList);
                    std::vector<dbgp::weightType>().swap(tmpWtList);
                } else {
                    if (totalVertexCount % comm.size()) {
                        // dummy values
                        for (auto i = (totalVertexCount % comm.size()); i < comm.size(); i++) {
                            listToWrite.emplace_back(NODE_ID_MISSING,tmpVxList,tmpWtList,NODE_WEIGHT_MISSING);
                        }
                    }
                }
                
                std::vector<dbgp::adjRecordType>().swap(cEdges); //empty list
                std::vector<dbgp::prtRecordType>(listToWrite).swap(listToWrite);
                mxx::stable_distribute_inplace(listToWrite, comm); // distribute equally

                std::ofstream outFilePtr(outFileName);
                //All ranks will output metadata
                dbgp::vertexIdType toVertexID;
                    outFilePtr << totalVertexCount;
                    outFilePtr << " " << nEdges;
                    outFilePtr << " " << "11";
                    outFilePtr << " " << (listToWrite.front().vertexU - 1); //fromVertexID
                if (comm.rank() != comm.size() - 1) {
                    outFilePtr << " " << (listToWrite.back().vertexU - 1); //toVertexID
                } else {
                    outFilePtr << " " << (totalVertexCount - 1);
                }
                    outFilePtr << std::endl;
                std::uint32_t minNodeWeight = 0, divNodeWeight = 0;
                std::size_t sumNodeWeight = 0;

#if !NDEBUG
//                computeWeightMetrics(comm, listToWrite, totalVertexCount, nEdges, dbgNodeWtType);
                computeWeightMetrics(comm, listToWrite, minNodeWeight, sumNodeWeight, dbgNodeWtType);
                LOG_IF(comm.rank() == 0, INFO) << "minNodeWeight : " << minNodeWeight;
                LOG_IF(comm.rank() == 0, INFO) << "sumNodeWeight : " << sumNodeWeight;
                divNodeWeight = (std::uint32_t) (sumNodeWeight/2147483648); //2^31
                divNodeWeight += 1;
                LOG_IF(comm.rank() == 0, INFO) << "divNodeWeight : " << divNodeWeight;
#endif

                for(auto &zx : listToWrite) {
                    emitEdgeList(zx, outFilePtr, minNodeWeight, divNodeWeight, dbgNodeWtType);
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
        writeVertexList(comm, cEdges, t2FileName);
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
        // write Adjacency List
        writeAdjacencyList(comm, cEdges, outFileName, totalVertexCount, dbgNodeWtType);
        timer.end_section("finished writing adjacency list");
    }


    int compactDBG(mxx::comm& comm, std::string ipFileName, uint64_t minKmerFreq, char dbgNodeWtType,
                   std::string outFileName, std::string filteredKmersFile,
                   std::string colorKmerMapFile, std::string colorVertexListFile){
        // Declare edgeList vectors to save edges
        // Conventions followed:
        // nonEdges : (u, v) s.t. both u and v are non-junction nodes; for all (u,v),
        //            (v,u) exist in nonEdges
        // ifcEdges : (u, v) s.t. u is a non-junction node and v is a junction node
        // jnnEdges : (u, v) s.t. u is a junction node
        //             v could be either junction or non-junction node;
        std::vector< dbgp::edgeRecordType >
            jnnEdges, //Edges w.r.t. junction nodes including interface edges
            nonEdges, //Edges w.r.t. non-junction nodes excluding interface edges
            ifcEdges; //Interface edges

        //Construct graph
        LOG_IF(!comm.rank(), INFO) << "Begin populating de Bruijn graph edges";
        //Object of the graph generator class
        dbgp::graphGen::deBruijnGraph g;

        // Populate the edges from the DBG
        g.populateEdgeList(jnnEdges, nonEdges, ifcEdges, minKmerFreq, ipFileName, filteredKmersFile, comm);
        LOG_IF(!comm.rank(), INFO) << "End populating de Bruijn graph edges";

        // Compute Connectivity
        LOG_IF(!comm.rank(), INFO) << "Begin coloring chains";
        std::vector<dbgp::cclNodeType> nonNodeColors;
        std::vector<dbgp::colorType> graphColors;
        dbgp::runCCL(comm, nonEdges, nonNodeColors, graphColors);
        LOG_IF(!comm.rank(), INFO) << "End coloring chains";


        // update weights
        LOG_IF(!comm.rank(), INFO) << "Begin updating weights";
        std::vector<dbgp::vertexRecordType> nonNodeTuples;
        dbgp::updateWeights(comm, nonEdges, nonNodeColors, nonNodeTuples, colorKmerMapFile, dbgNodeWtType);
        LOG_IF(!comm.rank(), INFO) << "End updating weights";
        // clear the node colors and edges
        std::vector<dbgp::edgeRecordType>().swap(ifcEdges);

/*
        for(auto eitr = nonNodeTuples.begin(); eitr != nonNodeTuples.end(); eitr++) {
              if ((eitr->vertexID == 2182137596314813776) || (eitr->vertexID == 3434791461203729892)) {
                auto tdist = std::distance(nonNodeTuples.begin(), eitr);
                LOG(INFO) << comm.rank() << " nNT " << tdist << " u: " << eitr->vertexID << " c: " << eitr->vertexColor << " w: " << eitr->vertexWeight;
              }
        }
*/
        
        LOG_IF(!comm.rank(), INFO) << "Begin propagating colors";
        std::vector<dbgp::adjRecordType> cmptEdges;
        dbgp::propagateColor(comm, jnnEdges, nonNodeTuples, cmptEdges, graphColors);
        LOG_IF(!comm.rank(), INFO) << "End propagating colors";

#if !NDEBUG
        LOG_IF(!comm.rank(), INFO) << "Begin checking compacted graph";
        dbgp::checkEdgeList(comm, cmptEdges, dbgNodeWtType);
        dbgp::checkGraphCycles(comm, graphColors);
        std::vector<dbgp::colorType>().swap(graphColors);
        LOG_IF(!comm.rank(), INFO) << "End checking compacted graph";
#endif

        LOG_IF(!comm.rank(), INFO) << "Begin translating vertex IDs";
        // change from kmer based node ids to 1...|V|, and write out adjacency list
        dbgp::translateVertexIds(comm, cmptEdges, outFileName, colorVertexListFile, dbgNodeWtType);
        LOG_IF(!comm.rank(), INFO) << "End translating vertex IDs";
        return 0;
    }

}
