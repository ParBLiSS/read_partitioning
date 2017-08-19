
#include <vector>
#include <string>
#include <numeric>

#include "mxx/comm.hpp"
#include "utils/mpi_utils.hpp"
#include "utils/io_utils.hpp"
#include "dbgp/recordTypes.hpp"
#include "dbgp/timer.hpp"

#define  BATCH_SIZE 50000000

namespace dbgp{


    static dbgp::PartitionValueType LAST_PART_ID;

    
    void buildPositionIndex(mxx::comm& comm, std::vector<PosIdxTupleType>& kmr_idx,
                            std::size_t& start_offset, std::string inputFile){
        dbgp::Timer timer;
        uint64_t localWorkLoad, minLoad, maxLoad, avgLoad;
        ::bliss::io::KmerFileHelper::read_file_posix<typename dbgp::PosIndexType::KmerParserType, dbgp::SeqParser,
                                                     bliss::io::SequencesIterator>(inputFile, kmr_idx, comm);
        mxx::stable_distribute_inplace(kmr_idx, comm); // distribute equally

#if !NDEBUG
        localWorkLoad = kmr_idx.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of kmr_idx_befr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif        

        std::size_t end_offset;
        std::vector<PosIdxTupleType> straddle_region;
        auto kmerIdxCompare = [&](const PosIdxTupleType& x,
                                  const PosIdxTupleType& y){
            return (x.second.get_id() == y.second.get_id());
        };
        shiftStraddlingRegion(comm, kmr_idx, start_offset, end_offset, straddle_region, kmerIdxCompare);
        kmr_idx.resize(end_offset + straddle_region.size());
        std::copy(straddle_region.begin(), straddle_region.end(), (kmr_idx.begin() + end_offset));

#if !NDEBUG
        localWorkLoad = kmr_idx.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of kmr_idx_aftr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif        

// not required
/*
        // update read ids from 0 to |R|-1
        std::size_t localReadCount = 0;
        auto prevReadID = std::numeric_limits<std::size_t>::max();
        for(auto itx = (kmr_idx.begin() + start_offset); itx != kmr_idx.end(); itx++){
            if(itx->second.get_id() != prevReadID) {
                localReadCount += 1;
                prevReadID = itx->second.get_id();
            }
        }
        
        std::size_t pxSumReadCount = mxx::exscan(localReadCount, comm);
#if !NDEBUG
        if (comm.rank() == 0) {assert (pxSumReadCount == 0);}
        LOG_IF(comm.rank() == comm.size() - 1, INFO) << "totalReadCount : " << (pxSumReadCount + localReadCount);
#endif        

        prevReadID = std::numeric_limits<std::size_t>::max();
        for(auto itx = (kmr_idx.begin() + start_offset); itx != kmr_idx.end(); itx++){
            if(itx->second.get_id() != prevReadID) {
                pxSumReadCount++;
                prevReadID = itx->second.get_id();
            }
            //To ensure that count starts from 0
            itx->second.id = (pxSumReadCount - 1);
        }
*/

        // we are no longer building the position index (as such)
        timer.end_section("finished building position index using BLISS");
    }


    void loadVertexPartitions(mxx::comm& comm,
                              std::string vertexListFile,
                              std::string partitionFile,
                              std::vector<dbgp::colorMapRecordType>& vertexPartitions){

        // load partition list file
        dbgp::PartitionValueType maxPartId = 0;
        std::vector<dbgp::PartitionValueType> partList;
        std::ifstream inStream(partitionFile);
        while(inStream.good()){
            dbgp::PartitionValueType pvx;
            inStream >> pvx;
            partList.emplace_back(pvx);
            if (pvx > maxPartId) {
                maxPartId = pvx;
            }
        }
        inStream.close();
        partList.pop_back(); // remove copy of last element
        std::vector<dbgp::PartitionValueType>(partList).swap(partList);
//            LOG_IF(!comm.rank(), INFO) << "partition on rank 0 before : " << partList[441];

/*
        std::ifstream inStream;
        std::vector<std::string> fileDataStore;
        // load partition list file
        dbgp::PartitionValueType maxPartId = 0;
        read_block(comm, partitionFile, fileDataStore);
//        LOG(INFO) << "fileDataStore size :  " << fileDataStore.size();
        std::vector<dbgp::PartitionValueType> partList(fileDataStore.size());
        auto pitr = partList.begin();
        for(auto px : fileDataStore) {
            dbgp::PartitionValueType pvx;
            std::stringstream outStream(px);
            outStream >> pvx;
            *pitr = pvx; pitr++;
            if (pvx > maxPartId) {
                maxPartId = pvx;
            }
        }
        std::vector<std::string>().swap(fileDataStore); // clear memory
//        LOG(INFO) << "partList front : " << partList.front();
//        LOG(INFO) << "partList back :  " << partList.back();
*/

#if !NDEBUG
        LAST_PART_ID = mxx::allreduce(maxPartId, mxx::max<dbgp::PartitionValueType>(), comm);
        LOG_IF(!comm.rank(), INFO) << "MAX DBG Partition ID : " << LAST_PART_ID;
#endif

//        LOG(INFO) << "partList size before : " << partList.size();
        // distribute equally
        mxx::stable_distribute_inplace(partList, comm);
//        LOG(INFO) << "partList size after :  " << partList.size();
//            LOG_IF(!comm.rank(), INFO) << "partition on rank 0 after : " << partList[441];

        // load vertex list file
        inStream.open(vertexListFile);
        std::vector<dbgp::vertexIdType> vertList;
        while(inStream.good()){
            dbgp::vertexIdType vx;
            inStream >> vx;
            vertList.emplace_back(vx);
        }
        inStream.close();
        vertList.pop_back(); // remove copy of last element
        std::vector<dbgp::vertexIdType>(vertList).swap(vertList);
//            LOG_IF(!comm.rank(), INFO) << "vertex on rank 0 before : " << vertList[441];
        
//        for (auto citrx = vertList.begin(); citrx != vertList.end(); citrx++) {
//            if (*citrx == 44583405185787572) {
//                auto tdist = std::distance(vertList.begin(), citrx);
//                LOG(INFO) << comm.rank() << " " << tdist << " v: " << *citrx;
//            }
//        }
        // distribute equally
//        LOG(INFO) << "vertList size before : " << vertList.size();
        mxx::stable_distribute_inplace(vertList, comm);
//        LOG(INFO) << "vertList size after :  " << vertList.size();
//            LOG_IF(!comm.rank(), INFO) << "vertex on rank 0 after : " << vertList[441];
//        for (auto citrx = vertList.begin(); citrx != vertList.end(); citrx++) {
//            if (*citrx == 44583405185787572) {
//                auto tdist = std::distance(vertList.begin(), citrx);
//                LOG(INFO) << comm.rank() << " " << tdist << " v: " << *citrx;
//            }
//        }
        
#if !NDEBUG
        // verify that sizes match
        int sizeCheck = (vertList.size() == partList.size());
        auto totalCheck = mxx::allreduce(sizeCheck, comm);
        if(totalCheck != comm.size()) {
          LOG_IF(!comm.rank(), INFO) << "PARTITION ID and VERTEX LIST SIZE NOT EQUAL";
          assert (0);
        }
#endif

        vertexPartitions.resize(partList.size());
        auto pitr = partList.begin();
        auto vitr = vertList.begin();
        for(auto kitr = vertexPartitions.begin(); kitr != vertexPartitions.end(); kitr++){
            kitr->set(*vitr, *vitr, *pitr);
            pitr++; vitr++;
        }
//            LOG_IF(comm.rank() == 0, INFO) << "vertexPartitions on rank 0 : " << vertexPartitions[441].vertexSrc << 
//            "," << vertexPartitions[441].vertexDest << "," << vertexPartitions[441].partitionId;
//        for (auto citrx = vertexPartitions.begin(); citrx != vertexPartitions.end(); citrx++) {
//            if (citrx->vertexSrc == 44583405185787572) {
//                auto tdist = std::distance(vertexPartitions.begin(), citrx);
//                LOG(INFO) << comm.rank() << " " << tdist << " v: " << citrx->vertexSrc << " c: " << citrx->vertexDest << " p: " << citrx->partitionId;
//            }
//        }
    }


    void loadColorMapping(mxx::comm& comm,
                          std::string colorMappingFile,
                          std::vector<dbgp::colorMapRecordType>& vertexPartitions,
                          std::vector<std::pair<dbgp::vertexIdType, dbgp::PartitionValueType>>& kmer_parts){

        // load color mapping assuming it is in the format - color vertex
        std::ifstream inStream(colorMappingFile);
        std::vector<dbgp::colorMapRecordType> colorMapping(vertexPartitions.size());
        std::copy(vertexPartitions.begin(), vertexPartitions.end(), colorMapping.begin());
        std::vector<dbgp::colorMapRecordType>().swap(vertexPartitions);
        while(inStream.good()){
            dbgp::vertexIdType src, dst;
            inStream >> dst; // color. destination will have color
            inStream >> src; // vertex id
            //std::numeric_limits<PartitionValueType>::max() is reserved for deleted kmers
            //cycle kmers will take below value
            dbgp::PartitionValueType pid = std::numeric_limits<PartitionValueType>::max() - 1;
            colorMapping.emplace_back(src, dst, pid);
        }
        inStream.close();
        colorMapping.pop_back(); // remove copy of last element
        std::vector<dbgp::colorMapRecordType>(colorMapping).swap(colorMapping);
//            LOG_IF(comm.rank() == 0, INFO) << "colorMapping on rank 0 : " << colorMapping[441].vertexSrc << 
//            "," << colorMapping[441].vertexDest << "," << colorMapping[441].partitionId;
//            LOG_IF(comm.rank() == 0, INFO) << "colorMapping on rank 0 : " << colorMapping[1441].vertexSrc << 
//            "," << colorMapping[1441].vertexDest << "," << colorMapping[1441].partitionId;

//        for (auto citrx = colorMapping.begin(); citrx != colorMapping.end(); citrx++) {
//            if (citrx->vertexSrc == 44583405185787572) {
//                auto tdist = std::distance(colorMapping.begin(), citrx);
//                LOG(INFO) << comm.rank() << " " << tdist << " v: " << citrx->vertexSrc << " c: " << citrx->vertexDest << " p: " << citrx->partitionId;
//            }
//        }
        mxx::distribute_inplace(colorMapping, comm); // distribute equally
        // sort by destination as it has the color assignment
        mxx::sort(colorMapping.begin(), colorMapping.end(),
                  [&](const dbgp::colorMapRecordType& x,
                      const dbgp::colorMapRecordType& y){
                      return (x.vertexDest < y.vertexDest) ||
                             ((x.vertexDest == y.vertexDest) &&
                              (x.partitionId < y.partitionId));
                  }, comm);

        // propagate color
        auto lastVal = colorMapping.back();
        auto bxitr = colorMapping.rbegin();
        for(; bxitr != colorMapping.rend(); bxitr++){
            if(bxitr->vertexDest != lastVal.vertexDest)
                break;
            lastVal = *bxitr;
        }
        
//#if !NDEBUG
        // check if all the entries in a processor have the same color
        int colorMapCheck = (bxitr != colorMapping.rend());
        auto totalCheck = mxx::allreduce(colorMapCheck, comm);
        if(totalCheck != comm.size()) {
          LOG_IF(!comm.rank(), INFO) << "COLOR MAPPING ASSUMPTION FAILED";
          assert (0);
        }
//#endif

        // update partitionId and copy to kmer_parts
        std::uint64_t cycleCount = 0;
#if !NDEBUG
        assert (std::distance(colorMapping.rbegin(), bxitr));
#endif
        auto prevVal = mxx::right_shift(lastVal, comm);
        //TODO: get rid of extra shift 
        std::uint64_t prevColorSize = mxx::right_shift((std::uint64_t) std::distance(colorMapping.rbegin(), bxitr), comm);
        if(comm.rank() == 0) {
            prevVal.vertexDest = std::numeric_limits<dbgp::vertexIdType>::max();
            prevColorSize = 0; //must be different from 1
        }
        kmer_parts.resize(colorMapping.size());
        auto kitr = kmer_parts.begin();
        for(auto citrx = colorMapping.begin(); citrx != colorMapping.end(); citrx++){
            if(citrx->vertexDest == prevVal.vertexDest) {
#if !NDEBUG
                if (citrx->partitionId != (std::numeric_limits<PartitionValueType>::max() - 1)) {
                    auto tdist = std::distance(colorMapping.begin(), citrx);
                    LOG(INFO) << comm.rank() << " v: " << prevVal.vertexSrc << " c: " << prevVal.vertexDest << " p: " << prevVal.partitionId;
                    if (tdist) {
                        LOG(INFO) << comm.rank() << " " << (tdist-1) << " v: " << (citrx-1)->vertexSrc << " c: " << (citrx-1)->vertexDest << " p: " << (citrx-1)->partitionId;
                    }
                    LOG(INFO) << comm.rank() << " " << tdist << " v: " << citrx->vertexSrc << " c: " << citrx->vertexDest << " p: " << citrx->partitionId;
                }
                assert (citrx->partitionId == (std::numeric_limits<PartitionValueType>::max() - 1));
#endif
                kitr->first = citrx->vertexSrc;
                kitr->second = prevVal.partitionId;
                kitr++;
                prevColorSize++;
            } else {
                // junction node
                if (prevColorSize == 1) {
                    kitr->first = prevVal.vertexSrc;
                    kitr->second = prevVal.partitionId;
                    kitr++;
                }
                // change of color
                prevColorSize = 1;
                prevVal = *citrx;
                // corresponds to cycle
                if (citrx->partitionId == (std::numeric_limits<PartitionValueType>::max() - 1)) {
#if !NDEBUG
                    cycleCount++;
#endif
                    kitr->first = citrx->vertexSrc;
                    kitr->second = citrx->partitionId;
                    kitr++;
                }
            }
        }
        kmer_parts.resize(std::distance(kmer_parts.begin(), kitr));
        std::vector<std::pair<dbgp::vertexIdType, dbgp::PartitionValueType>>(kmer_parts).swap(kmer_parts);
        
#if !NDEBUG
        cycleCount = mxx::reduce(cycleCount, 0, comm);
        LOG_IF(comm.rank() == 0, INFO) << "cycleCount : " << cycleCount;
        //TODO: log data for cycles
#endif
    }


  void loadKmerPartitions(mxx::comm& comm,
                          std::string colorMappingFile,
                          std::string vertexListFile,
                          std::string partitionFile,
                          dbgp::PartitionMapType& part_map){

    dbgp::Timer timer;
    uint64_t localWorkLoad, minLoad, maxLoad, avgLoad;
    // load partitions and color mapping input files
    std::vector<dbgp::colorMapRecordType> vertexPartitions;
    loadVertexPartitions(comm, vertexListFile, partitionFile, vertexPartitions);
#if !NDEBUG
        localWorkLoad = vertexPartitions.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of vertexPartitions. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif
    timer.end_section("finished loading vertices and partitions");
    std::vector<std::pair<dbgp::vertexIdType, dbgp::PartitionValueType>> kmer_parts;
    loadColorMapping(comm, colorMappingFile, vertexPartitions, kmer_parts);
#if !NDEBUG
        localWorkLoad = kmer_parts.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of kmer_parts. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif
    timer.end_section("finished loading color kmer mapping");

    // Insert partition into a distributed map
    part_map.insert(kmer_parts);
    std::vector<std::pair<dbgp::vertexIdType, dbgp::PartitionValueType>>().swap(kmer_parts);
#if !NDEBUG
        localWorkLoad = part_map.local_size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of part_map. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif

    timer.end_section("finished loading kmer partitions");
  }


    PartitionValueType voteSeqPartition(std::vector<dbgp::PartitionValueType>& read_records){
        auto beginItr = read_records.begin();
        auto endItr = read_records.end();
#if !NDEBUG
        assert (beginItr != endItr);
#endif

//        auto partIdCompare = [&](const dbgp::seqPartitionRecordType& x,
//                                 const dbgp::seqPartitionRecordType& y){
//            return (x.partitionId < y.partitionId);
//        };
//        std::sort(beginItr, endItr, partIdCompare);
        std::sort(beginItr, endItr);

        auto currVote = *beginItr;
        auto currVoteCount = 1u;
        beginItr++;
        auto finalVote = currVote;
        auto finalVoteCount = currVoteCount;

        // final voting by majority
        while(beginItr != endItr){
            if(*beginItr == currVote) {
                currVoteCount++;
            } else {
                if(currVoteCount > finalVoteCount){
                    finalVote = currVote;
                    finalVoteCount = currVoteCount;
                }
                currVote = *beginItr;
                currVoteCount = 1u;
            }
            beginItr++;
        }
        
        if(currVoteCount > finalVoteCount){
            finalVote = currVote;
            finalVoteCount = currVoteCount;
        }
        return finalVote;
    }

/*
    void votePartitions(mxx::comm& comm,
                        std::vector<PosIdxTupleType>& kmr_idx, std::size_t& start_offset,
                        dbgp::LocalPartMapType& part_local_map,
                        std::string rpFileName){
        dbgp::Timer timer;
        std::vector<dbgp::PartitionValueType> seq_votes;
#if !NDEBUG
        std::size_t totalKmerLookups = 0, filterKmerLookups = 0;
#endif

        // make the voting choice for each sequence
        std::vector<dbgp::PartitionValueType> read_records;
        auto prtNowItr = kmr_idx.begin() + start_offset;
        auto prtPrvItr = prtNowItr;
        auto prtEndItr = kmr_idx.end();
        for(; prtNowItr != prtEndItr; prtNowItr++){
            if (prtNowItr->second.get_id() != prtPrvItr->second.get_id()) {
                // start of a new read. process previous read
                seq_votes.emplace_back(voteSeqPartition(read_records));
                std::vector<dbgp::PartitionValueType>().swap(read_records);
                prtPrvItr = prtNowItr;
            }
            bliss::kmer::transform::lex_less<dbgp::KmerType> minKmer;
            auto kval = minKmer(prtNowItr->first).getData()[0];
            auto fx_itr = part_local_map.find(kval);
            if(fx_itr != part_local_map.end()){
                read_records.emplace_back(fx_itr->second);
            } else {
                // corresponds to filtered kmer
#if !NDEBUG
                filterKmerLookups++;
#endif
                read_records.emplace_back(std::numeric_limits<PartitionValueType>::max());
            }
#if !NDEBUG
            totalKmerLookups++;
#endif
        }

        //process last read
        assert (prtNowItr == prtEndItr);
        assert (prtPrvItr != prtNowItr);
        seq_votes.emplace_back(voteSeqPartition(read_records));
        std::vector<dbgp::PartitionValueType>().swap(read_records);
        std::vector<dbgp::PartitionValueType>(seq_votes).swap(seq_votes);
        timer.end_section("completed majority voting for all sequences");
        
#if !NDEBUG
        // compute balance for read partitioning
        std::vector<std::uint64_t> readPartitionSizes;
        // partition ids range from 0 to LAST_PART_ID
        // +2 for cycle kmers and deleted kmers
        readPartitionSizes.resize((LAST_PART_ID + 1 + 2), 0);
        for (auto sv : seq_votes) {
            if (sv == std::numeric_limits<PartitionValueType>::max()) {
                // deleted kmers
                (readPartitionSizes[LAST_PART_ID+2])++;
            } else if (sv == (std::numeric_limits<PartitionValueType>::max() - 1)) {
                // cycle kmers
                (readPartitionSizes[LAST_PART_ID+1])++;
            } else {
                assert (sv <= LAST_PART_ID);
                (readPartitionSizes[sv])++;
            }
        }

        totalKmerLookups = mxx::reduce(totalKmerLookups, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "totalKmerLookups  : " << totalKmerLookups;
        filterKmerLookups = mxx::reduce(filterKmerLookups, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "filterKmerLookups : " << filterKmerLookups;

        readPartitionSizes = mxx::reduce(readPartitionSizes, 0, comm);
        LOG_IF(comm.rank() == 0, INFO) << "DEL read count  : " << readPartitionSizes[LAST_PART_ID+2];
        LOG_IF(comm.rank() == 0, INFO) << "CYC read count  : " << readPartitionSizes[LAST_PART_ID+1];
        LOG_IF(comm.rank() == 0, INFO) << "TOT read count  : " << std::accumulate(readPartitionSizes.begin(), readPartitionSizes.end(), 0);
        LOG_IF(comm.rank() == 0, INFO) << "PARTITION count : " << readPartitionSizes.size();
        readPartitionSizes.pop_back(); // remove deleted kmers
        readPartitionSizes.pop_back(); // remove cycle kmers
        LOG_IF(comm.rank() == 0, INFO) << "MIN read count  : " << *std::min_element(readPartitionSizes.begin(), readPartitionSizes.end());
//        std::uint64_t maxReadCount = *std::max_element(readPartitionSizes.begin(), readPartitionSizes.end());
        LOG_IF(comm.rank() == 0, INFO) << "MAX read count  : " << *std::max_element(readPartitionSizes.begin(), readPartitionSizes.end());
//        std::uint64_t totReadCount = std::accumulate(readPartitionSizes.begin(), readPartitionSizes.end(), 0);
        LOG_IF(comm.rank() == 0, INFO) << "TOT read count  : " << std::accumulate(readPartitionSizes.begin(), readPartitionSizes.end(), 0);
        LOG_IF(comm.rank() == 0, INFO) << "PARTITION count : " << readPartitionSizes.size();
//        std::uint64_t avgReadCount = (std::uint64_t) (totReadCount/readPartitionSizes.size());
//        LOG_IF(comm.rank() == 0, INFO) << "AVG read count  : " << avgReadCount;
#endif
        
        // write out the voting choice
        std::ofstream outFilePtr(rpFileName);
        for(auto px : seq_votes){
            outFilePtr << px << std::endl;
        }
        outFilePtr.close();
        timer.end_section("finished writing reads and their partition IDs");
    }
*/

    void votePartitions(mxx::comm& comm,
                        std::vector<PosIdxTupleType>& kmr_idx, std::size_t& start_offset,
                        dbgp::PartitionMapType& part_map,
                        std::string rpFileName){
        dbgp::Timer timer;
        std::ofstream outFilePtr(rpFileName);
#if !NDEBUG
        uint64_t localWorkLoad, minLoad, maxLoad, avgLoad;
        std::size_t totalKmerLookups = 0, filterKmerLookups = 0;
        // compute balance for read partitioning
        std::vector<std::uint64_t> readPartitionSizes;
        // partition ids range from 0 to LAST_PART_ID
        // +2 for cycle kmers and deleted kmers
        readPartitionSizes.resize((LAST_PART_ID + 1 + 2), 0);
#endif
        
        auto prtNowItr = kmr_idx.begin() + start_offset;
        auto prtPrvItr = prtNowItr;
        auto prtEndItr = prtNowItr;
        if (std::distance(prtNowItr, kmr_idx.end()) < BATCH_SIZE) {
            prtEndItr = kmr_idx.end();
        } else {
            prtPrvItr += (BATCH_SIZE - 1);
            prtEndItr += BATCH_SIZE;
            // find a read boundary
            while ((prtEndItr != kmr_idx.end()) && (prtPrvItr->second.get_id() == prtEndItr->second.get_id())) {
                prtEndItr++;
            }
        }
        
        //TODO: Handle the case when only a subset of the processes have elements remaining 
        while(prtNowItr != kmr_idx.end()) {
            auto local_idx_size = std::distance(prtNowItr, prtEndItr);
            auto local_kmers = std::vector<dbgp::vertexIdType>(local_idx_size);
            auto kit = prtNowItr;
            auto wit = local_kmers.begin();
            bliss::kmer::transform::lex_less<dbgp::KmerType> minKmer;
            for(; kit != prtEndItr; kit++, wit++){
                *wit = minKmer(kit->first).getData()[0];
            }
#if !NDEBUG
            localWorkLoad = local_kmers.size();
            maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
            minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
            avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
            LOG_IF(comm.rank() == 0, INFO) << "Distribution of local_kmers. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif
            std::vector<std::pair<dbgp::vertexIdType, dbgp::PartitionValueType>> part_found;
            part_found = part_map.find(local_kmers);
            std::vector<dbgp::vertexIdType>().swap(local_kmers); // free memory
#if !NDEBUG
            localWorkLoad = part_found.size();
            maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
            minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
            avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
            LOG_IF(comm.rank() == 0, INFO) << "Distribution of part_found. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif
            dbgp::LocalPartMapType part_local_map;
            part_local_map.insert(part_found.begin(), part_found.end());
            std::vector<std::pair<dbgp::vertexIdType, dbgp::PartitionValueType>>().swap(part_found); // free memory
#if !NDEBUG
            localWorkLoad = part_local_map.size();
            maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
            minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
            avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
            LOG_IF(comm.rank() == 0, INFO) << "Distribution of part_local_map. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
#endif
            
            std::vector<dbgp::PartitionValueType> seq_votes;
            std::vector<dbgp::PartitionValueType> read_records;
            auto qrtNowItr = prtNowItr;
            auto qrtPrvItr = prtNowItr;
            auto qrtEndItr = prtEndItr;
            for(; qrtNowItr != qrtEndItr; qrtNowItr++){
                if (qrtNowItr->second.get_id() != qrtPrvItr->second.get_id()){
                    // start of a new read. process previous read
                    seq_votes.emplace_back(voteSeqPartition(read_records));
                    std::vector<dbgp::PartitionValueType>().swap(read_records);
                    qrtPrvItr = qrtNowItr;
                }
                auto kval = minKmer(qrtNowItr->first).getData()[0];
                auto fx_itr = part_local_map.find(kval);
                if(fx_itr != part_local_map.end()){
                    read_records.emplace_back(fx_itr->second);
                } else {
                    // corresponds to filtered kmer
#if !NDEBUG
                    filterKmerLookups++;
#endif
                    read_records.emplace_back(std::numeric_limits<PartitionValueType>::max());
                }
#if !NDEBUG
                totalKmerLookups++;
#endif
            }
            part_local_map.reset(); // free memory

            //process last read
            assert (qrtNowItr == qrtEndItr);
            assert (qrtPrvItr != qrtNowItr);
            seq_votes.emplace_back(voteSeqPartition(read_records));
            std::vector<dbgp::PartitionValueType>().swap(read_records);            
            std::vector<dbgp::PartitionValueType>(seq_votes).swap(seq_votes);            
#if !NDEBUG
            for (auto &sv : seq_votes) {
                if (sv == std::numeric_limits<PartitionValueType>::max()) {
                    // deleted kmers
                    (readPartitionSizes[LAST_PART_ID+2])++;
                } else if (sv == (std::numeric_limits<PartitionValueType>::max() - 1)) {
                    // cycle kmers
                    (readPartitionSizes[LAST_PART_ID+1])++;
                } else {
                    assert (sv <= LAST_PART_ID);
                    (readPartitionSizes[sv])++;
                }
            }
#endif
            for(auto &px : seq_votes){
                outFilePtr << px << std::endl;
            }
            std::vector<dbgp::PartitionValueType>().swap(seq_votes); // free memory
            
            // prepare for next batch
            prtNowItr = prtEndItr;
            if (std::distance(prtNowItr, kmr_idx.end()) < BATCH_SIZE) {
                prtEndItr = kmr_idx.end();
            } else {
                prtPrvItr = prtNowItr + BATCH_SIZE - 1;
                prtEndItr = prtNowItr + BATCH_SIZE;
                // find a read boundary
                while ((prtEndItr != kmr_idx.end()) && (prtPrvItr->second.get_id() == prtEndItr->second.get_id())) {
                    prtEndItr++;
                }
            }            
            timer.end_section("completed majority voting for a batch");
        }
        
        outFilePtr.close();
#if !NDEBUG
        totalKmerLookups = mxx::reduce(totalKmerLookups, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "totalKmerLookups  : " << totalKmerLookups;
        filterKmerLookups = mxx::reduce(filterKmerLookups, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "filterKmerLookups : " << filterKmerLookups;

        readPartitionSizes = mxx::reduce(readPartitionSizes, 0, comm);
        LOG_IF(comm.rank() == 0, INFO) << "DEL read count  : " << readPartitionSizes[LAST_PART_ID+2];
        LOG_IF(comm.rank() == 0, INFO) << "CYC read count  : " << readPartitionSizes[LAST_PART_ID+1];
        LOG_IF(comm.rank() == 0, INFO) << "TOT read count  : " << std::accumulate(readPartitionSizes.begin(), readPartitionSizes.end(), 0);
        LOG_IF(comm.rank() == 0, INFO) << "PARTITION count : " << readPartitionSizes.size();
        readPartitionSizes.pop_back(); // remove deleted kmers
        readPartitionSizes.pop_back(); // remove cycle kmers
        LOG_IF(comm.rank() == 0, INFO) << "MIN read count  : " << *std::min_element(readPartitionSizes.begin(), readPartitionSizes.end());
        LOG_IF(comm.rank() == 0, INFO) << "MAX read count  : " << *std::max_element(readPartitionSizes.begin(), readPartitionSizes.end());
        LOG_IF(comm.rank() == 0, INFO) << "TOT read count  : " << std::accumulate(readPartitionSizes.begin(), readPartitionSizes.end(), 0);
        LOG_IF(comm.rank() == 0, INFO) << "PARTITION count : " << readPartitionSizes.size();
#endif
        timer.end_section("finished writing read partition IDs");
    }


    void partitionReads(mxx::comm& comm, std::string inputFile, std::string partitionFile,
                        std::string rpFileName, std::string colorMappingFile,
                        std::string vertexListFile, std::string filteredKmersFile){

        std::size_t start_offset = 0;
        // construct index
        LOG_IF(!comm.rank(), INFO) << "Begin constructing kmer position index";
        std::vector<PosIdxTupleType> kmr_idx;
        buildPositionIndex(comm, kmr_idx, start_offset, inputFile);
        LOG_IF(!comm.rank(), INFO) << "End constructing kmer position index";

        // load partitions
        LOG_IF(!comm.rank(), INFO) << "Begin loading kmer partitions";
        // google dense hash map uses special keys
        dbgp::PartitionMapType part_map(comm);
        // we ignore start_offset as it does NOT affect correctness
        loadKmerPartitions(comm, colorMappingFile, vertexListFile, partitionFile, part_map);
        LOG_IF(!comm.rank(), INFO) << "End loading kmer partitions";

        // generate records and vote partitions
        LOG_IF(!comm.rank(), INFO) << "Begin sequence voting";
        votePartitions(comm, kmr_idx, start_offset, part_map, rpFileName);
        LOG_IF(!comm.rank(), INFO) << "End sequence voting";
    }


}
