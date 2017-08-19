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

//Own includes
#include "utils/mpi_utils.hpp"
#include "utils/io_utils.hpp"
#include "dbgp/timer.hpp"

//External includes
#include "extutils/logging.hpp"
#include "extutils/argvparser.hpp"
#include "mxx/comm.hpp"
#include "mxx/utils.hpp"
#include "mxx/reduction.hpp"
#include "mxx/sort.hpp"

INITIALIZE_EASYLOGGINGPP
using namespace CommandLineProcessing;

struct InputArgs{
  std::string partFileName;
  std::string samFileName;
  uint32_t kmerLength;
  uint32_t readLength;
  int16_t scoreThreshold;

  InputArgs(){
    kmerLength = 0;
    readLength = 0;
    scoreThreshold = std::numeric_limits<int16_t>::min();
  }
};

struct EvaluationStats{
    uint64_t totalPairs;
    uint64_t intraPairs;
    uint64_t interPairs;

    EvaluationStats(){
        totalPairs = 0;
        intraPairs = 0;
        interPairs = 0;
    }

    void print(mxx::comm& comm){
        auto tpv = mxx::allreduce(totalPairs, comm);
        auto gpv = mxx::allreduce(intraPairs, comm);
        auto bpv = mxx::allreduce(interPairs, comm);
        LOG_IF(!comm.rank(), INFO) << "Total : [" << tpv << "] Intra : [" << gpv << "] Inter : [" << bpv << "]";
    }
};

using PIDT = uint16_t; //PartitionIDType

struct SAMRecord{
    uint64_t readID;
    uint16_t samFlag;
    int16_t alignScore;
    uint32_t refName;
    uint32_t mapPosition;

//    SAMRecord(){
////        alignScore = 0;
//    }
};

MXX_CUSTOM_STRUCT(SAMRecord, readID, samFlag, alignScore, refName, mapPosition);

struct ReadRecord{
    uint16_t samFlag;
    int16_t alignScore;
    uint32_t refName;
    uint32_t mapPosition;
    PIDT partID;

//    ReadRecord(){
////        alignScore = 0;
//    }
};

MXX_CUSTOM_STRUCT(ReadRecord, samFlag, alignScore, refName, mapPosition, partID);

int parse_args(int argc, char* argv[], mxx::comm& comm, InputArgs& in_args){

  ArgvParser cmd;
  const char* pt_arg = "partition_file";
  const char* sm_arg = "sam_file";
  const char* km_arg = "kmer_length";
  const char* rl_arg = "read_length";
  const char* st_arg = "score_threshold";

  cmd.setIntroductoryDescription("Partitioning Evaluation");
  cmd.addErrorCode(0, "Success");
  cmd.addErrorCode(1, "Error");

  cmd.setHelpOption("h", "help", "Print this help page");

  cmd.defineOption(pt_arg, "Partition File Prefix : containing partition id for reads",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(pt_arg, "p");

  cmd.defineOption(sm_arg, "SAM File : containing alignments of reads",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(sm_arg, "s");

  cmd.defineOption(km_arg, "Kmer Length",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(km_arg, "k");

  cmd.defineOption(rl_arg, "Read Length",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(rl_arg, "l");

  cmd.defineOption(st_arg, "Score Threshold",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(st_arg, "t");

  int result = cmd.parse(argc, argv);

  //Make sure we get the right command line args
  if (result != ArgvParser::NoParserError) {
    if (!comm.rank()) std::cout << cmd.parseErrorDescription(result) << "\n";
    return 1;
  }

  if(cmd.foundOption(pt_arg)) {
      std::string outPfx = cmd.optionValue(pt_arg);
      std::stringstream outs;
      outs << outPfx << "_"
           << (comm.rank() < 10 ? "000" :
               (comm.rank() < 100 ? "00" :
                (comm.rank() < 1000 ? "0" : "")))
           << comm.rank() << ".txt";
      in_args.partFileName = outs.str();
  } else {
      if(comm.rank() == 0)
          std::cout << "Required option missing: " << pt_arg  << std::endl;
      return 1;
  }

  if(cmd.foundOption(sm_arg)) {
      in_args.samFileName = cmd.optionValue(sm_arg);
  } else {
      if(comm.rank() == 0)
          std::cout << "Required option missing: " << sm_arg  << std::endl;
      return 1;
  }

  if(cmd.foundOption(km_arg)) {
      in_args.kmerLength = std::stoi(cmd.optionValue(km_arg));
      if(in_args.kmerLength == 0) {
          std::cout << "Invalid argument for : " << km_arg  << std::endl;
          return 1;
      }
  } else {
      if(comm.rank() == 0)
          std::cout << "Required option missing: " << km_arg  << std::endl;
      return 1;
  }

  if(cmd.foundOption(rl_arg)) {
      in_args.readLength = std::stoi(cmd.optionValue(rl_arg));
      if(in_args.readLength == 0) {
          std::cout << "Invalid argument for : " << rl_arg  << std::endl;
          return 1;
      }
  } else {
      if(comm.rank() == 0)
          std::cout << "Required option missing: " << rl_arg  << std::endl;
      return 1;
  }

  if(cmd.foundOption(st_arg)) {
      in_args.scoreThreshold = (int16_t) std::stoi(cmd.optionValue(st_arg));
  }

  LOG_IF(!comm.rank(), INFO) << "-----------------------------------------------------";
  LOG_IF(!comm.rank(), INFO) << "Partition File     : " << in_args.partFileName;
  LOG_IF(!comm.rank(), INFO) << "SAM File           : " << in_args.samFileName;
  LOG_IF(!comm.rank(), INFO) << "Kmer Length        : " << in_args.kmerLength;
  LOG_IF(!comm.rank(), INFO) << "Read Length        : " << in_args.readLength;
  LOG_IF(!comm.rank(), INFO) << "Score Threshold    : " << in_args.scoreThreshold;
  LOG_IF(!comm.rank(), INFO) << "-----------------------------------------------------" ;

  return 0;
}

void loadPartitionRecords(mxx::comm& comm, InputArgs& in_args, std::vector<PIDT>& partRecords){
    std::ifstream inStream(in_args.partFileName);
    while(inStream.good()){
        PIDT pvx;
        inStream >> pvx;
        partRecords.emplace_back(pvx);
    }
    inStream.close();
    partRecords.pop_back(); // remove copy of last element
    std::vector<PIDT>(partRecords).swap(partRecords);
        LOG_IF(comm.rank() == 0, INFO) << " " << partRecords.front();
        LOG_IF(comm.rank() == 0, INFO) << " " << partRecords.back();
}

void loadSAMRecords(mxx::comm& comm, InputArgs& in_args, std::vector<SAMRecord>& samRecords) {

    std::vector<std::string> samData;
    read_block(comm, in_args.samFileName, samData);

    // load SAM records
    samRecords.resize(samData.size());
    std::size_t j = 0;
    std::size_t headerLineCount = 0;
    for(std::size_t i = 0; i < samData.size(); i++){
        if(samData[i][0] == '@') {
            headerLineCount++;
            continue;
        }
        std::stringstream ss(samData[i]);
        SAMRecord rdd;
        
        ss >> rdd.readID;
        ss >> rdd.samFlag;

        if (rdd.samFlag & 0x4) {
            // unmapped read
            // ignore reference, position, and alignScore
            rdd.refName = std::numeric_limits<uint32_t>::max();
            rdd.mapPosition = std::numeric_limits<uint32_t>::max();
            rdd.alignScore = std::numeric_limits<int16_t>::min();
        } else {
            ss >> rdd.refName;
            ss >> rdd.mapPosition;
            std::string ignore;
            for(std::size_t k = 5; k < 12; k++){
                ss >> ignore;
            }
            ss >> ignore;
            rdd.alignScore = (int16_t) std::stoi(ignore.substr(5));
        }
        
        samRecords[j] = rdd;
        j++;
    }
    std::vector<std::string>().swap(samData); // free memory
    samRecords.resize(j);
    std::vector<SAMRecord>(samRecords).swap(samRecords);
    
        headerLineCount = mxx::reduce(headerLineCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "headerLineCount : " << headerLineCount;
        LOG_IF(comm.rank() == 0, INFO) << " " << samRecords.front().readID
                                       << " " << samRecords.front().samFlag
                                       << " " << samRecords.front().refName
                                       << " " << samRecords.front().mapPosition
                                       << " " << samRecords.front().alignScore;
        LOG_IF(comm.rank() == 0, INFO) << " " << samRecords.back().readID
                                       << " " << samRecords.back().samFlag
                                       << " " << samRecords.back().refName
                                       << " " << samRecords.back().mapPosition
                                       << " " << samRecords.back().alignScore;
}

void collapseSAMRecords(mxx::comm& comm, InputArgs& in_args,
                        std::vector<SAMRecord>& samRecords, std::vector<ReadRecord>& allRecords) {

    uint64_t localWorkLoad, minLoad, maxLoad, avgLoad;

    // sort SAM records by readID
    mxx::sort(samRecords.begin(), samRecords.end(),
              [&](const SAMRecord& x,
                  const SAMRecord& y){
                  return (x.readID < y.readID);
              }, comm);

    // handle ambiguous mapping
    std::size_t start_offset, end_offset;
    std::vector<SAMRecord> straddle_region;
    auto readIDCompare = [&](const SAMRecord& x,
                             const SAMRecord& y){
        return (x.readID == y.readID);
    };
    shiftStraddlingRegion(comm, samRecords, start_offset, end_offset, straddle_region, readIDCompare);
    
    std::size_t ambiguousReadCount = 0;
    auto samNowItr = samRecords.begin() + start_offset;
    auto samPrvItr = samNowItr;
    allRecords.resize(samRecords.size() + straddle_region.size());
    ReadRecord rd_record;
    std::size_t ar_count = 0;
    std::size_t readCopies = 0;
    for (; samNowItr != samRecords.begin() + end_offset; samNowItr++) {
        if (samNowItr->readID == samPrvItr->readID) {
            readCopies++;
        } else {
            rd_record.samFlag = samPrvItr->samFlag;
            rd_record.alignScore = samPrvItr->alignScore;
            rd_record.refName = samPrvItr->refName;
            rd_record.mapPosition = samPrvItr->mapPosition;
            rd_record.partID = std::numeric_limits<PIDT>::max();
            if (readCopies == 1) {
                // unique read
            } else {
                // convert ambiguous read to unmapped read
                rd_record.samFlag = 0x4; // segment unmapped
                ambiguousReadCount++;
            }
            allRecords[ar_count] = rd_record;
            ar_count++;
            readCopies = 1;
            samPrvItr = samNowItr;
        }
    }
    // process last read
    assert (samPrvItr != samNowItr);
    assert (samNowItr == samRecords.begin() + end_offset);
            rd_record.samFlag = samPrvItr->samFlag;
            rd_record.alignScore = samPrvItr->alignScore;
            rd_record.refName = samPrvItr->refName;
            rd_record.mapPosition = samPrvItr->mapPosition;
            rd_record.partID = std::numeric_limits<PIDT>::max();
            if (readCopies == 1) {
                // unique read
            } else {
                // convert ambiguous read to unmapped read
                rd_record.samFlag = 0x4; // segment unmapped
                ambiguousReadCount++;
            }
            allRecords[ar_count] = rd_record;
            ar_count++;
    
    // process straddle_region
    if (straddle_region.size() == 0) {
        // last rank does not have straddle region
        assert (comm.rank() == (comm.size() - 1));
    } else {
        rd_record.samFlag = straddle_region.front().samFlag;
        rd_record.alignScore = straddle_region.front().alignScore;
        rd_record.refName = straddle_region.front().refName;
        rd_record.mapPosition = straddle_region.front().mapPosition;
        rd_record.partID = std::numeric_limits<PIDT>::max();
        if (straddle_region.size() == 1) {
            // unique read
        } else {
            // convert ambiguous read to unmapped read
            rd_record.samFlag = 0x4; // segment unmapped
            ambiguousReadCount++;
        }
        allRecords[ar_count] = rd_record;
        ar_count++;
    }

    std::vector<SAMRecord>().swap(samRecords); // free memory
    allRecords.resize(ar_count);
    std::vector<ReadRecord>(allRecords).swap(allRecords);
    
        ambiguousReadCount = mxx::reduce(ambiguousReadCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "ambiguousReadCount : " << ambiguousReadCount;
}

void loadRecords(mxx::comm& comm, InputArgs& in_args, std::vector<ReadRecord>& allRecords) {

    dbgp::Timer timer;
    uint64_t localWorkLoad, minLoad, maxLoad, avgLoad;
    
    // load from SAM file
    std::vector<SAMRecord> samRecords;
    loadSAMRecords(comm, in_args, samRecords);
        localWorkLoad = samRecords.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of samRecords. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
    timer.end_section("finished loading SAM records");
    
    // load from partition file
    std::vector<PIDT> partRecords;
    loadPartitionRecords(comm, in_args, partRecords);
        localWorkLoad = partRecords.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of partRecords_befr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
    mxx::stable_distribute_inplace(partRecords, comm); // distribute equally
        localWorkLoad = partRecords.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of partRecords_aftr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
    timer.end_section("finished loading partition records");
    
    collapseSAMRecords(comm, in_args, samRecords, allRecords);
        localWorkLoad = allRecords.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of allRecords_befr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
    mxx::stable_distribute_inplace(allRecords, comm); // distribute equally
        localWorkLoad = allRecords.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of allRecords_aftr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
    timer.end_section("finished populating all records");

        // verify that sizes match
        int sizeCheck = (partRecords.size() == allRecords.size());
        auto totalCheck = mxx::allreduce(sizeCheck, comm);
        if(totalCheck != comm.size()) {
          LOG(INFO) << comm.rank() << " partRecords size : " << partRecords.size();
          LOG(INFO) << comm.rank() << " allRecords size :  " << allRecords.size();
          LOG_IF(!comm.rank(), INFO) << "partRecords and allRecords SIZES NOT EQUAL";
          assert (0);
        }

    LOG_IF(comm.rank() == 0, INFO) << allRecords.front().partID;
    LOG_IF(comm.rank() == 0, INFO) << allRecords.back().partID;
    auto pitr = partRecords.begin();
    for (auto aitr = allRecords.begin(); aitr != allRecords.end(); aitr++, pitr++) {
        aitr->partID = *pitr;
    }
    LOG_IF(comm.rank() == 0, INFO) << allRecords.front().partID;
    LOG_IF(comm.rank() == 0, INFO) << allRecords.back().partID;
    
    timer.end_section("finished loading all records");
}

void filterRecords(mxx::comm& comm, InputArgs& in_args, std::vector<ReadRecord>& allRecords){
    dbgp::Timer timer;
    std::size_t inferiorReadCount = 0;
    std::size_t reversedReadCount = 0;
    std::size_t unmappedReadCount = 0;
    std::size_t reveppedReadCount = 0;
    for (auto aitr = allRecords.begin(); aitr != allRecords.end(); aitr++) {
        assert (aitr->samFlag < 0x8000);
        if (aitr->samFlag & 0x10) {
            reversedReadCount++;
            if (aitr->samFlag & 0x4) {
                reveppedReadCount++;
            }
        }
        if (aitr->alignScore < in_args.scoreThreshold) {
            if (aitr->samFlag & 0x4) {
                // already unmapped read
            } else {
                inferiorReadCount++;
                // convert inferior read to unmapped read
                aitr->samFlag = 0x4; // segment unmapped
            }
        }
        if (aitr->samFlag & 0x4) {
            unmappedReadCount++;
            // redesignate unmapped reads
            aitr->samFlag = 0x8000;
        }
    }
    
        reversedReadCount = mxx::reduce(reversedReadCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "reversedReadCount : " << reversedReadCount;
        reveppedReadCount = mxx::reduce(reveppedReadCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "reveppedReadCount : " << reveppedReadCount;
        inferiorReadCount = mxx::reduce(inferiorReadCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "inferiorReadCount : " << inferiorReadCount;
//        LOG_IF(comm.rank() == 0, INFO) << "unmappedReadCount : " << unmappedReadCount;
        unmappedReadCount = mxx::reduce(unmappedReadCount, 0, std::plus<std::size_t>(), comm);
        LOG_IF(comm.rank() == 0, INFO) << "unmappedReadCount : " << unmappedReadCount;
        LOG_IF(comm.rank() == 0, INFO) << "unmappedReadCount includes inferiorReadCount and ambiguousReadCount";
    
//    LOG_IF(comm.rank() == 0, INFO) << allRecords.back().samFlag;
    // drop unmapped reads
    std::sort(allRecords.begin(), allRecords.end(),
              [&](const ReadRecord& x,
                  const ReadRecord& y){
                  return (x.samFlag < y.samFlag);
              });
//    LOG_IF(comm.rank() == 0, INFO) << allRecords.back().samFlag;
    
    auto aitr = allRecords.rbegin();
    for (; aitr != allRecords.rend(); aitr++) {
        if (aitr->samFlag != 0x8000) break;
    }
    auto adist = std::distance(allRecords.rbegin(), aitr);
//    LOG(INFO) << comm.rank() << " " << adist << " " << allRecords.size();
    allRecords.resize(allRecords.size() - adist);
//    LOG(INFO) << comm.rank() << " " << allRecords.size();
    std::vector<ReadRecord>(allRecords).swap(allRecords);
    timer.end_section("finished filtering records");
}

void evaluatePairs(std::vector<ReadRecord>& read_records, std::size_t& size_before,
                   InputArgs& in_args, EvaluationStats& eStats){
    auto currItr = read_records.begin();
    auto prevItr = currItr;
    
    for (; prevItr != (read_records.begin() + size_before); prevItr++) {
        assert (prevItr->refName != std::numeric_limits<uint32_t>::max());
        currItr = prevItr + 1;
        while ((currItr != read_records.end()) &&
               (currItr->refName == prevItr->refName) &&
               ((prevItr->mapPosition + in_args.readLength) >= (currItr->mapPosition + in_args.kmerLength))) {
            eStats.totalPairs += 1;
            if (currItr->partID == prevItr->partID) {
                eStats.intraPairs += 1;
            } else {
                eStats.interPairs += 1;
            }
            currItr++;
        }        
    }
}

void evaluateQuality(mxx::comm& comm, InputArgs& in_args,
                     std::vector<ReadRecord>& allRecords,
                     EvaluationStats& eStats){
    dbgp::Timer timer;
    uint64_t localWorkLoad, minLoad, maxLoad, avgLoad;
    
    // sort by ref name and pos
        localWorkLoad = allRecords.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of evaluatePairs_befr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
    mxx::distribute_inplace(allRecords, comm); // distribute equally
        localWorkLoad = allRecords.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of evaluatePairs_aftr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
    mxx::sort(allRecords.begin(), allRecords.end(),
              [&](const ReadRecord& x,
                  const ReadRecord& y){
                  return (x.refName < y.refName) ||
                      ((x.refName == y.refName) &&
                       (x.mapPosition < y.mapPosition));
              }, comm);

    // shift straddle region
    auto posCompare = [&](const ReadRecord& x,
                          const ReadRecord& y){
        if (x.refName == y.refName) {
            assert (x.mapPosition >= y.mapPosition);
        }
        return (x.refName == y.refName) &&
               ((y.mapPosition + in_args.readLength) >= (x.mapPosition + in_args.kmerLength));
    };

    std::vector<ReadRecord> straddleRegion;
    shiftStraddlingRegion(comm, allRecords, straddleRegion, posCompare);
        localWorkLoad = straddleRegion.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of straddleRegion_befr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;
    LOG_IF(comm.rank() == 0, INFO) << "allRecords size : " << allRecords.size();

/*
    localWorkLoad = straddleRegion.size();
    maxLoad = mxx::allreduce(localWorkLoad, mxx::max<uint64_t>(), comm);
    if (straddleRegion.size() == maxLoad) {
        LOG(INFO) << comm.rank() << " - straddleRegion";
        for (auto srx : straddleRegion) {
            LOG(INFO) << srx.refName << " " << srx.mapPosition << " " << srx.partID;
        }
    }
*/
    
    auto size_before = allRecords.size();
    if (straddleRegion.size()) {
        allRecords.resize(allRecords.size() + straddleRegion.size());
        std::copy(straddleRegion.begin(), straddleRegion.end(), (allRecords.begin() + size_before));
        std::vector<ReadRecord>(allRecords).swap(allRecords);
    } else {
        // last rank does not have straddle region
        assert (comm.rank() == (comm.size() - 1));
    }
        localWorkLoad = allRecords.size();
        maxLoad = mxx::reduce(localWorkLoad, 0, mxx::max<uint64_t>(), comm);
        minLoad = mxx::reduce(localWorkLoad, 0, mxx::min<uint64_t>(), comm);
        avgLoad = (uint64_t) (mxx::reduce(localWorkLoad, 0, std::plus<uint64_t>(), comm) / comm.size());
        LOG_IF(comm.rank() == 0, INFO) << "Distribution of straddleRegion_aftr. min-avg-max : " << minLoad << "," << avgLoad << "," << maxLoad;

    LOG_IF(comm.rank() == 0, INFO) << "straddleRegion size : " << straddleRegion.size();
    LOG_IF(comm.rank() == 0, INFO) << "allRecords size : " << allRecords.size();
    LOG_IF(comm.rank() == 0, INFO) << "size_before : " << size_before;
    // count the number of pairs
    evaluatePairs(allRecords, size_before, in_args, eStats);
    timer.end_section("finished evaluating quality");
}

int main(int argc, char* argv[]) {

  // Initialize the MPI library:
  MPI_Init(&argc, &argv);

  //Initialize the communicator
  mxx::comm comm;

  //Print mpi rank distribution
  mxx::print_node_distribution();

  // COMMAND LINE ARGUMENTS
  LOG_IF(!comm.rank(), INFO) << "Start EVAL-PARTITION";

  //Parse command line arguments
  InputArgs cargs;

  if(parse_args(argc, argv, comm, cargs))
      return 1;

  // load parition and SAM data
  LOG_IF(!comm.rank(), INFO) << "Begin loading read records";
  std::vector<ReadRecord> allRecords;
  loadRecords(comm, cargs, allRecords);
  LOG_IF(!comm.rank(), INFO) << "End loading read records";

  // filter records based on quality
  LOG_IF(!comm.rank(), INFO) << "Begin filtering read records";
  filterRecords(comm, cargs, allRecords);
  LOG_IF(!comm.rank(), INFO) << "End filtering read records";

  LOG_IF(!comm.rank(), INFO) << "Begin evaluating quality";
  EvaluationStats evStats;
//  evStats.print(comm);
  evaluateQuality(comm, cargs, allRecords, evStats);
  evStats.print(comm);
  LOG_IF(!comm.rank(), INFO) << "End evaluating quality";

  MPI_Finalize();
  return(0);
}
