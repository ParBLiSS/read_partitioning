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

//External includes
#include "extutils/logging.hpp"
#include "extutils/argvparser.hpp"
#include "mxx/reduction.hpp"
#include "mxx/utils.hpp"

#include "dbgp/compactdeBruijnGraph.cpp"
#include "dbgp/readPartition.cpp"
INITIALIZE_EASYLOGGINGPP
using namespace CommandLineProcessing;

struct InputArgs{
  char runType;
  std::string ipFileName;
  std::string ajFilePrefix;
  std::string filteredKmersPrefix; //Stores (low frequency) kmers filtered out
  std::string colorKmerMapPrefix; //Stores colors and kmers making up the colors
  std::string colorVertexListPrefix; //Stores graph nodes in color space (before transformation to contiguous space)
  std::string ptFileName;
  std::string rpFilePrefix;
  uint64_t minKmerFreq;
  char dbgNodeWtType; 

  InputArgs(){
    minKmerFreq = 5;
    dbgNodeWtType = 'c';
  }
};


namespace dbgp{
    void init(){
        dbgp::compactInit();
    }
};
int parse_args(int argc, char* argv[], mxx::comm& comm,
               InputArgs& in_args){
  ArgvParser cmd;
  const char* rt_arg = "run_type";
  const char* ip_arg = "input_file";
  const char* aj_arg = "adj_list_prefix";
  const char* fk_arg = "filtered_kmers_prefix"; //Stores (low frequency) kmers filtered out
  const char* ck_arg = "color_kmer_map_prefix"; //Stores colors and kmers making up the colors
  const char* cv_arg = "color_vert_list_prefix"; //Stores graph nodes in color space (before transformation to contiguous space)
  const char* pf_arg = "dbg_part_file";
  const char* rp_arg = "read_part_prefix";
  const char* mk_arg = "min_kmer";
  const char* wt_arg = "db_node_wt_type";

  cmd.setIntroductoryDescription("Read Partitioning");

  cmd.addErrorCode(0, "Success");
  cmd.addErrorCode(1, "Error");

  cmd.setHelpOption("h", "help", "Print this help page");
  cmd.defineOption(rt_arg, "Run Type. Should be one of 'c' or 'p', "
                   "where 'c' constructs and compacts the DBG, "
                   "and 'p' partitions reads based on DBG node partitions.",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(rt_arg, "r");

  cmd.defineOption(ip_arg, "Input FASTA File",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(ip_arg, "i");

  cmd.defineOption(aj_arg, "Adjacency List Prefix. Required when run type is 'c'",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(aj_arg, "a");

  cmd.defineOption(fk_arg, "Filtered Kmers File Prefix",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(fk_arg, "f");

  cmd.defineOption(ck_arg, "Color Kmer Mapping File Prefix",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(ck_arg, "c");

  cmd.defineOption(cv_arg, "Color Vertex List File Prefix",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOptionAlternative(cv_arg, "v");

  cmd.defineOption(pf_arg, "DBG Partition File. Required when run type is 'p'",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(pf_arg, "p");

  cmd.defineOption(rp_arg, "Read Partition Prefix. Required when run type is 'p'",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(rp_arg, "o");

  cmd.defineOption(mk_arg,
                   "(k+1)-mers below this frequency will be filtered out. Default is 5",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(mk_arg, "k");

  cmd.defineOption(wt_arg, "de Bruijn graph node weight type. Should be one of 'c' or 's', "
                   "where 'c' stands for count and 's' for sum. Required when run type is 'c'.",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOptionAlternative(wt_arg, "w");

  int result = cmd.parse(argc, argv);

  //Make sure we get the right command line args
  if (result != ArgvParser::NoParserError) {
    if (!comm.rank()) std::cout << cmd.parseErrorDescription(result) << "\n";
    return 1;
  }

  if(cmd.foundOption(rt_arg)) {
    in_args.runType = cmd.optionValue(rt_arg)[0];
    if(in_args.runType != 'c' && in_args.runType != 'p'){
      if(comm.rank() == 0)
          std::cout << in_args.runType << " is invalid option of " << rt_arg
                    << ". Option should be one of 'c' or 'p'. " << std::endl;
      return 1;
    }
  } else {
    if(comm.rank() == 0)
      std::cout << "Required option missing: " << rt_arg  << std::endl;
    return 1;
  }

  if(cmd.foundOption(ip_arg)) {
    in_args.ipFileName = cmd.optionValue(ip_arg);
  } else {
    if(comm.rank() == 0)
      std::cout << "Required option missing: " << ip_arg  << std::endl;
    return 1;
  }

  if (in_args.runType == 'c') {
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

  if(cmd.foundOption(ck_arg)) { //Stores colors and kmers making up the colors
      std::string outPfx = cmd.optionValue(ck_arg);
      std::stringstream outs;
      outs << outPfx << "_"
           << (comm.rank() < 10 ? "000" :
               (comm.rank() < 100 ? "00" :
                (comm.rank() < 1000 ? "0" : "")))
           << comm.rank() << ".txt";
      in_args.colorKmerMapPrefix = outs.str();
  } else {
    if(comm.rank() == 0)
      std::cout << "Required option missing: " << ck_arg  << std::endl;
    return 1;
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

  if(in_args.runType == 'p') {
    if(cmd.foundOption(pf_arg)) {
//      in_args.ptFileName = cmd.optionValue(pf_arg);
      //Updated from file to prefix
      std::string outPfx = cmd.optionValue(pf_arg);
      std::stringstream outs;
      outs << outPfx << "_"
           << (comm.rank() < 10 ? "000" :
               (comm.rank() < 100 ? "00" :
                (comm.rank() < 1000 ? "0" : "")))
           << comm.rank() << ".txt";
      in_args.ptFileName = outs.str();
    } else {
      if(comm.rank() == 0)
        std::cout << "Required option missing: " << pf_arg  << std::endl;
      return 1;
    }
  }

  if (in_args.runType == 'p') {
    if(cmd.foundOption(rp_arg)) {
        std::string outPfx = cmd.optionValue(rp_arg);
        std::stringstream outs;
        outs << outPfx << "_"
             << (comm.rank() < 10 ? "000" :
                 (comm.rank() < 100 ? "00" :
                  (comm.rank() < 1000 ? "0" : "")))
             << comm.rank() << ".txt";
        in_args.rpFilePrefix = outs.str();
    } else {
      if(comm.rank() == 0)
        std::cout << "Required option missing: " << rp_arg  << std::endl;
      return 1;
    }
  }

  if(cmd.foundOption(mk_arg))
    in_args.minKmerFreq = std::stoi(cmd.optionValue(mk_arg));

  if(in_args.runType == 'c') {
    if(cmd.foundOption(wt_arg)) {
      in_args.dbgNodeWtType = cmd.optionValue(wt_arg)[0];
      if(in_args.dbgNodeWtType != 'c' && in_args.dbgNodeWtType != 's'){
        if(comm.rank() == 0)
            std::cout << in_args.dbgNodeWtType << " is invalid option of " << wt_arg
                      << ". Option should be one of 'c' or 's'. " << std::endl;
        return 1;
      }
    }
  }

  LOG_IF(!comm.rank(), INFO) << "--------------------------------------" ;
  LOG_IF(!comm.rank(), INFO) << "Run Type       : " << in_args.runType;
  LOG_IF(!comm.rank(), INFO) << "Input File     : " << in_args.ipFileName;
  LOG_IF(!comm.rank(), INFO) << "Adj File Prefix: " << in_args.ajFilePrefix;
  LOG_IF(!comm.rank(), INFO) << "Filtered Kmers File Prefix : " << in_args.filteredKmersPrefix;
  LOG_IF(!comm.rank(), INFO) << "Color Kmer File Prefix     : " << in_args.colorKmerMapPrefix;
  LOG_IF(!comm.rank(), INFO) << "Color Vertex File Prefix   : " << in_args.colorVertexListPrefix;
  LOG_IF(!comm.rank(), INFO) << "Partition File : " << in_args.ptFileName;
  LOG_IF(!comm.rank(), INFO) << "RP File Prefix : " << in_args.rpFilePrefix;
  LOG_IF(!comm.rank(), INFO) << "Min kmer Freq  : " << in_args.minKmerFreq;
  LOG_IF(!comm.rank(), INFO) << "dbg N Wt Type  : " << in_args.dbgNodeWtType;
  LOG_IF(!comm.rank(), INFO) << "--------------------------------------" ;

  return 0;
}

int main(int argc, char* argv[])
{

  // Initialize the MPI library:
  MPI_Init(&argc, &argv);

  //Initialize the communicator
  mxx::comm comm;

  //Print mpi rank distribution
  mxx::print_node_distribution();

  // COMMAND LINE ARGUMENTS
  LOG_IF(!comm.rank(), INFO) << "Start PARTITION";

  //Parse command line arguments
  InputArgs cargs;

  if(parse_args(argc, argv, comm, cargs)) return 1;


  dbgp::init(); // initialize static variables
  if(cargs.runType == 'c'){
      dbgp::compactDBG(comm, cargs.ipFileName, cargs.minKmerFreq, cargs.dbgNodeWtType,
                       cargs.ajFilePrefix, cargs.filteredKmersPrefix, cargs.colorKmerMapPrefix, cargs.colorVertexListPrefix);
  } else if(cargs.runType == 'p'){
      dbgp::partitionReads(comm, cargs.ipFileName, cargs.ptFileName,
                           cargs.rpFilePrefix, cargs.colorKmerMapPrefix, cargs.colorVertexListPrefix, cargs.filteredKmersPrefix);
  } else {
      return 1;
  }
  MPI_Finalize();
  return(0);
}


