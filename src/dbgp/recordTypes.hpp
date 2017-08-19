#ifndef DBGP_RECORD_TYPES_H
#define DBGP_RECORD_TYPES_H

#include <array>

//
#include "coloring/labelProp_utils.hpp"
#include "coloring/labelProp.hpp"

//External includes
#include "index/kmer_hash.hpp"
#include "index/kmer_index.hpp"
#include "common/kmer_transform.hpp"
#include "iterators/transform_iterator.hpp"
#include "containers/densehash_map.hpp"
#include "containers/distributed_densehash_map.hpp"
#include "debruijn/de_bruijn_node_trait.hpp"
#include "debruijn/de_bruijn_construct_engine.hpp"
#include "debruijn/de_bruijn_nodes_distributed.hpp"

namespace dbgp {
    using vertexIdType = uint64_t;
    using weightType = uint32_t;

    // ccl types
    using cclType = conn::coloring::ccl<vertexIdType,
                                        conn::coloring::lever::ON>;

    using cclNodeType = cclType::tupleType;
    using colorType = cclType::pIdtype;

    // edge types
    struct edgeRecord {
        vertexIdType vertexU;
        vertexIdType vertexV;
        weightType edgeWeight;

        edgeRecord(vertexIdType ux,vertexIdType vx, weightType ew) {
            set(ux, vx, ew);
        }
        edgeRecord(){} // empty constructor for

        void set(vertexIdType ux,vertexIdType vx, weightType ew) {
            vertexU = ux; vertexV = vx; edgeWeight = ew;
        }
        bool operator==(const edgeRecord& other){
            return (vertexU == other.vertexU) &&
                (vertexV == other.vertexV);
        }
    };

    using edgeRecordType = edgeRecord;

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

    // partition record type
    struct prtRecord {
        vertexIdType vertexU;
        std::array<vertexIdType, 8> vertexV;
        std::array<weightType, 8> edgeWeight;
        weightType vUWeight;

        prtRecord(vertexIdType ux, std::vector<vertexIdType>& vx, std::vector<weightType>& ew, weightType nw) {
            set(ux, vx, ew, nw);
        }
        prtRecord(){} // empty constructor for

        void set(vertexIdType ux, std::vector<vertexIdType>& vx, std::vector<weightType>& ew, weightType nw) {
            vertexU = ux;
            for(auto &zx : vertexV) {
                zx = std::numeric_limits<dbgp::vertexIdType>::max();
            }
            auto aitr = vertexV.begin();            
            for(auto &zx : vx) {
                *aitr = zx;
                aitr++;
            }
            for(auto &zx : edgeWeight) {
                zx = std::numeric_limits<dbgp::weightType>::max();
            }
            auto bitr = edgeWeight.begin();
            for(auto &zx : ew) {
                *bitr = zx;
                bitr++;
            }
            vUWeight = nw;
        }
    };

    using prtRecordType = prtRecord;

    // node tuple type
    struct vertexRecord{
        vertexIdType vertexID;
        colorType vertexColor;
        weightType vertexWeight;

        vertexRecord(vertexIdType vid, colorType clr, weightType ew){
            set(vid, clr, ew);
        }
        void set(vertexIdType vid, colorType clr, weightType ew){
            vertexID = vid; vertexColor = clr; vertexWeight = ew;
        }
        vertexRecord(){}
    };

    using vertexRecordType = vertexRecord;

    struct xferRecord{
        colorType xferColor;
        vertexIdType xferUID;
        vertexIdType xferVID;
        weightType xferWeight;

        xferRecord(){}
        xferRecord(colorType xc, vertexIdType uid,
                   vertexIdType vid, weightType wt){
            set(xc, uid, vid, wt);
        }

        void set(colorType xc, vertexIdType uid,
                 vertexIdType vid, weightType wt){
            xferColor = xc;
            xferUID = uid;
            xferVID = vid;
            xferWeight = wt;
        }
    };
    using xferRecordType = xferRecord;

    using SeqIdType = bliss::common::ShortSequenceKmerId;
    using PosValType = SeqIdType;
    using PartitionValueType = uint16_t;

/*
    struct seqPartitionRecord{
        SeqIdType seqId;
        PartitionValueType partitionId;

        seqPartitionRecord(){}

        seqPartitionRecord(SeqIdType sq, PartitionValueType pid){
            set(sq, pid);
        }
        void set(SeqIdType sq, PartitionValueType pid){
            seqId = sq;
            partitionId = pid;
        }
    };
    using seqPartitionRecordType = seqPartitionRecord;
*/

    struct colorMapRecord{
        vertexIdType vertexSrc;
        vertexIdType vertexDest;
        PartitionValueType partitionId;

        colorMapRecord(){}

        colorMapRecord(vertexIdType vs, vertexIdType vd, PartitionValueType px){
            set(vs, vd, px);
        }
        void set(vertexIdType vs, vertexIdType vd, PartitionValueType px){
            vertexSrc = vs;
            vertexDest = vd;
            partitionId = px;
        }
    };
    using colorMapRecordType = colorMapRecord;


    //Kmer size set to 31, and alphabets set to 4 nucleotides
    using Alphabet = bliss::common::DNA;
    using KmerType = bliss::common::Kmer<31, Alphabet, uint64_t>;
    template <typename KM>
    using CntIndexDistHash = bliss::kmer::hash::farm<KM, true>;
    template <typename KM>
    using CntIndexStoreHash = bliss::kmer::hash::farm<KM, false>;
    template <typename Key>
    using CntIndexMapParams = bliss::index::kmer::BimoleculeHashMapParams<Key, CntIndexDistHash, CntIndexStoreHash>;
    //BLISS internal data structure for storing de bruijn graph
    template <typename EdgeEnc>
    using NodeMapType = typename
        bliss::de_bruijn::de_bruijn_nodes_distributed<
        dbgp::KmerType,
        bliss::de_bruijn::node::edge_counts<EdgeEnc, uint32_t>,
        dbgp::CntIndexMapParams>;


    //Parser type, depends on the sequence file format
    //We restrict the usage to FASTA format
    template <typename baseIter>
    using SeqParser = typename bliss::io::FASTAParser<baseIter>;

    // POS Index type
    template <typename KM>
    using PosIndexDistHash = bliss::kmer::hash::farm<KM, true>;
    template <typename KM>
    using PosIndexStoreHash = bliss::kmer::hash::farm<KM, false>;
    template <typename Key>
    using PosIndexMapParams = bliss::index::kmer::CanonicalHashMapParams<Key, PosIndexDistHash, PosIndexStoreHash>;

    using PosIndexMapType = ::dsc::unordered_multimap<KmerType, PosValType, PosIndexMapParams>;
    using PosIndexType = bliss::index::kmer::PositionIndex<PosIndexMapType>;

//    // Map type for parition
//    template <typename Key>
//    using PartitionMapParams = ::bliss::index::kmer::CanonicalHashMapParams<Key, PosIndexDistHash, PosIndexStoreHash>;
//    using PartitionMapType = ::dsc::unordered_map<KmerType,
//                                                  PartitionValueType,
//                                                  PartitionMapParams>;
    
    template<typename KVT>
    struct farm_hash_store{
        farm_hash_store(){}
        farm_hash_store(const unsigned int){}
        inline uint64_t operator()(const KVT& kval) const{
            return ::util::Hash(reinterpret_cast<const char*>(&kval) + 3, sizeof(KVT) - 3);
        }
    };

    template<typename KVT>
    struct farm_hash_dist{
        farm_hash_dist(){}
        farm_hash_dist(const unsigned int){}
        inline uint64_t operator()(const KVT& kval) const{
            return ::util::Hash(reinterpret_cast<const char*>(&kval), 3);
        }
    };

    template<typename Key>
    using PartitionMapParams = ::dsc::HashMapParams<
        Key,
        ::bliss::transform::identity,  // precanonalizer
        ::bliss::transform::identity,  // could be iden, xor, lex_less
        farm_hash_dist,
        ::std::equal_to,
        ::bliss::transform::identity,  // only one that makes sense given InputTransform
        farm_hash_store,
        ::std::equal_to
        >;

    using PartitionMapType = ::dsc::densehash_map<dbgp::vertexIdType,
                                                   PartitionValueType,
                                                   PartitionMapParams>;

/*
    template<typename Key>
    using LocalPartMapParams = ::fsc::HashMapParams<
        Key,
        ::bliss::transform::identity,  // precanonalizer
        ::bliss::transform::identity,  //ould be iden, xor, lex_less
        farm_hash,
        ::std::equal_to,
        ::bliss::transform::identity,  // only one that makes sense given InputTransform
        farm_hash,
        ::std::equal_to
        >;

    using LocalPartMapType = ::fsc::densehash_map<dbgp::vertexIdType,
                                                  PartitionValueType,
                                                  PartitionMapParams>;
*/
    using LocalPartMapType = ::fsc::densehash_map<dbgp::vertexIdType,
                                                  dbgp::PartitionValueType>;
    using PosIdxTupleType = typename dbgp::PosIndexType::TupleType;

/*
    template<typename KVT>
    struct farm_hash{
        farm_hash(){}
        farm_hash(const unsigned int){}
        inline uint64_t operator()(const KVT& kval) const{
            return ::util::Hash(reinterpret_cast<const char*>(&kval), 1);
        }
    };

    template<typename Key>
    using KmerVertMapParams = ::dsc::HashMapParams<
        Key,
        ::bliss::transform::identity,  // precanonalizer
        ::bliss::transform::identity,  //ould be iden, xor, lex_less
        farm_hash,
        ::std::equal_to,
        ::bliss::transform::identity,  // only one that makes sense given InputTransform
        farm_hash,
        ::std::equal_to
        >;

  using KmerVertIDMapType = ::dsc::unordered_map<dbgp::vertexIdType,
                                                 uint64_t,
                                                 KmerVertMapParams>;
*/


}

MXX_CUSTOM_STRUCT(dbgp::edgeRecordType, vertexU, vertexV, edgeWeight);
MXX_CUSTOM_STRUCT(dbgp::adjRecordType, vertexU, vertexV, edgeWeight, vUWeight);
MXX_CUSTOM_STRUCT(dbgp::prtRecordType, vertexU, vertexV, edgeWeight, vUWeight);
MXX_CUSTOM_STRUCT(dbgp::vertexRecordType, vertexID, vertexColor, vertexWeight);
MXX_CUSTOM_STRUCT(dbgp::xferRecordType, xferColor, xferUID, xferVID, xferWeight);
//MXX_CUSTOM_STRUCT(dbgp::seqPartitionRecordType, seqId, partitionId);
MXX_CUSTOM_STRUCT(dbgp::colorMapRecordType, vertexSrc, vertexDest, partitionId);

#endif
