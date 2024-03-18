from pytorch_metric_learning import miners


def get_miner(params):
    miner = None
    # @Triplets
    # Outputs a triplet
    if params['miner'] == 'bhard':
        miner = miners.BatchHardMiner()

    # return triplet with angle greater than input angle
    elif params['miner'] == 'angular':
        miner = miners.AngularMiner(angle=20)

    # margin --> diff between anchor_+ & anchor_- distances
    # type_of_triplets --> type triples that violate margin: all, hard, semihard, easy
    elif params['miner'] == 'triplet_margin':
        miners.PairMarginMiner(margin=0.2, type_of_triplets='all')

    # @Pairs
    # Returns positive and negative pairs according 
    # to the specified pos_strategy and neg_strategy
    elif params['miner'] == 'beasyhard':
        miner = miners.BatchEasyHardMiner(pos_strategy=miners.BatchEasyHardMiner.EASY, neg_strategy=miners.BatchEasyHardMiner.EASY)

    # cutoff --> Pairwise distances fixed to this value if below
    # nonzero_loss_cutoff --> Pair with greater distances discarded
    elif params['miner'] == 'distance_weighted':
        miner = miners.DistanceWeightedMiner(cutoff=0.5, nonzero_loss_cutoff=1.4)

    # Chooses pairs between the positive margin and negative margin
    elif params['miner'] == 'pair_margin':
        miner = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8)

    # filter_percentage --> percentage of pairs that will be returned. (hardest n)
    elif params['miner'] == 'HDC':
        miner = miners.HDCMiner(filter_percentage=0.5)

    # epsilon --> Negative pairs chosen if similarity > the hardest positive pair - epsilon
    #             Positive pairs chosen if similarity < the hardest negative pair + epsilon
    elif params['miner'] == 'multi_similarity':
        miner = miners.MultySimilarityMiner(epsilon=0.1)

    else:
        raise ValueError(f"{params['miner']} is not a viable miner option")

    return miner
