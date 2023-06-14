def test_create_shards():
    from flexfl.data.loader import create_shards
    from torchvision.datasets import MNIST

    target_iidness = 1
    sample_size_iidness = 0.5

    dataset = MNIST(root="~/data", train=True, download=True)
    shards = create_shards(dataset, num_shards=5, target_iidness=target_iidness, sample_size_iidness=sample_size_iidness)

    # Get target distribution per shard. Each shard is a list of tuples (data, target).
    # As a dictionary whose keys are the shard indices and values are the target distributions.
    target_distributions = {}
    for i, shard in enumerate(shards):
        targets = [target for _, target in shard]
        target_distributions[i] = {target: targets.count(target) / len(targets) for target in set(targets)}

    # Create dataframe with target distributions per shard and plot class distribution per shard
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(target_distributions)
    df.plot.bar()
    plt.title(f"Target distribution per shard. Non-IIDness: {target_iidness=} {sample_size_iidness=}")
    plt.show()


    pass
