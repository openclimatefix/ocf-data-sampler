from ocf_data_sampler.datasets.pvnet import PVNetDataset


def test_pvnet(config_filename):
    # ------------------ basic usage ---------------------

    # Create dataset object
    dataset = PVNetDataset(config_filename)

    # Print number of samples
    print(f"Found {len(dataset.valid_t0_times)} possible samples")

    idx = 0
    t_index, loc_index = dataset.index_pairs[idx]

    location = dataset.locations[loc_index]
    t0 = dataset.valid_t0_times[t_index]

    # Print coords
    print(t0)
    print(location)

    # Generate sample - no printing since its BIG
    _ = dataset[idx]
