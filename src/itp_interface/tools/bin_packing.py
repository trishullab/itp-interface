import random
import copy

def best_fit_packing(bin_sizes, item_sizes):
    """
    Best fit algorithm for bin packing problem with no empty bins, modified to ensure
    no non-zero capacity bin is left empty if possible. It minimizes the absolute difference
    between the bin capacity and the packed items, without creating new bins.
    :param bin_sizes: list of bin sizes (each bin's maximum capacity)
    :param item_sizes: list of item sizes (sizes of items to be packed)
    :returns: list of lists containing the items packed in each bin
    """
    sorted_item_sizes_with_index = sorted(enumerate(item_sizes), key=lambda x: x[1], reverse=True)
    bins = [[] for _ in range(len(bin_sizes))]
    bin_remaining_capacity = copy.deepcopy(bin_sizes)
    # Reduce the bin capacity a bit to ensure no empty bins
    for i, size in enumerate(bin_remaining_capacity):
        if size > 0:
            bin_remaining_capacity[i] -= 1
    bin_remaining_capacity = sorted(enumerate(bin_remaining_capacity), key=lambda x: x[1], reverse=True)
    items_packed = set()
    for item_index, item_size in sorted_item_sizes_with_index:
        best_fit_index = -1
        best_fit_remaining_capacity = float('inf')
        sorted_bin_idx = -1
        for _idx, (i, remaining_capacity) in enumerate(bin_remaining_capacity):
            # don't consider bins with zero capacity
            if bin_sizes[i] == 0:
                continue
            if remaining_capacity >= item_size and remaining_capacity - item_size < best_fit_remaining_capacity:
                best_fit_index = i
                sorted_bin_idx = _idx
                best_fit_remaining_capacity = remaining_capacity - item_size
        if best_fit_index != -1:
            bins[best_fit_index].append(item_index)
            bin_remaining_capacity[sorted_bin_idx] = (best_fit_index, best_fit_remaining_capacity)
            items_packed.add(item_index)
            bin_remaining_capacity = sorted(bin_remaining_capacity, key=lambda x: x[1], reverse=True)
    # Randomly pack the remaining items in the bins
    while len(items_packed) < len(item_sizes):
        remaining_items = [i for i in range(len(item_sizes)) if i not in items_packed]
        random.shuffle(remaining_items)
        for idx, item_index in enumerate(remaining_items):
            bin_idx = idx % len(bins)
            if bin_sizes[bin_idx] == 0:
                continue
            bins[idx % len(bins)].append(item_index)
            items_packed.add(item_index)
    return bins

if __name__ == '__main__':
    # Example usage
    item_sizes = [0, 0, 0, 0, 0, 2, 5, 1, 0, 3, 2, 0, 1, 2, 23, 0, 0, 0, 25, 1, 0, 0, 0, 4, 0, 0, 10, 0, 1, 6, 4, 1, 3, 0, 0, 17, 0, 8, 0, 17, 0, 0, 0, 7, 0, 0, 4, 0, 0, 4, 0, 0, 0, 3, 1, 0, 0, 9, 0, 0, 15, 7, 9, 2, 1, 0, 10, 15, 2, 4, 0, 0, 0, 1, 0, 0, 9, 19, 3, 0, 0, 0, 0, 0, 20, 0, 6, 23, 0, 5, 11, 0, 0, 0, 13, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 6, 3, 0, 12, 1, 3, 2, 14, 6, 16, 4, 0, 0, 0, 1, 2, 12, 0, 31, 13, 0, 15, 16, 0, 0, 4, 1, 0, 0, 0, 1, 3, 2, 0, 7, 0, 0, 0, 0, 0, 1, 0, 0, 7, 1, 0, 0, 0, 0, 4, 0, 0, 2, 0, 0, 2, 5, 0, 0, 2, 0, 3, 1, 0, 0, 0, 8, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 2, 38, 1, 0, 1, 1, 1, 0, 2, 28, 0, 0, 0, 1, 3, 0, 13, 1, 0, 1, 5, 1, 0, 0, 0, 0, 1, 6, 0, 10, 0, 2, 1, 1, 1, 4, 0, 7, 1, 2, 14, 0, 0, 2, 0]
    print(f"Total items: {len(item_sizes)}")
    sum_item_sizes = sum(item_sizes)
    bin_sizes = [int(0.925 * sum_item_sizes), int(0.0375 * sum_item_sizes), int(0.0375 * sum_item_sizes)]
    bins = best_fit_packing(bin_sizes, item_sizes)
    for i, b in enumerate(bins):
        items = [item_sizes[j] for j in b]
        print(f'Bin {i + 1}: {items}, Total size: {sum(items)}, Remaining capacity: {bin_sizes[i] - sum(items)}, Bin size: {bin_sizes[i]}')
    print(f"All items packed: {sum([len(b) for b in bins]) == len(item_sizes)}")