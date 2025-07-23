import torch

def upsample_path(path: torch.Tensor, target_len: int = 50) -> torch.Tensor:
    new_path = [path[0]]
    total_insert = target_len - len(path)
    insert_counts = [0] * (len(path) - 1)

    for i in range(total_insert):
        insert_counts[i % (len(path) - 1)] += 1
    print(f"Insert counts: {insert_counts}")

    for i in range(1, len(path)):
        prev = path[i - 1]
        curr = path[i]
        num_insert = insert_counts[i - 1]
        for j in range(1, num_insert + 1):
            alpha = j / (num_insert + 1)
            interp = (1 - alpha) * prev + alpha * curr
            new_path.append(interp)
        new_path.append(curr)
    return torch.stack(new_path)