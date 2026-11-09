
import torch

from train import (
    optimize_model,
    memory,
    policy_net,
    BATCH_SIZE
)


def fill_memory(all_terminal=False):

    memory.memory.clear()

    for i in range(BATCH_SIZE):

        state = torch.rand(4, 1, 84, 84)
        action = 0
        reward = 1.0

        if all_terminal or i % 2 == 0:
            next_state = None
            done = True
        else:
            next_state = torch.rand(4, 1, 84, 84)
            done = False

        memory.push(
            state,
            action,
            next_state,
            reward,
            done
        )


def test_mixed_transitions():

    fill_memory(all_terminal=False)

    loss = optimize_model()

    assert loss is not None
    assert torch.isfinite(torch.tensor(loss))

    print("Mixed terminal/non-terminal test passed!")


def test_all_terminal_transitions():

    fill_memory(all_terminal=True)

    loss = optimize_model()

    assert loss is not None
    assert torch.isfinite(torch.tensor(loss))

    print("All-terminal transitions test passed!")


if __name__ == "__main__":

    test_mixed_transitions()
    test_all_terminal_transitions()

    print("All optimization tests passed!")
