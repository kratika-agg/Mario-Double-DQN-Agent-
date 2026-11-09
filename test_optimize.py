import torch

from train import (
    optimize_model,
    memory,
    policy_net,
    BATCH_SIZE
)


def test_optimize_model():

    # Store dummy experiences in replay memory
    for _ in range(BATCH_SIZE):

        state = torch.rand(4, 1, 84, 84)
        action = 0
        next_state = torch.rand(4, 1, 84, 84)
        reward = 1.0
        done = False

        memory.push(
            state,
            action,
            next_state,
            reward,
            done
        )

    # Save network weights before optimization
    old_weights = {
        name: param.clone()
        for name, param in policy_net.named_parameters()
    }

    # Run optimization
    loss = optimize_model()

    # Check whether loss was calculated
    print("Loss:", loss)
    assert loss is not None
    assert torch.isfinite(torch.tensor(loss))

    # Check whether network weights changed
    weights_changed = any(
        not torch.equal(old_weights[name], param)
        for name, param in policy_net.named_parameters()
    )

    assert weights_changed

    print("Optimization test passed!")
    print("Policy network weights updated successfully!")


if __name__ == "__main__":
    test_optimize_model()