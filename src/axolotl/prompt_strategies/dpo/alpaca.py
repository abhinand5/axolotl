"""
DPO strategies for alpaca
"""

def argilla(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"{sample['system']}\n\n"
                f"### Instruction:\n{sample['instruction']}\n\n### Response:\n"
            )
        else:
            sample["prompt"] = (
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['instruction']}\n\n### Response:\n"
            )
        
        sample["chosen"] = f"{sample['chosen_response']}"
        sample["rejected"] = f"{sample['rejected_response']}"

    return transform_fn


def argilla_chat(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for argilla/dpo-mix-7k conversations
    """

    def transform_fn(sample):
        sample[
            "prompt"
        ] = sample["prompt"] = (
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['chosen'][0]['content']}\n\n### Response:\n"
            )
        sample["chosen"] = f"{sample['chosen'][1]['content']}"
        sample["rejected"] = f"{sample['rejected'][1]['content']}"
        return sample

    return transform_fn


def intel(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    For Intel Orca DPO Pairs
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"{sample['system']}\n\n"
                f"### Instruction:\n{sample['question']}\n\n### Response:\n"
            )
        else:
            sample["prompt"] = (
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['question']}\n\n### Response:\n"
            )

        sample["chosen"] = f"{sample['chosen']}"
        sample["rejected"] = f"{sample['rejected']}"
        return sample

    return transform_fn


def icr(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    chatml transforms for datasets with system, input, chosen, rejected
    ex. https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"{sample['system']}\n\n"
                f"### Instruction:\n{sample['input']}\n\n### Response:\n"
            )
        else:
            sample["prompt"] = (
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['input']}\n\n### Response:\n"
            )

        sample["chosen"] = f"{sample['chosen']}"
        sample["rejected"] = f"{sample['rejected']}"
        return sample

    return transform_fn
