from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",  # Re-enabling Flash Attention for GPU
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# Cluedo System Prompt (Example - adjust as needed)
cluedo_system_prompt = """\"\"\"You are an AI assistant playing the game of Cluedo. Your goal is to deduce the murderer, weapon, and room by making suggestions, evaluating challenges, and updating your memory based on game events. Respond accurately and strategically based on the provided information. Respond ONLY with the requested JSON format.\"\"\"
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task_data: dict, # Now expects the dictionary loaded from JSONL
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:

    model.eval()
    
    # Extract data from this task, handling the format from the custom collate function
    # With batch_size=1 and custom_collate, we get {key: [value]} instead of just {key: value}
    # Get the first (and only) item from each list if it exists
    if isinstance(task_data, dict):
        interaction_type_val = task_data.get("interaction_type", ["unknown"])
        prompt_text_val = task_data.get("prompt", [None])
        ground_truth_val = task_data.get("ground_truth_deductions", [None])
        
        # Check if we have lists and extract the first item if they're not empty
        interaction_type = interaction_type_val[0] if isinstance(interaction_type_val, list) and interaction_type_val else "unknown"
        prompt_text = prompt_text_val[0] if isinstance(prompt_text_val, list) and prompt_text_val else None
        ground_truth_deductions = ground_truth_val[0] if isinstance(ground_truth_val, list) and ground_truth_val else None
    else:
        # If somehow task_data is not a dict, provide safe defaults
        print(f"Warning: task_data is not a dictionary: {type(task_data)}")
        interaction_type = "unknown"
        prompt_text = None
        ground_truth_deductions = None

    if not prompt_text:
        print("Warning: Skipping rollout due to missing prompt text.")
        # Return empty tensors or handle appropriately
        dummy_tensor = torch.empty((num_rollouts, 0), dtype=torch.long, device=model.device)
        dummy_rewards = torch.zeros((num_rollouts, 1), dtype=torch.float, device=model.device)
        return dummy_tensor, dummy_rewards, dummy_tensor.bool(), []

    # 1. format prompt with JSON formatting instructions
    if interaction_type == "memory_update":
        if not prompt_text.endswith("Respond ONLY with a JSON object."):
            # Add JSON formatting instruction to the end of the prompt
            chat_prompt = prompt_text + "\n\nIMPORTANT: Respond ONLY with a JSON object containing newly_deduced_held_cards as an array. Example format: {\"newly_deduced_held_cards\": [\"Card1\", \"Card2\"]}"
        else:
            chat_prompt = prompt_text
    else:
        # For other interaction types, use the prompt as-is
        chat_prompt = prompt_text

    # Efficient batched tokenization for GPU
    model_inputs = tokenizer(
        [chat_prompt] * num_rollouts, # Repeat prompt for batch generation
        return_tensors="pt",
        padding=True,
        padding_side="left", # Important for generation
        truncation=True,
        max_length=max_length - 100, # Ensure space for generation
        return_attention_mask=True,
    ).to(model.device)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    # 2. sample completions with efficient generation config
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=100, # Limit generated output length
        pad_token_id=pad_token_id,
        do_stream=False, # Disable streaming for batch efficiency
    )
    
    # Use efficient generation
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Use mixed precision on GPU
        sequence_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
    
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1]:], skip_special_tokens=True
    )

    # Create action mask (masking prompt tokens, keeping completion tokens)
    action_mask = torch.ones_like(sequence_ids, dtype=torch.bool)
    action_mask[:, :input_ids.shape[1]] = False # Mask prompt tokens
    # Mask padding tokens in the completion part
    completion_start_index = input_ids.shape[1]
    for i in range(sequence_ids.shape[0]):
        # Find the first pad token *after* the prompt
        pads = (sequence_ids[i, completion_start_index:] == pad_token_id).nonzero()
        if len(pads) > 0:
            first_pad_index = pads[0].item() + completion_start_index
            action_mask[i, first_pad_index:] = False

    action_mask = action_mask[:, 1:] # Align with log_probs (logits[:, :-1])


    # 3. determine rewards based on interaction type
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        reward = 0.0
        if interaction_type == "memory_update":
            # Use ground truth deductions for reward calculation
            reward = calculate_memory_update_reward(completion, ground_truth_deductions)
        else:
            # Use basic reward (e.g., JSON validity) for other types
            reward = calculate_basic_reward(completion)

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    try:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
        output = model.forward(
            input_ids=sequence_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = output["logits"]
        
        # Debug shapes to help identify potential mismatches
        if logits.shape[1] - 1 != sequence_ids.shape[1] - 1:
            print(f"Warning: Shape mismatch! logits[:, :-1]: {logits[:, :-1].shape}, sequence_ids[:, 1:]: {sequence_ids[:, 1:].shape}")
        
        log_probs = sequence_log_probs_from_logits(
            logits=logits[:, :-1].to(torch.float32),
            output_ids=sequence_ids[:, 1:],
        )
        return log_probs
    except RuntimeError as e:
        print(f"Error in sequences_log_probs: {e}")
        print(f"Shapes: sequence_ids={sequence_ids.shape}, attention_mask={attention_mask.shape}, logits={logits.shape if 'logits' in locals() else 'N/A'}")
        raise


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


# Helper function to calculate reward for memory updates
def calculate_memory_update_reward(completion_text: str, ground_truth_deductions: list[str]) -> float:
    try:
        # First attempt: parse as-is
        completion_json = json.loads(completion_text)
        # Adjust key based on expected LLM output format for memory update
        predicted_deductions = set(completion_json.get("newly_deduced_held_cards", []))
        truth_set = set(ground_truth_deductions if ground_truth_deductions else [])

        if not predicted_deductions and not truth_set:
            return 1.0 # Correctly deduced nothing new when nothing was expected

        intersection = len(predicted_deductions.intersection(truth_set))
        # Use precision: reward based on how many of the *predicted* deductions were correct
        reward = intersection / len(predicted_deductions) if len(predicted_deductions) > 0 else 0.0
        # Small bonus if prediction is non-empty and fully correct
        if len(predicted_deductions) > 0 and intersection == len(truth_set) and intersection == len(predicted_deductions):
             reward = 1.0
        # Penalize hallucinating deductions when ground truth is empty? Maybe later.

        return reward

    except json.JSONDecodeError:
        # Instead of failing, treat free text as a summary and give a small fixed reward
        print(f"Generating JSON wrapper for text response: {completion_text[:100]}...")
        # Give a small fixed reward (could change this based on token match heuristics)
        return 0.02 # Small but non-zero to encourage formatting improvement
    except Exception as e:
        print(f"Warning: Error calculating reward: {e}")
        return 0.0


# Helper function for basic reward (e.g., suggestion/accusation format check)
def calculate_basic_reward(completion_text: str) -> float:
     try:
        json.loads(completion_text)
        # Basic reward for outputting valid JSON
        # Could be extended to check structure against chosen_response later
        return 0.1
     except json.JSONDecodeError:
         return 0.0 # Penalize invalid JSON


def custom_collate(batch):
    """Custom collate function that can handle None values in the dataset"""
    if not batch:
        return {}
    
    # If we get a list of dictionaries, convert to a dictionary of lists
    if isinstance(batch[0], dict):
        result = {}
        # For each key in the first dictionary
        for key in batch[0].keys():
            # Collect all values for this key across all dictionaries
            values = [d.get(key) for d in batch]
            # Store in the result dictionary
            result[key] = values
        return result
    
    # For non-dictionary batches, just return as is
    return batch


def main():
    seed = 42
    wandb_project = "cluedo_grpo"  # Set WandB project name
    device_index = 0
    # Larger model is better with GPU available
    model_name = "meta-llama/Llama-3.2-7B-Instruct"  # Upgraded to 7B model
    checkpoint_path = Path("./output_cluedo")  # Separate output dir
    checkpoint_interval = 20
    train_batch_size = 32  # Increased for GPU (adjust based on GPU memory)
    lr = 5e-6  # Starting learning rate
    kl_weight = 0.01  # D_KL coefficient in loss
    clip_eps = 0.2  # PPO clipping epsilon

    group_size = 24  # Doubled number of rollouts per prompt in a GRPO step
    rollouts_per_step = 64  # Doubled number of prompts processed per training step
    epochs_per_step = 1  # Number of training epochs on collected rollouts
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024  # Max sequence length (prompt + generation)
    top_p = 0.9  # Sampling params for generation
    temperature = 0.7  # Sampling params for generation

    # Prioritize CUDA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda", device_index)
        print(f"Using CUDA device: {device_index}")
        print(f"GPU: {torch.cuda.get_device_name(device_index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device_index).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU. Training will be much slower.")
    
    cpu_device = torch.device("cpu")
    init_rng(seed)

    # Load models
    print(f"Loading models: {model_name}")
    reference_model, _ = load_model(model_name, device_map="auto")  # Use auto device mapping
    model, tokenizer = load_model(model_name, device_map="auto")
    print("Models loaded successfully")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Ensure pad token is set for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.pad_token_id

    # Load Cluedo data
    print("Loading Cluedo interaction data...")
    # No predicate needed for now, load all data
    # Correct the path relative to the workspace root where the script is run
    prompts_data = read_jsonl("tiny-grpo/data/cluedo_interactions.jsonl")
    prompts_list = list(prompts_data) # Load all into memory for DataLoader
    print(f"Loaded {len(prompts_list)} Cluedo interaction examples.")

    if not prompts_list:
        print("Error: No data loaded. Check data/cluedo_interactions.jsonl path and format.")
        return

    # Note: DataLoader will yield dictionaries from prompts_list
    prompt_loader = DataLoader(
        prompts_list,
        batch_size=1, # Process one prompt data dict at a time for rollout
        shuffle=True,
        drop_last=True, # Avoid partial batches if rollouts_per_step doesn't divide dataset size
        collate_fn=custom_collate, # Use our custom collate function
    )
    prompt_iterator = iter(prompt_loader) # Make it an iterator

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is not None:
        wandb.init(project=wandb_project)

    # Create checkpoint dir
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    # Adjust total steps based on dataset size and how many times you want to iterate
    total_training_steps = 500 # Example: Set a fixed number of steps

    for step in range(total_training_steps):
        print(f"--- Step {step + 1} / {total_training_steps} ---")
        experiences = []
        model.eval() # Set model to eval for rollouts

        # --- Rollout Phase ---
        # Collect rollouts for a number of prompts determined by rollouts_per_step
        prompts_processed_this_step = 0
        rollout_data_collected = []

        while prompts_processed_this_step < rollouts_per_step:
            try:
                # Get next task data dictionary from the loader
                current_task_data = next(prompt_iterator)
                
                # Log the type we're processing
                interaction_type = "unknown"
                if isinstance(current_task_data, dict) and "interaction_type" in current_task_data:
                    if isinstance(current_task_data["interaction_type"], list):
                        if current_task_data["interaction_type"]:
                            interaction_type = current_task_data["interaction_type"][0]
                    else:
                        interaction_type = current_task_data["interaction_type"]
                        
                print(f"Rolling out for prompt type: {interaction_type}...")
                
            except StopIteration:
                # Reset iterator if dataset is exhausted
                print("Resetting prompt data loader.")
                prompt_iterator = iter(prompt_loader)
                try:
                    current_task_data = next(prompt_iterator)
                except StopIteration:
                    print("Error: Dataset is empty or DataLoader is not working correctly.")
                    break
                
                # Get the interaction type after resetting
                interaction_type = "unknown"
                if isinstance(current_task_data, dict) and "interaction_type" in current_task_data:
                    if isinstance(current_task_data["interaction_type"], list):
                        if current_task_data["interaction_type"]:
                            interaction_type = current_task_data["interaction_type"][0]
                    else:
                        interaction_type = current_task_data["interaction_type"]
                print(f"Rolling out for prompt type (after reset): {interaction_type}...")
            
            # Perform rollouts for the current task
            try:
                seq_ids, returns, action_mask, completions = rollout(
                    model=reference_model, # Use reference model for rollouts
                    tokenizer=tokenizer,
                    task_data=current_task_data,
                    num_rollouts=group_size, # Generate 'group_size' completions per prompt
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                if seq_ids.numel() == 0: # Handle case where rollout was skipped
                    print("Skipping empty rollout result.")
                    continue

                # Calculate advantages for this group of rollouts
                advantages = group_advantages(returns)

                # Get log probs from reference model for KL divergence penalty
                with torch.no_grad():
                    ref_log_probs = sequences_log_probs(
                        reference_model,
                        sequence_ids=seq_ids,
                        attention_mask=(seq_ids != pad_token_id),
                    )
                    ref_log_probs = ref_log_probs.detach()

                # Store data needed for training step
                rollout_data_collected.append(
                    {
                        "sequence_ids": seq_ids.to(cpu_device),
                        "action_mask": action_mask.to(cpu_device),
                        "returns": returns.to(cpu_device),
                        "advantages": advantages.to(cpu_device),
                        "ref_log_probs": ref_log_probs.to(cpu_device),
                    }
                )
                prompts_processed_this_step += 1
                print(f"Rollouts collected for prompt {prompts_processed_this_step}/{rollouts_per_step}")
            except Exception as e:
                print(f"Error during rollout: {e}")
                continue  # Skip to the next prompt


        # --- Training Phase ---
        model.train() # Set model to train
        
        # Skip training if no valid rollout data was collected
        if not rollout_data_collected:
            print("No valid rollout data collected this step. Skipping training.")
            continue
            
        # Combine collected data into batches
        combined_data = join_experience_batch(rollout_data_collected)

        # Create DataLoader for training steps
        experience_loader = DataLoader(
            Experience(**combined_data), # Use the custom Experience dataset/iterable
            batch_size=train_batch_size, # Actual training batch size
            shuffle=True,
        )


        for _ in range(epochs_per_step): # Train for specified epochs on collected data
            for experience_batch in experience_loader:
                optimizer.zero_grad()

                # Move batch to training device
                seq_ids = experience_batch.sequence_ids.to(device)
                action_mask = experience_batch.action_mask.to(device)
                returns = experience_batch.returns.to(device)
                advantages = experience_batch.advantages.to(device)
                ref_log_probs = experience_batch.ref_log_probs.to(device)

                # Calculate current log probs using the trainable model
                attention_mask = (seq_ids != pad_token_id).to(device)
                log_probs = sequences_log_probs(
                    model, seq_ids, attention_mask=attention_mask
                )

                # Calculate loss using the GRPO objective
                loss, stats = objective(
                    log_probs=log_probs,
                    old_log_probs=ref_log_probs,
                    advantages=advantages,
                    returns=returns,
                    action_mask=action_mask,
                )

                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=max_norm) # Clip gradients
                optimizer.step()
                global_step += 1

                # Log stats (optional)
                if wandb_project is not None:
                    wandb.log({**stats, "loss": loss.item()}, step=global_step)
                print(f"Step: {global_step}, Loss: {loss.item():.4f}, Reward Mean: {stats.get('reward/mean', 0.0):.3f}, KL: {stats.get('kl_div', 0.0):.4f}")


        # Save checkpoint periodically
        if (step + 1) % checkpoint_interval == 0:
            output_dir = checkpoint_path / f"checkpoint-{step + 1}"
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Checkpoint saved to {output_dir}")

    print("Training finished.")
    # Save final model
    output_dir = checkpoint_path / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final model saved to {output_dir}")


if __name__ == "__main__":
    main()
