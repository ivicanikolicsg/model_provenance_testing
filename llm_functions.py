import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub.errors import RepositoryNotFoundError, LocalEntryNotFoundError
from misc import *


def get_device(random=False, print_info=False):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    # Get the list of GPUs
    nvidia_smi = "nvidia-smi --query-gpu=memory.free --format=csv"
    output = subprocess.check_output(nvidia_smi.split())
    output = output.decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for x in output]

    if print_info:
        print(f"Available GPU memory: {memory_free_values}")
    
    # Find the GPU with the most free memory
    max_memory_gpu = max(range(len(memory_free_values)), key=lambda i: memory_free_values[i])

    if random:
        import random
        max_memory_gpu = random.randint(0, len(memory_free_values)-1)
    
    return torch.device(f"cuda:{max_memory_gpu}")


def get_all_devices():
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] + [torch.device("cpu")]



def safe_tokenize(device, model, tokenizer, input_text):
    # Define fallback token ID
    fallback_token_id = getattr(tokenizer, 'unk_token_id', None)
    if fallback_token_id is None:
        fallback_token_id = 0  # Using 0 as a safe default
    
    # If empty input, return a single fallback token
    if not input_text or input_text.strip() == "":
        return torch.tensor([[fallback_token_id]], device=device)
    
    # Get the actual vocab size and be super conservative
    real_vocab_size = model.get_input_embeddings().weight.shape[0]
    safe_vocab_size = real_vocab_size
    
    # First encode
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # If tokenization gave empty tensor, return fallback token
    if input_ids.numel() == 0:
        return torch.tensor([[fallback_token_id]], device=device)
    
    # Replace ALL tokens that are even close to the limit
    mask = input_ids >= safe_vocab_size
    if torch.any(mask):
        input_ids[mask] = fallback_token_id
    
    # Double check - specify dim for max operation
    if input_ids.dim() > 1:
        max_token = torch.max(input_ids, dim=-1)[0].item()
    else:
        max_token = torch.max(input_ids).item()
    
    # Final verification
    if max_token >= safe_vocab_size:
        print("Emergency token replacement activated")
        input_ids = torch.clamp(input_ids, 0, safe_vocab_size - 1)
    
    return input_ids



def get_model_output(device, model, tokenizer, input_text, top_k=1):
    input_ids = safe_tokenize(device, model, tokenizer, input_text)

    # Get model's max length
    max_model_length = getattr(model.config, 'max_position_embeddings', 256)
    
    # Truncate if necessary
    if input_ids.shape[1] > max_model_length:
        #print(f"Truncating input from {input_ids.shape[1]} to {max_model_length} tokens")
        input_ids = input_ids[:, -max_model_length:]  # Take last max_model_length tokens

    real_vocab_size = model.get_input_embeddings().weight.shape[0]
    max_token = torch.max(input_ids).item()

    try:
        with torch.no_grad():
            output = model(input_ids)
            logits = output.logits
    
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
        top_k_words = [tokenizer.decode(i).strip() for i in top_k_indices]
        
        return top_k_probs.tolist(), top_k_indices.tolist(), top_k_words

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return [], [], []



def get_batch_model_output(device, model, tokenizer, input_texts, top_k=1, batch_size=8):
    all_probs = []
    all_indices = []
    all_words = []
    
    # Define a fallback token ID (usually 0 or 1 works for most models)
    fallback_token_id = getattr(tokenizer, 'unk_token_id', None)
    if fallback_token_id is None:
        fallback_token_id = 0  # Using 0 as a safe default
    
    # Process in batches
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i + batch_size]
        
        # Tokenize each text separately
        batch_input_ids = []
        max_length = getattr(model.config, 'max_position_embeddings', 256)
        
        for text in batch_texts:
            try:
                # Use safe_tokenize instead of direct tokenizer call
                input_ids = safe_tokenize(device, model, tokenizer, text)
                
                # Truncate if necessary
                if input_ids.shape[1] > max_length:
                    input_ids = input_ids[:, -max_length:]
                    
                batch_input_ids.append(input_ids)
            except Exception as e:
                print(f"Error during tokenization: {str(e)}")
                # Add a default tensor with fallback token
                batch_input_ids.append(torch.tensor([[fallback_token_id]], device=device))

        processed_in_batch = 0
        try:
            # Process each input separately
            for input_ids in batch_input_ids:
                try:
                    with torch.no_grad():
                        # Add attention_mask
                        attention_mask = torch.ones_like(input_ids)
                        outputs = model(
                            input_ids,
                            attention_mask=attention_mask,
                            output_attentions=False
                        )
                        logits = outputs.logits

                    if logits.shape[1] > 0:
                        next_token_logits = logits[0, -1, :]
                        probs = F.softmax(next_token_logits, dim=-1)
                        top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                        top_k_words = [tokenizer.decode(i).strip() for i in top_k_indices]
                    else:
                        top_k_probs = torch.zeros(top_k, device=device)
                        top_k_indices = torch.zeros(top_k, dtype=torch.long, device=device)
                        top_k_words = [''] * top_k

                    all_probs.append(top_k_probs.tolist())
                    all_indices.append(top_k_indices.tolist())
                    all_words.append(top_k_words)
                    processed_in_batch += 1

                except Exception as e:
                    print(f"Error processing single item in batch: {str(e)}")
                    all_probs.append([0.0] * top_k)
                    all_indices.append([0] * top_k)
                    all_words.append([''] * top_k)
                    processed_in_batch += 1
                    
        except Exception as e:
            print(f"Error during batch processing: {str(e)}.")
            remaining = len(batch_texts) - processed_in_batch
            for _ in range(remaining):
                all_probs.append([0.0] * top_k)
                all_indices.append([0] * top_k)
                all_words.append([''] * top_k)
    
    return all_probs, all_indices, all_words




def get_batch_model_output_single(device, model, tokenizer, input_texts, batch_size=8):
    final_outputs = []
    
    all_probs, _, all_words = get_batch_model_output(device, model, tokenizer, input_texts, top_k=1, batch_size=batch_size)
    
    for i in range(len(input_texts)):
        words = all_words[i]
        if len(words) == 0:
            final_outputs.append('')
        else:
            final_outputs.append(words[0])
    
    return final_outputs




def load_llm_on_device(model_name, user_device):
    if user_device is not None:
        use_device = torch.device(user_device)
    else:
        use_device = get_device()

    model = tokenizer = None
    error_type = None
    
    # Try local files first
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False, trust_remote_code=False, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False, local_files_only=True).to(use_device)
        return use_device, model, tokenizer, error_type
    except OSError:
        pass
    
    # Try downloading from hub
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False).to(use_device)
        return use_device, model, tokenizer, error_type
    except (RepositoryNotFoundError, LocalEntryNotFoundError, OSError):
        error_type = 'repository_not_found'
        return use_device, None, None, error_type
    except torch.cuda.OutOfMemoryError:
        error_type = 'out_of_memory'
        return use_device, None, None, error_type
    except AttributeError:
        pass
    
    # Try force download for corrupted cache
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False, trust_remote_code=False, force_download=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False, force_download=True).to(use_device)
        return use_device, model, tokenizer, error_type
    except torch.cuda.OutOfMemoryError:
        error_type = 'out_of_memory'
        return use_device, None, None, error_type
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        error_type = 'general'
        return use_device, None, None, error_type

def clear_model_tokenizer(model, tokenizer):
    try:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during clearing model/tokenizer: {str(e)}")