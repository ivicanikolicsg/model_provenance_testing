from misc import *
from sampler_classes import *
from llm_functions import *



def get_last_cache_id(folder_path):
    if not os.path.exists(folder_path):
        return -1
    
    max_id = -1
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            match = re.match(r'^(\d+)_', item)
            if match:
                current_id = int(match.group(1))
                max_id = max(max_id, current_id)
    
    return max_id



def cache_prompts(parents, sampler, no_prompts=5_000, chunks=1, batch_size=128, user_device=None):

    print(f'Caching prompts of {len(parents)} parent models')

    last_cache_id = get_last_cache_id(cached_prompts_folder)
    new_cache_id = last_cache_id + 1

    cache_folder = f'{cached_prompts_folder}/{new_cache_id}__{datetime.now().strftime("%Y-%m-%d")}__no_models_{len(parents)}__no_prompts_{no_prompts}__{sampler.name}__{chunks}'
    os.makedirs(cache_folder, exist_ok=True)
    print(f'Caching prompts to folder: {cache_folder}')


    # devices to load models
    devices = get_all_devices()
    if user_device is not None:
        devices = [torch.device(user_device.strip())]


    same        = { (mn1, mn2): 0 for mn1 in parents for mn2 in parents if mn1 != mn2 }
    hits        = [ 0 for mn in parents ]
    no_iter     = 0
    outputs     = []
    sampled_prompts = []
    mfact    = 0.1
    batches  = no_prompts * chunks

    while True:

        no_iter += 1


        # produce outputs for all models
        if len(outputs) < chunks:
            print(f'\nProducing {batches} outputs for {len(parents)} models to get {no_prompts} outputs', flush=True)
            new_prompts     = [ sampler.sample(no_repeat=True) for i in range(batches) ]
            new_outputs     = [ [] for i in range(len(new_prompts)) ]
            for mi, model_name in enumerate(parents):
                time_start = time.time()

                print(f'\n{mi+1:3d}/{len(parents):3d} : {model_name}', flush=True)
                print(f'\tLoading model on device {devices[0]}', flush=True)
                device, model, tokenizer, error_type = load_llm_on_device(model_name, devices[0])
                if error_type == 'out_of_memory' and len(devices) > 1:
                    print(f'\tOut of memory error, trying next device', flush=True)
                    device, model, tokenizer, error_type = load_llm_on_device(model_name, devices[1])
                if model is None or tokenizer is None:
                    raise ValueError(f"Error: could not load model {model_name}, cannot proceed")

                print(f'\tQuerying on {len(new_prompts)} prompts', flush=True)
                mod_outs = get_batch_model_output_single(device, model, tokenizer, new_prompts, batch_size=batch_size)
                for i in range(len(new_prompts)):
                    new_outputs[i].append( mod_outs[i] )

                clear_model_tokenizer(model, tokenizer)
                        
                print(f"\tSeconds to process {time.time()-time_start:.1f}", flush=True)

            outputs.extend(new_outputs)
            sampled_prompts.extend(new_prompts)


        # infer for all models
        full_results = []
        prompts      = []
        for i in range(chunks):
            full_results.append(outputs.pop(0))
            prompts.append(sampled_prompts.pop(0))


        #
        # not greedy
        # 
        if 1 == chunks:

            # write to file
            filename = f'{cache_folder}/iter_{no_iter}.pkl'
            with open(filename, 'wb') as file:
                prompt          = prompts[0]
                model_outputs   = { parents[i]:full_results[0][i] for i in range(len(parents)) }
                pickle.dump( (prompt, model_outputs, 0), file)

            no_prompts -= 1
            if no_prompts <= 0:
                break

            continue


        #
        # greedy search for best prompt
        #

        best_score      = None
        best_result     = None
        best_prompt     = None
        best_pair       = None



        # compute max/min values (for assigning weights)        
        max_hit         = max(hits)
        min_p_value     = 1
        max_p_value     = 0
        p_value_scores  = dict()
        for j1 in range(len(parents)):
            for j2 in range(len(parents)):
                if j1 == j2: continue
                # do not use Fischer for smaller values because then it requires too much time and
                # too many not so good prompts
                cbm = compare_bernoulli_means(  mfact * hits[j1], 
                                                hits[j1], 
                                                mfact * same[ (parents[j1], parents[j2]) ] , 
                                                hits[j1],
                                                small_count_threshold = -1 ) 
                min_p_value = min(min_p_value, cbm)
                max_p_value = max(max_p_value, cbm)

                p_value_scores[ (parents[j1], parents[j2]) ] = cbm
        if max_p_value == 0: 
            max_p_value = 1

        # print bottom 10 lowest hits
        print('\nBottom 20 lowest hits')
        hits_list = [ (k, v) for k,v in zip(parents, hits) ]
        hits_list.sort(key=lambda x: x[1])
        for i in range( min(20, len(hits_list)) ):
            print(f'{hits_list[i][1]:4d} {hits_list[i][0]:40s}', flush=True)
        
        # print top 10 p_value_scores
        print('\nTop 20 p_value_scores')
        p_values_scores_list = [ (k, v) for k,v in p_value_scores.items() ]
        p_values_scores_list.sort(key=lambda x: x[1], reverse=True)
        for i in range( min(20, len(p_values_scores_list)) ):
            print(f'{hits[parents.index(p_values_scores_list[i][0][0])]:4d} {p_values_scores_list[i][0][0]:40s}  {hits[parents.index(p_values_scores_list[i][0][1])]:4d} {p_values_scores_list[i][0][1]:40s} : { same[ p_values_scores_list[i][0] ]:5d} : {p_values_scores_list[i][1]:.8f}  ', flush=True)

        # decide whether to try to increase min hits or min p_value        
        iter_type       = 'items' if no_iter < 2 or random.random() < 0.25 else 'p_value'

        print(f'\nIter {no_iter} Type: {iter_type} max_hit: {max_hit} min/max p_value: {min_p_value}  {max_p_value}', flush=True)


        # find best prompt
        if iter_type == 'items':
            for i in range(chunks):
                score = 0
                for j in range(len(parents)):
                    word = full_results[i][j]
                    score += (1 if len(word) > 0 else 0) * 2**(max_hit - hits[j])
                if best_score is None or score > best_score:
                    best_score  = score
                    best_prompt = prompts[i]
                    best_result = copy.deepcopy(full_results[i])
                    print('\tNew score:', best_score, flush=True)

        else:
            for i in range(chunks):
                score = 0
                p_values_improved = 0
                for j1 in range(len(parents)):
                    words_model_1 = {full_results[i][j1]} if len(full_results[i][j1]) > 0 else set()
                    for j2 in range(len(parents)):
                        if j1 == j2: continue
                        words_model_2 = {full_results[i][j2]} if len(full_results[i][j2]) > 0 else set()

                        match = words_model_1.intersection( words_model_2 )

                        old_p_value = compare_bernoulli_means( 
                            mfact * hits[j1], 
                            hits[j1], 
                            mfact * same[ (parents[j1], parents[j2]) ], 
                            hits[j1],
                            small_count_threshold = -1  )
                        new_p_value  = compare_bernoulli_means( 
                            mfact * (hits[j1] + (1 if len(words_model_1) > 0 else 0)), 
                            hits[j1],
                            mfact * (same[ (parents[j1], parents[j2]) ] + (1 if len(match) > 0 else 0)), 
                            hits[j1],
                            small_count_threshold = -1  )
                        weight = 2**(15*old_p_value/max_p_value)
                        score += (1 if new_p_value < old_p_value else 0) * weight
                        p_values_improved += 1 if new_p_value < old_p_value else 0

                if score > 0 and \
                    (best_score is None or score > best_score):
                        best_score      = score
                        best_prompt     = prompts[i]
                        best_result     = copy.deepcopy(full_results[i])
                        print('\tNew score:', int(best_score), p_values_improved, flush=True)


        if best_prompt is None:
            print('No good next token found', flush=True)
            if iter_type == 'p_value':
                #
                # just switch to items iteration type
                #
                print('Trying items search', flush=True)
                for i in range(chunks):
                    score = 0
                    for j in range(len(parents)):
                        output = full_results[i][j]
                        score += (1 if len(output) > 0 else 0) * 2**(max_hit - hits[j])
                    if best_score is None or score > best_score:
                        best_score  = score
                        best_prompt = prompts[i]
                        best_result = copy.deepcopy(full_results[i])
                        print('\tNew score:', int(best_score), flush=True)                

            if best_prompt is None:
                print('No good next token found', flush=True)
                continue

        # write to file
        filename = f'{cache_folder}/iter_{no_iter}.pkl'
        with open(filename, 'wb') as file:
            prompt = best_prompt
            model_outputs = { mn: best_result[ parents.index(mn) ] for mn in parents }
            pickle.dump( (prompt, model_outputs, best_score), file)

        print(f'\n[{no_iter:7d}] prompt: {best_prompt} Score: {int(best_score)}', flush=True)
        for i, mn1 in enumerate(parents):
            output1 = best_result[i]
            
            # Update hits if the output string is not empty
            if len(output1) > 0:
                hits[i] += 1

            # Update same_hits by comparing strings directly
            for j, mn2 in enumerate(parents):
                if i >= j: continue  # Avoid self-comparison and double-counting
                
                output2 = best_result[j]
                
                # If both models produced the same non-empty output, count it
                if len(output1) > 0 and output1 == output2:
                    same[ (mn1, mn2) ] += 1
                    same[ (mn2, mn1) ] += 1 # The 'same' dict should be symmetrical    


        no_prompts -= 1
        if no_prompts <= 0:
            break

    return new_cache_id

def load_cached_prompts(cache_id, parents, sampler, no_prompts, batch_size=128, user_device=None):

    # create new cache prompts
    if cache_id < 0:
        cache_id = cache_prompts(parents, sampler, no_prompts=no_prompts, chunks=1, batch_size=batch_size, user_device=user_device)

    pickle_files = glob.glob(f'{cached_prompts_folder}/{cache_id}_*/*.pkl')
    if 0 == len(pickle_files):
        max_cache = get_last_cache_id(cached_prompts_folder)
        raise ValueError(f'No cached prompts found for cache id {cache_id} in folder {cached_prompts_folder}. Max cache id is {max_cache}')
    print(f'\nLoading cached prompts from {len(pickle_files)} files from cache id {cache_id}')
    prompts = []
    for f in pickle_files:
        with open(f, 'rb') as file:
            prompts.append(pickle.load(file))
        if len(prompts) >= no_prompts:
            break

    tot_model_outputs = { x: 0 for x in parents }
    input_list = []
    input_word_to_model = []
    for d in prompts:
        phrase = d[0]
        input_list.append( phrase )
        word_to_model = dict()
        for model in d[1]:
            if model not in tot_model_outputs:
                continue

            full_output = d[1][model]
            # cannot be in the prompt
            if phrase in full_output:
                full_output = ''
            if len(full_output) > 0:
                tot_model_outputs[model] += 1 
                if full_output not in word_to_model:
                    word_to_model[full_output] = []
                word_to_model[full_output].append(model)

        input_word_to_model.append(word_to_model)

    print('Total parent models with non-empty outputs:', len([x for x in tot_model_outputs if tot_model_outputs[x] > 0]))

    # check if all models have at least one output
    for model in tot_model_outputs:
        if tot_model_outputs[model] == 0:
            print(f'\033[91mWarning: model {model} has no outputs in the cached prompts\033[0m')


    return input_list, input_word_to_model



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cache outputs of parents')
    parser.add_argument('--file_parents',   help='Path to the parents file',type=str, required=True)  
    parser.add_argument('--no_prompts',     help='Number of prompts', type=int, default=5_000)
    parser.add_argument('--chunks',         help='Number of chunks for greedy search (by default, chunks=1, meaning no greedy search)', type=int, default=1)
    parser.add_argument('--sampler',        help='Sampler type', type=str, default='random_sentences')
    parser.add_argument('--sampler_file',   help='Path to the random sentences file', type=str, default=file_random_sentences)
    parser.add_argument('--batch_size',     help='Batch size', type=int, default=128)
    parser.add_argument('--device',         help='Specify "cpu" or "cuda" or "cuda:1" if you want to specific device(s). By default, it will use all available GPUs, if none, then only CPU', type=str, default=None)


    args            = parser.parse_args()
    file_parents    = args.file_parents
    no_prompts      = args.no_prompts
    chunks          = args.chunks     # search greedily among this many trials
    sampling_policy = args.sampler
    samp_file       = args.sampler_file
    batch_size      = args.batch_size
    user_device     = args.device

    parents         = load_models_from_file(file_parents)
    sampler         = get_sampler(parents, sampling_policy, samp_file)
    cache_prompts(parents, sampler, no_prompts=no_prompts, chunks=chunks, batch_size=batch_size, user_device=user_device)
