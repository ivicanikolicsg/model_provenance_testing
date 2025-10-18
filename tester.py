from misc import *
from cache_parent_prompts import load_cached_prompts
from sampler_classes import get_sampler
from llm_functions import load_llm_on_device, clear_model_tokenizer, get_batch_model_output_single




# Model provenance tester
# given hits and trials for each model, finds the best model and check if it is significantly better than all others
def find_best_model(hits, trials, alpha):
    probs = [x / y if y > 0 else 0 for x,y in zip(hits, trials)]
    best_idx = np.argmax(probs)
    
    # Compare against all other models
    n = len(hits)
    p_values = []

    for i in range(n):
        if i == best_idx:
            continue
            
        p_value = compare_bernoulli_means(
            hits[best_idx], trials[best_idx],
            hits[i], trials[i]
        )
        p_values.append(p_value)
    
    # Sort p-values while keeping track of original indices
    sorted_pairs        = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_indices      = [pair[0] for pair in sorted_pairs]
    sorted_p_values     = [pair[1] for pair in sorted_pairs]
    
    # Apply Holm-Bonferroni
    significant = [False] * (n-1)
    for k, p_value in enumerate(sorted_p_values):
        if p_value > alpha / (n - 1 - k):  # n-1 total comparisons
            break
        significant[sorted_indices[k]] = True
    
    is_significant = all(significant)
    max_p_value = max(p_values)
    
    return best_idx, is_significant, max_p_value





if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Model provenance tester')
    parser.add_argument('--prompt_id',      help='Id of the cached prompts for the parents. Use -1 to create a new.',type=int, default=-1)  
    parser.add_argument('--file_parents',   help='Path to the parents file',type=str,)  
    parser.add_argument('--file_candidates',help='Path to the candidates file, one per line, either <child_model, parent_model> or <child_model> only',type=str, )  
    parser.add_argument('--no_prompts',     help='(max) Number of prompts', type=int, default=5_000)
    parser.add_argument('--sampler',        help='Sampler type ', type=str, default='random_sentences')
    parser.add_argument('--sampler_file',   help='Path to the random sentences file', type=str, default=file_random_sentences)
    parser.add_argument('--sample_same',    help='Sample the same set of prompts across all testers', action='store_true', )
    parser.add_argument('--alpha',          help='Significance level alpha', type=float, default=0.05)
    parser.add_argument('--device',         help='Specify "cpu" or "cuda" or "cuda:1" if you want to specific device(s). By default, it will use all available GPUs, if none, then only CPU', type=str, default=None)
    parser.add_argument('--batch_size',     help='Batch size', type=int, default=128)
    parser.add_argument('--only_models',    help='Consider only these models  for testing', type=str, default=None)
    parser.add_argument('--only_parents',   help='Consider only these parents for testing', type=str, default=None)
    parser.add_argument('--redo_bad',       help='Redo models that failed instead of ignoring them', action='store_true', default=False)
    parser.add_argument('--no_file',        help='Do not produce file output', action='store_true', default=False)
    parser.add_argument('--quick_test',     help='Run a quick test without needing parent/candidate files and just provide comma separated list of models', type=str)

    args                    = parser.parse_args()
    prompt_id               = args.prompt_id
    file_parents            = args.file_parents
    file_candidates         = args.file_candidates
    no_prompts              = args.no_prompts
    sampling_policy         = args.sampler
    samp_file               = args.sampler_file
    sample_same             = args.sample_same
    batch_size              = args.batch_size
    alpha                   = args.alpha
    user_device             = args.device
    only_models             = None if args.only_models is None else [x.strip() for x in args.only_models.split(',')]
    only_parents            = None if args.only_parents is None else [x.strip() for x in args.only_parents.split(',')]
    redo_bad                = args.redo_bad
    no_file                 = args.no_file

    if not args.quick_test:
        if not args.file_parents or not args.file_candidates:
            parser.error("--file_parents and --file_candidates are required unless --quick_test is used.")
    else:
        prompt_id       = 0
        file_parents    = 'data/bench_A_and_B_parents.txt'
        file_candidates = 'data/quick_test_candidates.txt'
        candidates = [ x.strip() for x in args.quick_test.split(',') if x.strip() != '' ]
        with open(file_candidates, 'w') as f:
            for c in candidates:
                f.write(c + '\n')




    random.seed( int(time.time()) )


    # load parent models
    parents = load_models_from_file(file_parents)

    # get the sampler (by default it is random sentences)
    sampler = get_sampler(parents, sampling_policy, samp_file)

    # load cached prompts of parents (will produce if missing)
    input_list, input_word_to_model = load_cached_prompts(prompt_id, parents, sampler, no_prompts, batch_size=batch_size, user_device=user_device)

    # load provenance candidates from file 
    # each line can be of the form:
    #   1. <child_model, parent_model>, then the parent model is used as ground truth (and NOT to infer the parent). the parent can be None
    #   2. <child_model> then the parent is assumed unknown. 
    if not os.path.exists(file_candidates):
        print(f"Error: candidates file {file_candidates} not found")
        sys.exit(1)
    provenance_pairs = []
    with open(file_candidates, 'r') as file:
        for line in file:
            fields = line.strip().split(',')
            if not fields or fields[0] == '': 
                continue
            if len(fields) == 1:
                child_model, parent_model, parent_provided = fields[0], None, False 
            else:
                child_model, parent_model, parent_provided = fields[0], fields[1], True
                if parent_model == "None":
                    parent_model = None
            provenance_pairs.append((child_model, parent_model, parent_provided))
    print(f'\nFound {len(provenance_pairs)} candidates for testing of provenance')

    # if want to restrict to only some models or parents
    if only_models is not None:
        provenance_pairs = [ x for x in provenance_pairs if x[0] in only_models ]
    if only_parents is not None:
        parents = [ x for x in parents if x in only_parents ]


    # output files
    unix_time = int(time.time())
    file_out            = f'{tester_runs_folder}/tester_{unix_time}.csv'
    if not no_file:
        print('\nResults will be written to file: ' + file_out)
        # write the header
        with open(file_out, 'w') as file:
            file.write('outcome,precision,recall,accuracy,p-value,highest mu, highest hits, tot prompts, model name, parent true, parent guess\n')


    # true/false positive/negatives
    TP = FP = TN = FN = TOT = 0
    random_prompt_indices   = random.sample(range(len(input_list)), min(no_prompts  ,len(input_list)))

    ii = 0
    total_prov_pairs = len(provenance_pairs)
    while len(provenance_pairs) > 0:
        
        ppair                                = provenance_pairs.pop(0)
        mn1, parent_true, parent_provided    = ppair

        if only_models is not None and mn1 not in only_models:
            continue

        if mn1 in parents and parent_provided:
            print(f"\nSkipping {mn1} as it is in parent models")
            continue

        print('\n' + ('-'*80), flush=True )
        print(f"{ii:4d}/{total_prov_pairs:4d} Provenance for model {mn1}", flush=True)

        # load the model
        use_device, model, tokenizer, error = load_llm_on_device(mn1, user_device)
        if model is None or tokenizer is None:
            print(f"Error: could not load model {mn1}")
            if redo_bad:
                provenance_pairs.append(ppair)
            continue


        hits = { x: 0 for x in parents }
        tots = { x: 0 for x in parents }

        # sample the prompts from all available
        if not sample_same:
            random_prompt_indices   = random.sample(range(len(input_list)), min(no_prompts,len(input_list)))
        sampled_input_list          = [input_list[i] for i in random_prompt_indices]
        sampled_input_word_to_model = [input_word_to_model[i] for i in random_prompt_indices]
        total_trials                = 0
        
        # process prompts 
        for jj in range(0, len(sampled_input_list), no_prompts):

            end_idx = min( jj + no_prompts, len(sampled_input_list) )
            total_trials += (end_idx - jj)
    
            # produce outputs for this model and batch
            mod_outs = get_batch_model_output_single(use_device, model, tokenizer, sampled_input_list[jj:end_idx], batch_size=128)
            
            # increase hits for each parent that produced the same output
            for i in range(len(mod_outs)):
                full_output = mod_outs[i]
                if full_output in sampled_input_word_to_model[jj+i]:
                    for model_name in sampled_input_word_to_model[jj+i][full_output]:
                        hits[model_name] += 1


        # clear the model
        clear_model_tokenizer(model, tokenizer)

        # find the best model according to our algorithm
        best_index, is_significant, max_p_value = find_best_model([hits[x] for x in parents], [total_trials for x in parents], alpha)
        if is_significant:
            parent_guess = parents[best_index]
        else:
            parent_guess = None
        

        # for logging purposes
        scores = {}
        for i in range(len(parents)):
            scores[i] = (hits[parents[i]] / (total_trials) , hits[parents[i]], total_trials )
        sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        best_prob = scores[best_index][0]
        best_hits = scores[best_index][1]
        best_tot  = scores[best_index][2]


        if not parent_provided:

            print(f"Guessed parent       : {parent_guess} ")
            print(f'P-value              : {max_p_value:.5f}')

        else:

            # compute TP,FP,TN,FN, accuracy, precision, recall
            oneTP = 1 if parent_true is not None and parent_guess is not None and parent_true == parent_guess else 0
            oneFP = 1 if (parent_true is None and parent_guess is not None) or \
                    (parent_true is not None and parent_guess is not None and parent_true != parent_guess) else 0
            oneTN = 1 if parent_true is None and parent_guess is None else 0
            oneFN = 1 if parent_true is not None and parent_guess is None else 0
            TP += oneTP
            FP += oneFP
            TN += oneTN
            FN += oneFN
            TOT += 1
            precision   = TP / (TP + FP) if TP + FP > 0 else 0
            recall      = TP / (TP + FN) if TP + FN > 0 else 0
            accuracy    = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN > 0 else 0

            if oneTP: print(f"\033[92m < TP > \033[0m")
            if oneTN: print(f"\033[93m < TN > \033[0m")
            if oneFP: print(f"\033[91m < FP > \033[0m")
            if oneFN: print(f"\033[91m < FN > \033[0m")

            print(f"\tTrue    parent       : {parent_true}" + ('' if parent_true is None else f" with score   {scores[parents.index(parent_true)] if parent_true in parents else 0} "))
            print(f"\tGuessed parent       : {parent_guess} ")
            print(f'\tP-value              : {max_p_value:.5f}')
            print(f"\tprecision : {precision:.2f} recall: {recall:.2f} accuracy: {accuracy:.2f} : TP {TP:5d}, FP {FP:5d}, TN {TN:5d}, FN {FN:5d}")



            # write to file
            if not no_file:
                with open(file_out, 'a') as file:
                    fr = 'TP' if oneTP else 'FP' if oneFP else 'TN' if oneTN else 'FN'
                    file.write( f"{fr},  {precision:4.2f},{recall:4.2f},{accuracy:4.2f},  "
                                f"{max_p_value:.8f},{best_prob:.4f},{best_hits:4d},{best_tot:5d},    "
                                f"{mn1},{parent_true},{parent_guess}\n")


        # top 5 scores (for logging purposes)
        print('Top 5 scores:')
        for i in range(5):
            print(f"{i:3d} : {parents[sorted_scores[i][0]]:50s}    mu  {sorted_scores[i][1][0]:.3f}      {sorted_scores[i][1][1]:4d}  /  {sorted_scores[i][1][2]:5d}    ")



        

        ii += 1
