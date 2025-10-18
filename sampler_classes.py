from misc import *



def model_name_to_file(model_name):
   return f"{model_name.replace('/', '_')}" 


class SamplingPolicy:
    def __init__(self, basemodel_names, filepath=None):
        self.basemodel_names = basemodel_names
        self.filepath = filepath
        self.allowed_output = None
        self.name = self.__class__.__name__.lower().replace('sampling', '').replace('randomtokens', 'random_tokens')    
        self.setup()

    def setup(self):
        pass

    def sample(self, no_repeat=False):
        raise NotImplementedError("Sampling method must be implemented")


class RandomTokensSampling(SamplingPolicy):
    def setup(self):
        self.prompt_len_range = [3, 20]
        
        # get most popular tokens FIRST
        self.freq_allowed_tokens = self._get_allowed_tokens(self.basemodel_names)
        threshold_num = 0.75 * len(self.basemodel_names)
        self.allowed_tokens = list( {k for k in self.freq_allowed_tokens if self.freq_allowed_tokens[k] > threshold_num} ) 
        print(f'Found {len(self.allowed_tokens)} allowed tokens')

        # get prohibited tokens SECOND (this will use the initial allowed_tokens)
        if self.basemodel_names is not None:
            self.prohibited_tokens = produce_prohibited_tokens(
                sampler=self,
                models=self.basemodel_names,
            )
        else:
            self.prohibited_tokens = set()
        print(f'Found {len(self.prohibited_tokens)} prohibited tokens')

        # FINALLY remove prohibited tokens from allowed tokens
        self.allowed_tokens = list(set(self.allowed_tokens).difference(self.prohibited_tokens))
        print(f'Found {len(self.allowed_tokens)} allowed tokens after removing prohibited tokens')

        # produce allowed output
        self.allowed_output = self.allowed_tokens

        # confirm allowed_output are sufficient for each model
        self._confirm_allowed_tokens()

    def sample(self, no_repeat=False):
        return self._generate_prompt(self.allowed_tokens, self.prompt_len_range)

    def _get_allowed_tokens(self, model_names):
        hash_basemodels = hashlib.md5(str(model_names).encode()).hexdigest()
        pickle_allowed_tokens = f'{cached_tokens_folder}/allowed_tokens_{hash_basemodels}.pkl'
        if os.path.exists(pickle_allowed_tokens):
            with open(pickle_allowed_tokens, 'rb') as file:
                return pickle.load(file)

        print(f'Producing allowed tokens for {len(model_names)} basemodels  {hash_basemodels}', flush=True)
        allowed_tokens = None
        freq_tokens = dict()
        for mn in model_names:

            pickle_tokens = f'{cached_tokens_folder}/tokens_{model_name_to_file(mn)}.pkl'
            
            if not os.path.exists(pickle_tokens):
                print(f'Producing tokens for {mn:50s} ... ', flush=True, end='')
                device = get_device(random=True)
                tokenizer = AutoTokenizer.from_pretrained(mn, clean_up_tokenization_spaces=False, trust_remote_code=False, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(mn, trust_remote_code=False, local_files_only=True).to(device)
                _, _, tokens = get_model_output(device, model, tokenizer, 'any text', top_k=1_000_000)
                tokens = set(tokens)
                with open(pickle_tokens, 'wb') as file:
                    pickle.dump(tokens, file)
                print(f'Done,  produced {len(tokens):8d} tokens', flush=True)

            # load the tokens
            with open(pickle_tokens, 'rb') as file:
                tokens = pickle.load(file)

            if allowed_tokens is None:
                allowed_tokens = tokens
            else:
                allowed_tokens = allowed_tokens.intersection(tokens)

            for token in tokens:
                if token not in freq_tokens:
                    freq_tokens[token] = 0
                freq_tokens[token] += 1

            print(f'\t\tTotal tokens: {len(tokens)}', flush=True)

        with open(pickle_allowed_tokens, 'wb') as file:
            pickle.dump(freq_tokens, file)   
        return freq_tokens

    def _confirm_allowed_tokens(self):
        for mn in self.basemodel_names:
            pickle_tokens = f'{cached_tokens_folder}/tokens_{model_name_to_file(mn)}.pkl'
            with open(pickle_tokens, 'rb') as file:
                tokens = pickle.load(file)
            
            model_allowed = tokens.intersection(self.allowed_tokens)
            print(f'{mn:50s} : {len(model_allowed):8d} / {len(tokens):8d} allowed tokens')



    def _generate_prompt(self, allowed_tokens, prompt_len_range):
        toks = allowed_tokens
        return ''.join(random.choices(toks, k=random.randint(prompt_len_range[0], prompt_len_range[1])))


class RandomWords(SamplingPolicy):
    def setup(self):
        self.common_nouns = [ "time", "person", "year", "way", "day", "thing", "man", "world", "life", "hand", "part", "child", "eye", "woman", "place", "work", "week", "case", "point", "company","number", "group", "problem", "fact", "idea", "water", "money", "question", "area", "family", "head", "book", "house", "power", "game", "line", "end", "member", "law", "car", "city", "community", "name", "room", "business", "issue", "market", "court", "party", "school", "country", "state", "mind", "system", "story", "food", "month", "book", "paper", "music","data", "theory", "law", "bird", "mother", "movie", "action", "teacher", "phone", "dog","cat", "tree", "computer", "friend", "father", "student", "heart", "door", "office", "art","war", "history", "window", "table", "air", "earth", "sun", "product", "building", "road","girl", "boy", "food", "animal", "family", "doctor", "river", "sea", "phone", "night","word", "plant", "fish", "street", "king", "home", "garden", "box", "island", "kitchen","hotel", "planet", "radio", "chair", "letter", "morning", "film", "people", "baby", "army","bank", "body", "class", "club", "college", "wall", "road", "rain", "stage", "space","store", "summer", "train", "winter", "church", "rock", "skill", "team", "paper", "wood","fire", "field", "ground", "horse", "ship", "light", "metal", "oil", "picture", "plant","river", "sound", "stone", "town", "village", "wind", "arm", "beach", "boat", "book","camera", "card", "case", "clock", "cloud", "coat", "corn", "desk", "door", "dust", "farm", "feet", "fish", "flag", "floor", "fruit", "glass", "gold", "grass", "hall","hat", "hill", "ice", "iron", "key", "lake", "land", "leaf", "leg", "library", "list", "map", "meat", "milk", "moon", "mouth", "page", "park", "pen", "pencil","plane", "plate", "pool", "rain", "ring", "river", "road", "room", "salt", "sand", "sea", "seat", "shoe", "shop", "sign", "silver", "snow", "soil", "star", "street","sugar", "table", "teeth", "ticket", "tire", "tool", "tooth", "town", "train", "tree", "truck", "voice", "watch", "water", "wheel", "window", "wing", "wire", "wood", "wool","world", "writer", "zoo", "apple", "arm", "banana", "basin", "bath", "bed", "bee", "bell", "berry", "bird", "blade", "board", "bone", "book", "bottle", "box", "brain", "brake", "branch", "bread", "brick", "bridge", "brush", "bucket", "bulb", "button", "cake","camera", "card", "cart", "cheese", "chest", "chin", "church", "circle", "clock", "cloud","coat", "collar", "comb", "cord", "cow", "cup", "curtain", "desk", "dish", "dog","door", "drain", "drawer", "dress", "drink", "drum", "ear", "egg", "engine", "eye","face", "farm", "feather", "finger", "fish", "flag", "floor", "fly", "foot", "fork","fowl", "frame", "garden", "girl", "glove", "goat", "gun", "hair", "hammer", "hand" ]
        self.common_verbs = ["accept", "act", "add", "admire", "admit", "advise", "agree", "allow", "announce", "answer", "appear", "approve", "argue", "arrange", "arrive", "ask", "attack", "avoid", "back", "bake", "balance", "ban", "battle", "be", "beat", "become", "beg", "begin", "believe", "belong", "bend", "bet", "bite", "bleed", "blow", "boil", "borrow", "bounce", "bow", "break", "breathe", "bring", "build", "burn", "buy", "calculate", "call", "calm", "can", "care", "carry", "catch", "cause", "change", "charge", "chase", "cheat", "check", "cheer", "chew", "choose", "clap", "clean", "clear", "climb", "close", "collect", "come", "complain", "complete", "concentrate", "connect", "consider", "consist", "contain", "continue", "copy", "correct", "cost", "count", "cover", "crack", "crash", "crawl", "create", "cry", "cut", "dance", "dare", "deal", "decay", "decide", "decorate", "delay", "deliver", "depend", "describe", "desert", "deserve", "destroy", "detect", "develop", "dig", "disagree", "disappear", "discover", "discuss", "divide", "do", "double", "doubt", "drag", "drain", "draw", "dream", "dress", "drink", "drive", "drop", "dry", "earn", "eat", "educate", "empty", "end", "enjoy", "enter", "entertain", "escape", "examine", "exist", "expand", "expect", "explain", "explode", "extend", "face", "fade", "fail", "fall", "feed", "feel", "fight", "fill", "find", "finish", "fit", "fix", "flash", "float", "flood", "flow", "fly", "fold", "follow", "fool", "force", "forget", "forgive", "form", "found", "freeze", "fry", "gather", "gaze", "get", "give", "glow", "go", "grab", "grow", "guard", "guess", "guide", "happen", "hate", "have", "hear", "help", "hide", "hit", "hold", "hope", "hug", "hurt", "identify", "imagine", "impress", "improve", "include", "increase", "inform", "invent", "invite", "jump", "keep", "kick", "kiss", "knock", "know", "land", "laugh", "lead", "learn", "leave", "lend", "let", "lie", "lift", "light", "like", "listen", "live", "look", "lose", "love", "make", "march", "mark", "matter", "mean", "measure", "meet", "melt", "memorize", "mend", "mention", "mind", "miss", "mix", "move", "must", "name", "need", "nod", "note", "notice", "number", "obey", "occur", "offer", "open", "order", "own", "pack", "paint", "park", "pass", "pause", "pay", "peel", "perform", "permit", "pick", "pinch", "plan", "plant", "play", "please", "point", "polish", "pour", "practice", "praise", "pray", "preach", "prepare", "press", "pretend", "prevent", "print", "produce", "promise", "protect", "prove", "pull", "push", "put", "question", "queue", "race", "rain", "raise", "reach", "read", "realize", "receive", "recognize", "record", "reduce", "reflect", "refuse", "regret", "reign", "reject", "relax", "release", "rely", "remain", "remember", "remind", "remove", "repair", "repeat", "replace", "reply", "report", "require", "rest", "return", "ride", "ring", "rise", "risk", "rob", "roll", "rot", "rub", "rule", "run", "rush", "save", "say", "scratch", "scream", "see", "sell", "send", "sense", "serve"]

    def sample(self, no_repeat=False):
        words_per_sentence = random.randint(3, 8)  # Choose length randomly each time
        sentence = []
        
        for i in range(words_per_sentence):
            if random.random() < 0.6:
                word = random.choice(self.common_nouns)
            else:
                word = random.choice(self.common_verbs)
            
            if i == 0:
                word = word.capitalize()
            
            sentence.append(word)
        
        return " ".join(sentence)

class RandomSentences(SamplingPolicy):
    def __init__(self, basemodel_names, filepath):
        # raise if filepath is None or if file does not exist
        if filepath is None or not os.path.exists(filepath):
            raise ValueError("Filepath must be provided and the file must exist")
        self.filepath = filepath
        super().__init__(basemodel_names, filepath)  # Call parent's init after setting filepath


    def setup(self):
        # read file line by line
        with open(self.filepath, 'r') as f:
            self.sentences = f.readlines()
        self.sentences = [s.strip() for s in self.sentences if len(s.strip()) > 0]

    def sample(self, no_repeat = False):
        sentence = orig_sentence = random.choice(self.sentences)
        if no_repeat:
            self.sentences.remove(orig_sentence)
        return sentence


def produce_prohibited_tokens(sampler, models, no_sentences=100):
    prohibited_tokens = set()
    freq_threshold = 0.1

    sampler_name = sampler.__class__.__name__.lower()  # get the sampler's class name

    for mn in models:
        pickle_file = f'{cached_tokens_folder}/prohibited_tokens_{sampler_name}_{model_name_to_file(mn)}.pkl'
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as file:
                prohibited_tokens = prohibited_tokens.union(pickle.load(file))
            continue
        
        print(f'Producing prohibited tokens for {mn} using {sampler_name}', flush=True)

        device = get_device(random)
        tokenizer = AutoTokenizer.from_pretrained(mn, clean_up_tokenization_spaces=False, trust_remote_code=False, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(mn, trust_remote_code=False, local_files_only=True).to(device)

        token_freq = dict()
        for i in range(no_sentences):
            seed_prompt = sampler.sample()  # Use the sampler's own sampling method
            _, _, tokens = get_model_output(device, model, tokenizer, seed_prompt, top_k=1)
            word = tokens[0]
            if word not in token_freq:
                token_freq[word] = 0
            token_freq[word] += 1

        model_prohibited_tokens = set()
        for word in token_freq:
            if token_freq[word] > freq_threshold * no_sentences:
                model_prohibited_tokens.add(word)
                print(f'\t{word:20s} : {token_freq[word]/no_sentences:4.2f}', flush=True)

        entropy = 0
        for word in token_freq:
            p = token_freq[word] / no_sentences
            entropy += p * math.log(p)

        print(f'\tEntropy: {entropy}', flush=True)
        print(f'\tFound {len(model_prohibited_tokens)} prohibited tokens: {model_prohibited_tokens}', flush=True)
        with open(pickle_file, 'wb') as file:
            pickle.dump(model_prohibited_tokens, file)

        prohibited_tokens = prohibited_tokens.union(model_prohibited_tokens)

    return prohibited_tokens






SAMPLING_POLICIES = {
    'random_tokens': RandomTokensSampling,
    'random_words': RandomWords,
    'random_sentences': RandomSentences
}



def get_sampler(parents, sampling_policy, samp_file, print_debug=False):
    print(f'\nSampling policy: {sampling_policy}')
    if sampling_policy not in SAMPLING_POLICIES:
        print(f"Error: Unknown sampling policy {sampling_policy}")
        print(f"Available policies: {list(SAMPLING_POLICIES.keys())}")
        sys.exit(1)
    sampler = SAMPLING_POLICIES[sampling_policy](parents, filepath=samp_file if sampling_policy=='random_sentences' else None)
    if print_debug:
        print('Sample prompts:')
        for _ in range(5):
            prompt = sampler.sample()
            print(f"\t{prompt}")
        print('', flush=True)

    return sampler
