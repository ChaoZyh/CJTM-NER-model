import os


from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_tags_boundary = load_vocab(self.filename_tags_boundary)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords         = len(self.vocab_words)
        self.nchars         = len(self.vocab_chars)
        self.ntags          = len(self.vocab_tags)
        self.ntags_boundary = len(self.vocab_tags_boundary)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)
        self.processing_tag_boundary = get_processing_word(self.vocab_tags_boundary,
                                                  lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)
        self.char_embeddings = (get_trimmed_glove_vectors(self.filename_char_trimmed)
                           if self.use_pretrained else None)


    # general config
    dir_output = r"results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 100
    dim_char = 26

    # glove files
    filename_glove = "data/weibo/word_model.txt"
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/weibo/word2vec.npz"
    filename_char_W2V = "data/weibo/Onehot"
    filename_char_trimmed = "data/weibo/Onehot.npz"
    use_pretrained = True

    # dataset
    five_stroke_ = "./data/weibo/86five_stroke.json"
    five_stroke = "./data/weibo/86five_stroke1.json"
    filename_dev = "data/weibo/weiboNER_2nd_conll.dev"
    filename_test = "data/weibo/weiboNER_2nd_conll.test"
    filename_train = "data/weibo/weiboNER_2nd_conll.train"

    #filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/weibo/words.txt"
    filename_tags = "data/weibo/tags.txt"
    filename_tags_boundary = "data/weibo/tags_boundary.txt"
    filename_chars = "data/weibo/chars.txt"

    # training
    train_embeddings = False
    nepochs          = 80
    dropout          = 0.5
    batch_size       = 8
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.99
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 80 # lstm on chars
    hidden_size_lstm = 400 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU
    use_muti = True
