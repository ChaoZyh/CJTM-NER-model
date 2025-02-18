from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.processing_tag_boundary, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.processing_tag_boundary, config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
#all_results  acc:0.9543731477014267, pre:0.5601851851851852, recall:0.622107969151671, f1:0.5895249695493302