import sys
sys.path.insert(0, "/Users/ivan/Dropbox/Code/soco2/")
from soco.classifier import *
from soco.scripts.data import write_data, DataFeature
import csv


EVAL_DATA_DIR = os.path.join(BASE_DIR, '../data/gender/eval')


def get_eval_data():
    texts, labels_index, labels = process_text()
    sequences, word_index = vectorize(texts)
    data, labels = label(sequences, labels)
    return shuffle_master_set(data, labels)


def clean_data_dir(build_dir: str = "../data/gender"):
    import shutil
    if os.path.isdir(f"{build_dir}/simple"):
        shutil.rmtree(f"{build_dir}/simple")


def build_data_dirs(data_path: str = "../data/gender"):
    with open(f"../data/gender-classifier-DFE-791531.csv", 'rt') as file:
        reader = csv.reader(file, delimiter=",")
        data = []
        for row in reader:
            data.append(row)

        def dir_check(d):
            if not os.path.isdir(d):
                os.mkdir(d)

        dir_check(f"{data_path}/simple")
        dir_check(f"{data_path}/simple/f")
        dir_check(f"{data_path}/simple/m")
        dir_check(f"{data_path}/simple/b")

        feature_list = [
            DataFeature.DESCRIPTION,
            DataFeature.NAME,
            DataFeature.SIDEBAR_COLOR,
            DataFeature.TEXT
        ]
        write_data(data, mode="SIMPLE", feature_list=feature_list)


def eval_util():
    x, y = get_eval_data()
    model = load_from_json()
    loss_func = 'categorical_crossentropy'
    optimizer = 'rmsprop'
    metrics = ['acc']
    print(f"\nLoss Function: {loss_func}"
          f"\nOptimizer:     {optimizer}"
          f"\nMetrics:       {metrics}")
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=metrics)
    return model.evaluate(x, y, verbose=0)