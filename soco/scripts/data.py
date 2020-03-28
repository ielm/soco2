import random
import csv


class DataFeature:
    UNIT_ID_ = 0
    GOLDEN_ = 1
    UNIT_STATE_ = 2
    TRUSTED_JUDGMENTS_ = 3
    LAST_JUDGMENT_AT_ = 4
    GENDER = 5
    GENDER_CONFIDENCE = 6
    PROFILE_YN = 7
    PROFILE_YN_CONFIDENCE = 8
    CREATED = 9
    DESCRIPTION = 10
    FAV_NUMBER = 11
    GENDER_GOLD = 12
    LINK_COLOR = 13
    NAME = 14
    PROFILE_YN_GOLD = 15
    PROFILEIMAGE = 16
    RETWEET_COUNT = 17
    SIDEBAR_COLOR = 18
    TEXT = 19
    TWEET_COORD = 20
    TWEET_COUNT = 21
    TWEET_CREATED = 22
    TWEET_ID = 23
    TWEET_LOCATION = 24
    USER_TIMEZONE = 25


def create_file(filepath: str, filename: str, content: str, debug: bool = False, ghost: bool = False):
    filepath = f"../{filepath}"
    log = lambda fp, fn, c: print(f"\nFILEPATH: {fp}\nFILENAME: {fn}\nCONTENT: {c}")
    if ghost:
        print("\n\nGHOST MODE: NOT WRITING TO FILE")
        log(filepath, filename, content)
    else:
        if debug:
            log(filepath, filename, content)
        with open(f"{filepath}{filename}", "w") as file:
            file.write(content)


def data_generator(_data: list,
                   feature_list: list = [],
                   separate_for_training: bool = False):
    if not feature_list:
        feature_list = [DataFeature.TEXT]

    def aggregate_data(r: list, f_list: list):
        content = ""
        for feature in f_list:
            content = f"{content}{r[feature]}"
        return content

    def build(_data: list) -> tuple:
        _f, _m, _b = [], [], []
        fi, mi, bi = 0, 0, 0
        for r in _data:
            if r[DataFeature.GENDER] == "female":
                _f.append((f"{r[DataFeature.UNIT_ID_]}",
                           f"{aggregate_data(r, feature_list)}"))
                fi += 1
            if r[DataFeature.GENDER] == "male":
                _m.append((f"{r[DataFeature.UNIT_ID_]}",
                           f"{aggregate_data(r, feature_list)}"))
                mi += 1
            if r[DataFeature.GENDER] == "brand":
                _b.append((f"{r[DataFeature.UNIT_ID_]}",
                           f"{aggregate_data(r, feature_list)}"))
                bi += 1
        return _f, _m, _b

    random.shuffle(_data)

    if separate_for_training:
        test = _data[int(len(_data) * 0.8):]
        train = _data[:int(len(_data) * 0.8)]

        ftrain, mtrain, btrain = build(train)
        ftest, mtest, btest = build(test)

        return ftrain, ftest, mtrain, mtest, btrain, btest
    else:
        return build(_data)


def write_data(_data: list, mode: str = "SIMPLE", feature_list: list = []):
    """Write all the data files to disk"""
    def build_class_set(l: list, fp: str, debug: bool = False):
        for datapoint in l:
            create_file(f"{fp}", datapoint[0], datapoint[1], debug=debug)

    if not feature_list:
        feature_list = [DataFeature.TEXT]
    _data = _data[1:]

    if mode == "SIMPLE":
        f, m, b = data_generator(_data=_data, feature_list=feature_list)
        filepath = "data/gender/simple/"
        build_class_set(f, f"{filepath}f/")
        build_class_set(m, f"{filepath}m/")
        build_class_set(b, f"{filepath}b/")
    elif mode == "SPLIT":
        ftrain, ftest, mtrain, \
            mtest, btrain, btest = data_generator(_data, separate_for_training=True)
        filepath = "data/simple_set/"
        build_class_set(ftrain, f"{filepath}/train/f/", debug=True)
        build_class_set(mtrain, f"{filepath}/train/m/", debug=True)
        build_class_set(btrain, f"{filepath}/train/b/", debug=True)
        build_class_set(ftest, f"{filepath}/test/f/", debug=True)
        build_class_set(mtest, f"{filepath}/test/m/", debug=True)
        build_class_set(btest, f"{filepath}/test/b/", debug=True)


if __name__ == '__main__':
    with open('../../data/gender-classifier-DFE-791531.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            data.append(row)

        write_data(data)
